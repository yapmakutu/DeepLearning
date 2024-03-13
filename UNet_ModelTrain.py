import os
import numpy as np
from glob import glob
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.layers import Layer, Conv2D, Dropout, UpSampling2D, concatenate, Add, Multiply, Input, MaxPool2D, \
    BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.metrics import MeanIoU


# Function to load a single image
def load_image(image_path, size):
    img = load_img(image_path, target_size=(size, size))
    img = img_to_array(img) / 255.0
    return img


# Function to load images and masks
def load_dataset(image_dir, mask_dirs, image_size):
    image_list = []
    mask_list = []
    for mask_dir in mask_dirs:
        image_files = glob(os.path.join(image_dir, mask_dir, '*.png'))
        for image_file in image_files:
            if '_mask' in image_file:
                continue
            image = load_image(image_file, image_size)
            image_list.append(image)

            mask_files = glob(image_file.replace('.png', '_mask*.png'))
            mask = np.zeros((image_size, image_size, 1), dtype=np.float32)
            for mask_file in mask_files:
                mask_img = load_image(mask_file, image_size)
                mask_img = np.expand_dims(mask_img[:, :, 0], axis=-1)
                mask = np.maximum(mask, mask_img)
            mask_list.append(mask)
    return np.array(image_list), np.array(mask_list)


# image and mask directories
IMAGE_DIR = r'C:\Users\AhmetSahinCAKIR\Desktop\Ahmet\Bitirme\Dataset_BUSI_with_GT'
MASK_DIRS = ['benign', 'malignant', 'normal']
IMAGE_SIZE = 256

# Load the dataset
images, masks = load_dataset(IMAGE_DIR, MASK_DIRS, IMAGE_SIZE)


class EncoderBlock(Layer):

    def __init__(self, filters, rate, pooling=True, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)

        self.filters = filters
        self.rate = rate
        self.pooling = pooling

        self.c1 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu',
                         kernel_initializer='he_normal')
        self.drop = Dropout(rate)
        self.c2 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu',
                         kernel_initializer='he_normal')
        self.pool = MaxPool2D()

    def call(self, inputs, **kwargs):
        x = self.c1(inputs)
        x = self.drop(x)
        x = self.c2(x)
        if self.pooling:
            y = self.pool(x)
            return y, x
        else:
            return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            'rate': self.rate,
            'pooling': self.pooling
        }


class DecoderBlock(Layer):

    def __init__(self, filters, rate, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)

        self.filters = filters
        self.rate = rate

        self.up = UpSampling2D()
        self.net = EncoderBlock(filters, rate, pooling=False)

    def call(self, inputs, **kwargs):
        X, skip_X = inputs
        x = self.up(X)
        c_ = concatenate([x, skip_X])
        x = self.net(c_)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            'rate': self.rate,
        }


class AttentionGate(Layer):

    def __init__(self, filters, bn, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)

        self.filters = filters
        self.bn = bn

        self.normal = Conv2D(filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.down = Conv2D(filters, kernel_size=3, strides=2, padding='same', activation='relu',
                           kernel_initializer='he_normal')
        self.learn = Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')
        self.resample = UpSampling2D()
        self.BN = BatchNormalization()

    def call(self, inputs, **kwargs):
        X, skip_X = inputs

        x = self.normal(X)
        skip = self.down(skip_X)
        x = Add()([x, skip])
        x = self.learn(x)
        x = self.resample(x)
        f = Multiply()([x, skip_X])
        if self.bn:
            return self.BN(f)
        else:
            return f

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            "bn": self.bn
        }


# Inputs
input_layer = Input(shape=images.shape[-3:])

# Encoder
encoder1 = EncoderBlock(32, 0.1, name="Encoder1")
p1, c1 = encoder1(input_layer)

encoder2 = EncoderBlock(64, 0.1, name="Encoder2")
p2, c2 = encoder2(p1)

encoder3 = EncoderBlock(128, 0.2, name="Encoder3")
p3, c3 = encoder3(p2)

encoder4 = EncoderBlock(256, 0.2, name="Encoder4")
p4, c4 = encoder4(p3)

# Encoding
encoding_block = EncoderBlock(512, 0.3, pooling=False, name="Encoding")
encoding = encoding_block(p4)

# Attention + Decoder
attention1 = AttentionGate(256, bn=True, name="Attention1")
a1 = attention1([encoding, c4])

decoder1 = DecoderBlock(256, 0.2, name="Decoder1")
d1 = decoder1([encoding, a1])

attention2 = AttentionGate(128, bn=True, name="Attention2")
a2 = attention2([d1, c3])

decoder2 = DecoderBlock(128, 0.2, name="Decoder2")
d2 = decoder2([d1, a2])

attention3 = AttentionGate(64, bn=True, name="Attention3")
a3 = attention3([d2, c2])

decoder3 = DecoderBlock(64, 0.1, name="Decoder3")
d3 = decoder3([d2, a3])

attention4 = AttentionGate(32, bn=True, name="Attention4")
a4 = attention4([d3, c1])

decoder4 = DecoderBlock(32, 0.1, name="Decoder4")
d4 = decoder4([d3, a4])

# Output
output_layer = Conv2D(1, kernel_size=1, activation='sigmoid', padding='same')(d4)

# Model
model = Model(
    inputs=[input_layer],
    outputs=[output_layer]
)

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', MeanIoU(num_classes=2)]
)

# callbacks for each stage
callbacks_stage_1 = [
    ModelCheckpoint('model_after_17_epochs.h5', save_best_only=True)
]

callbacks_stage_2 = [
    ModelCheckpoint('model_after_34_epochs.h5', save_best_only=True)
]

callbacks_stage_3 = [
    ModelCheckpoint('final_model.h5', save_best_only=True)
]

# Train the model in stages
# Stage 1
results_1 = model.fit(
    x=images,
    y=masks,
    batch_size=8,
    epochs=17,
    validation_split=0.1,
    callbacks=callbacks_stage_1
)

# Stage 2
results_2 = model.fit(
    x=images,
    y=masks,
    batch_size=8,
    epochs=34,
    validation_split=0.1,
    callbacks=callbacks_stage_2,
    initial_epoch=17
)

# Stage 3
results_3 = model.fit(
    x=images,
    y=masks,
    batch_size=8,
    epochs=51,
    validation_split=0.1,
    callbacks=callbacks_stage_3,
    initial_epoch=34
)

# Compare the results
print("Stage 1 Results:")
print("Loss:", results_1.history['loss'][-1], "Accuracy:", results_1.history['accuracy'][-1], "Validation Loss:",
      results_1.history['val_loss'][-1], "Validation Accuracy:", results_1.history['val_accuracy'][-1])

print("\nStage 2 Results:")
print("Loss:", results_2.history['loss'][-1], "Accuracy:", results_2.history['accuracy'][-1], "Validation Loss:",
      results_2.history['val_loss'][-1], "Validation Accuracy:", results_2.history['val_accuracy'][-1])

print("\nStage 3 Results:")
print("Loss:", results_3.history['loss'][-1], "Accuracy:", results_3.history['accuracy'][-1], "Validation Loss:",
      results_3.history['val_loss'][-1], "Validation Accuracy:", results_3.history['val_accuracy'][-1])
