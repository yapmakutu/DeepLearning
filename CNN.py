from google.colab import drive
drive.mount('/content/drive')
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Pre_process and loading data
def load_and_preprocess_image(filepath):
    # Görüntüyü yükle ve ölçeklendir
    image = tf.io.read_file(filepath)
    image = tf.image.decode_png(image, channels=1)  # PNG and one channel (gray mode)
    image = tf.image.convert_image_dtype(image, tf.float32)  # Scaling between 0-1
    image = tf.image.resize(image, [image_height, image_width])  # Size fixing
    return image

def prepare_dataset(dataset_path, batch_size,is_training=True):
    # Image data generator
    datagen = ImageDataGenerator(
        rescale=1./255  # Scaling between 0-1
    )


    # data generator
    generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        color_mode='grayscale',  # for graysclae
        class_mode='binary',  # to classification binary as 'bening' or 'malignant'
        shuffle=is_training

    )
    return generator

# Buiding model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 1)),
        MaxPooling2D(2, 2),#reducing the computational burden and helps prevent model overfitting
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Using sigmoid for classification as binary
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # Cross entropy for binary classification
                  metrics=['accuracy'])
    return model

# Parametreler
image_height = 256  # Heigth of image
image_width = 256  # Weidght of image
batch_size = 32  # each training step taking 32 example
dataset_path = '/content/drive/My Drive/training'  # datapath

# loading data set and pre_process
train_generator = prepare_dataset(dataset_path, batch_size)

# Build model
model = build_model()

# Training
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10
)
# Evaluating of training process
accuracy = history.history['accuracy']
loss = history.history['loss']

print(f"Train Correction: {accuracy[-1]:.2f}")
print(f"Train Loss: {loss[-1]:.2f}")


model.save('/content/drive/My Drive/trained_model_size256.h5')
validation_dataset_path = '/content/drive/My Drive/validation'
validation_generator = prepare_dataset(validation_dataset_path, batch_size, is_training=False)

# Evaluating the model on the validation set
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"The correction of correction: {val_accuracy:.2f}")
print(f"Correction loss: {val_loss:.2f}")