import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Layer, Conv2D, Dropout, MaxPool2D, UpSampling2D, concatenate
from keras.models import load_model


class EncoderBlock(Layer):
    def __init__(self, filters, rate, pooling=True, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        if filters <= 0:
            raise ValueError("Filters must be a positive integer")
        if not (0.0 <= rate <= 1.0):
            raise ValueError("Rate must be a float between 0.0 and 1.0")

        self.filters = filters
        self.rate = rate
        self.pooling = pooling
        self.c1 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu',
                         kernel_initializer='he_normal')
        self.drop = Dropout(rate)
        self.c2 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu',
                         kernel_initializer='he_normal')
        self.pool = MaxPool2D()

    def call(self, X, **kwargs):
        if not isinstance(X, tf.Tensor):
            raise TypeError("Input must be a TensorFlow tensor")

        x = self.c1(X)
        x = self.drop(x)
        x = self.c2(x)
        if self.pooling:
            y = self.pool(x)
            return y, x
        else:
            return x


class DecoderBlock(Layer):
    def __init__(self, filters, rate, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.filters = filters
        self.rate = rate
        self.up = UpSampling2D()
        self.net = EncoderBlock(filters, rate, pooling=False)

    def call(self, X, **kwargs):
        X, skip_X = X
        x = self.up(X)
        c_ = concatenate([x, skip_X])
        x = self.net(c_)
        return x


class DeepLearning:
    def __init__(self, unet_model):
        self.segmentation_model = load_model(unet_model,
                                             custom_objects={
                                                 'EncoderBlock': EncoderBlock,
                                                 'DecoderBlock': DecoderBlock
                                             })

    def segment(self, image_path):
        test_image = cv2.imread(image_path)
        resized_image = cv2.resize(test_image, (256, 256))
        resized_image = np.expand_dims(resized_image, axis=0)

        predicted_mask = self.segmentation_model.predict(resized_image)
        predicted_mask_resized = cv2.resize(predicted_mask[0], (test_image.shape[1], test_image.shape[0]))

        _, binary_mask = cv2.threshold(predicted_mask_resized, 0.5, 1, cv2.THRESH_BINARY)
        binary_mask = (binary_mask * 255).astype(np.uint8)

        # Save the visual comparison
        self.save_visual_comparison(test_image, binary_mask, image_path)

        return binary_mask

    @staticmethod
    def save_visual_comparison(original_image, predicted_mask, original_path):
        mask_overlay = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2BGR)  # Convert mask back to BGR for overlay
        mask_overlay = np.where(mask_overlay > 0, (0, 0, 255), original_image)  # Overlay mask in red

        combined_image = np.concatenate((original_image, mask_overlay), axis=1)  # Combine images side-by-side
        save_path = original_path.replace('.png', '_comparison.png')  # Modify the original path for saving

        cv2.imwrite(save_path, combined_image)  # Save the combined image


if __name__ == "__main__":
    # Load the U-Net model
    unet_model = r"C:\Users\AhmetSahinCAKIR\Desktop\Ahmet\Bitirme\Modeller\final_model.h5"
    image_path = r"C:\Users\AhmetSahinCAKIR\Desktop\Test Result\benign (1).png"

    custom_objects = {
        'EncoderBlock': EncoderBlock,
        'DecoderBlock': DecoderBlock
    }

    # Load the model with the custom objects specified
    model = load_model(unet_model, custom_objects=custom_objects)

    # Perform segmentation
    deep_learning = DeepLearning(unet_model)
    predicted_mask = deep_learning.segment(image_path)
