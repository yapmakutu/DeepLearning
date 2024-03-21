import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Pre-process and loading data
def load_and_preprocess_image(filepath):
    # Load and scale the image
    image = tf.io.read_file(filepath)
    image = tf.image.decode_png(image, channels=1)  # PNG and one channel (gray mode)
    image = tf.image.convert_image_dtype(image, tf.float32)  # Scaling between 0-1
    image = tf.image.resize(image, [image_height, image_width])  # Size fixing
    return image

def prepare_dataset(dataset_path, batch_size, is_training=True):
    # Image data generator
    datagen = ImageDataGenerator(
        rescale=1./255  # Scaling between 0-1
    )

    # Data generator
    generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        color_mode='grayscale',  # for grayscale
        class_mode='binary',  # for classification binary as 'benign' or 'malignant'
        shuffle=is_training
    )
    return generator

# Building model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Using sigmoid for classification as binary
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Parameters
image_height = 256  # Height of image
image_width = 256  # Width of image
batch_size = 32  # Each training step taking 32 examples
dataset_path = r'C:\Dataset_BUSI_with_GT\benign_train'

# Loading dataset and pre_process
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

print(f"Train Accuracy: {accuracy[-1]:.2f}")
print(f"Train Loss: {loss[-1]:.2f}")

model.save('./trained_model_size256.h5')  # Update this path for saving the model

validation_dataset_path = './validation'  # Update this path to your validation dataset's location
validation_generator = prepare_dataset(validation_dataset_path, batch_size, is_training=False)

# Evaluating the model on the validation set
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {val_accuracy:.2f}")
print(f"Validation Loss: {val_loss:.2f}")