import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # Using softmax for multi-class classification
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Parameters
image_height = 256
image_width = 256
batch_size = 32

def prepare_dataset(dataset_path, batch_size, is_training=True):
    datagen = ImageDataGenerator(rescale=1./255)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The directory {dataset_path} does not exist.")

    generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=is_training
    )
    return generator

def train_cnn():
    dataset_path = os.path.join(PROJECT_ROOT, 'Dataset', 'Dataset_BUSI_with_GT_split')

    train_generator = prepare_dataset(os.path.join(dataset_path, 'train'), batch_size)
    validation_generator = prepare_dataset(os.path.join(dataset_path, 'validation'), batch_size, is_training=False)

    input_shape = (image_height, image_width, 1)
    model = build_cnn_model(input_shape)

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )

    accuracy = history.history['accuracy']
    loss = history.history['loss']
    val_accuracy = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    print(f"Train Accuracy: {accuracy[-1]:.2f}")
    print(f"Train Loss: {loss[-1]:.2f}")
    print(f"Validation Accuracy: {val_accuracy[-1]:.2f}")
    print(f"Validation Loss: {val_loss[-1]:.2f}")

    model_save_path = os.path.join(PROJECT_ROOT, 'trained_model_size256.h5')
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    test_generator = prepare_dataset(os.path.join(dataset_path, 'test'), batch_size, is_training=False)
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Test Loss: {test_loss:.2f}")

if __name__ == "__main__":
    train_cnn()
