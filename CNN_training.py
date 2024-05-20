import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from CNN_model import build_cnn_model
import tensorflow as tf

# Parameters
image_height = 256
image_width = 256
batch_size = 16

def prepare_dataset(dataset_path, batch_size, is_training=True):
    if is_training:
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        datagen = ImageDataGenerator(rescale=1./255)

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
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(PROJECT_ROOT, 'Dataset', 'Dataset_BUSI_with_GT_split')

    EPOCHS = 20

    train_generator = prepare_dataset(os.path.join(dataset_path, 'train'), batch_size)
    validation_generator = prepare_dataset(os.path.join(dataset_path, 'validation'), batch_size, is_training=False)

    # Check class indices
    print(f"Train classes: {train_generator.class_indices}")
    print(f"Validation classes: {validation_generator.class_indices}")

    input_shape = (image_height, image_width, 1)
    model = build_cnn_model(input_shape)

    steps_per_epoch = len(train_generator)
    validation_steps = len(validation_generator)

    callbacks = [
        ModelCheckpoint(os.path.join(PROJECT_ROOT, 'best_cnn_model.h5'), save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    accuracy = history.history['accuracy']
    loss = history.history['loss']
    val_accuracy = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    print(f"Train Accuracy: {accuracy[-1]:.2f}")
    print(f"Train Loss: {loss[-1]:.2f}")
    print(f"Validation Accuracy: {val_accuracy[-1]:.2f}")
    print(f"Validation Loss: {val_loss[-1]:.2f}")

    model_save_path = os.path.join(PROJECT_ROOT, 'trained_cnn_model_size256.h5')
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    test_generator = prepare_dataset(os.path.join(dataset_path, 'test'), batch_size, is_training=False)
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Test Loss: {test_loss:.2f}")

if __name__ == "__main__":
    train_cnn()
