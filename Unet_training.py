import os
import numpy as np
from glob import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from Unet_model import build_unet_model  # Make sure this imports your U-Net model correctly

# Function to load a single image
def load_image(image_path, size):
    img = load_img(image_path, target_size=(size, size), color_mode='grayscale')
    img = img_to_array(img) / 255.0
    return img

# Function to load images and masks
def load_dataset(image_dir, mask_dir, image_size):
    image_list = []
    mask_list = []
    image_files = glob(os.path.join(image_dir, '*.png'))
    for image_file in image_files:
        image = load_image(image_file, image_size)
        image_list.append(image)

        mask_file = os.path.join(mask_dir, os.path.basename(image_file).replace('.png', '_mask.png'))
        if os.path.exists(mask_file):
            mask_img = load_image(mask_file, image_size)
            mask_img = np.expand_dims(mask_img[:, :, 0], axis=-1)
            mask_list.append(mask_img)
        else:
            print(f"Warning: Mask for {image_file} not found. Expected path: {mask_file}")
            mask_list.append(np.zeros((image_size, image_size, 1), dtype=np.float32))
    return np.array(image_list), np.array(mask_list)

def train_unet():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(PROJECT_ROOT, 'Dataset', 'Dataset_BUSI_with_GT_split')

    IMAGE_SIZE = 256
    BATCH_SIZE = 8

    def get_split_data(split):
        images, masks = [], []
        for class_name in ['benign', 'malignant', 'normal']:
            img_dir = os.path.join(dataset_path, split, class_name, 'images')
            mask_dir = os.path.join(dataset_path, split, class_name, 'masks')
            img_list, mask_list = load_dataset(img_dir, mask_dir, IMAGE_SIZE)
            images.extend(img_list)
            masks.extend(mask_list)
        return np.array(images), np.array(masks)

    train_images, train_masks = get_split_data('train')
    validation_images, validation_masks = get_split_data('validation')

    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)
    model = build_unet_model(input_shape)

    # Callbacks
    callbacks = [
        ModelCheckpoint(os.path.join(PROJECT_ROOT, 'best_model.h5'), save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_gen = datagen.flow(train_images, train_masks, batch_size=BATCH_SIZE)
    val_gen = ImageDataGenerator().flow(validation_images, validation_masks, batch_size=BATCH_SIZE)

    # Train the model
    model.fit(
        train_gen,
        steps_per_epoch=len(train_images) // BATCH_SIZE,
        epochs=100,
        validation_data=val_gen,
        validation_steps=len(validation_images) // BATCH_SIZE,
        callbacks=callbacks
    )

if __name__ == "__main__":
    train_unet()