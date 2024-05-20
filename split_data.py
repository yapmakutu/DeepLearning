import os
import shutil
import random

# Kaynak ve hedef dizinler
SOURCE_DIR = 'Dataset/Dataset_BUSI_with_GT'
TARGET_DIR = 'Dataset/Dataset_BUSI_with_GT_split'

# Sınıf isimleri
classes = ['benign', 'malignant', 'normal']

# Yüzdeler
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

def split_data(class_name):
    # Görüntü ve maske dosyalarının listelenmesi
    class_dir = os.path.join(SOURCE_DIR, class_name)
    images = [f for f in os.listdir(class_dir) if not ('_mask' in f or '_mask_' in f)]
    masks = [f for f in os.listdir(class_dir) if '_mask' in f or '_mask_' in f]

    # Sıralama ve karıştırma
    images.sort()
    masks.sort()
    random.seed(42)
    random.shuffle(images)

    # Bölme işlemi
    total_images = len(images)
    train_count = int(total_images * train_ratio)
    validation_count = int(total_images * validation_ratio)

    train_images = images[:train_count]
    validation_images = images[train_count:train_count + validation_count]
    test_images = images[train_count + validation_count:]

    # Hedef dizinlerin oluşturulması
    for split in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(TARGET_DIR, split, class_name, 'images'), exist_ok=True)
        os.makedirs(os.path.join(TARGET_DIR, split, class_name, 'masks'), exist_ok=True)

    # Dosyaların taşınması
    def move_files(split, images, masks):
        for img in images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(TARGET_DIR, split, class_name, 'images', img))
        for mask in masks:
            shutil.copy(os.path.join(class_dir, mask), os.path.join(TARGET_DIR, split, class_name, 'masks', mask))

    def get_associated_masks(images, masks):
        associated_masks = []
        for img in images:
            base_name = os.path.splitext(img)[0]
            img_masks = [mask for mask in masks if base_name in mask]
            associated_masks.extend(img_masks)
        return associated_masks

    # Her split için maskeleri ayır
    train_masks = get_associated_masks(train_images, masks)
    validation_masks = get_associated_masks(validation_images, masks)
    test_masks = get_associated_masks(test_images, masks)

    move_files('train', train_images, train_masks)
    move_files('validation', validation_images, validation_masks)
    move_files('test', test_images, test_masks)

# Tüm sınıflar için işlemi tekrarlayın
for class_name in classes:
    split_data(class_name)

print("Data splitting complete.")
