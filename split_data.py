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
    images = [f for f in os.listdir(class_dir) if not f.endswith('_mask.png')]
    masks = [f for f in os.listdir(class_dir) if f.endswith('_mask.png')]

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
    def move_files(split, images):
        for img in images:
            mask = img.replace('.png', '_mask.png')
            shutil.copy(os.path.join(class_dir, img), os.path.join(TARGET_DIR, split, class_name, 'images', img))
            if mask in masks:
                shutil.copy(os.path.join(class_dir, mask), os.path.join(TARGET_DIR, split, class_name, 'masks', mask))

    move_files('train', train_images)
    move_files('validation', validation_images)
    move_files('test', test_images)


# Tüm sınıflar için işlemi tekrarlayın
for class_name in classes:
    split_data(class_name)

print("Data splitting complete.")
