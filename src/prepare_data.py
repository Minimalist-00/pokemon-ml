
import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np

# Configuration
SOURCE_DIR = 'data/pokemon/pokemon'
DATASET_DIR = 'dataset'
IMG_SIZE = (224, 224)
NUM_CLASSES = 150
AUGMENT_FACTOR = 20 # Generate 20 images per class
VAL_SPLIT = 0.2

def create_dataset():
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    os.makedirs(os.path.join(DATASET_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'val'), exist_ok=True)

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )

    print(f"Generating dataset for {NUM_CLASSES} classes...")

    for i in range(1, NUM_CLASSES + 1):
        filename = f"{i}.png"
        src_path = os.path.join(SOURCE_DIR, filename)
        
        if not os.path.exists(src_path):
            # Try to find variants if base doesn't exist (unlikely for 1-150 but possible)
            # Actually, let's just skip if not found and print warning
            print(f"Warning: {filename} not found.")
            # Try to find ANY file starting with i? No, complicate logic.
            # But let's check for jpg too just in case
            jpg_path = os.path.join('data/pokemon_jpg/pokemon_jpg', f"{i}.jpg")
            if os.path.exists(jpg_path):
                src_path = jpg_path
            else:
                continue

        try:
            img = load_img(src_path, target_size=IMG_SIZE) # Resize too
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            # Create class directories
            class_name = str(i) # Using ID as class name
            train_dir = os.path.join(DATASET_DIR, 'train', class_name)
            val_dir = os.path.join(DATASET_DIR, 'val', class_name)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            # Save check: save original image first?
            # Actually let's just generate N augmented images
            # Split: val first
            
            count = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=None, save_prefix='aug', save_format='jpg'):
                aug_img = batch[0].astype('uint8')
                # Convert back to image to save? Or use save_to_dir of flow?
                # flow's save_to_dir saves randomly. We want control.
                # Actually we can just use tf.keras.preprocessing.image.save_img
                
                if count < int(AUGMENT_FACTOR * VAL_SPLIT):
                   target_dir = val_dir
                else:
                   target_dir = train_dir

                save_path = os.path.join(target_dir, f"{i}_{count}.jpg")
                tf.keras.preprocessing.image.save_img(save_path, aug_img)
                
                count += 1
                if count >= AUGMENT_FACTOR:
                    break
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("Dataset generation complete.")

if __name__ == "__main__":
    create_dataset()
