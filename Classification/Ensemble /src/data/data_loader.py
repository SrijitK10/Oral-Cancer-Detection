"""
Data loading and preprocessing utilities for the Oral Cancer Classification Pipeline.
"""

import os
import random
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from typing import Dict, Tuple, List, Any


def count_images_in_directory(directory: str) -> Dict[str, int]:
    """
    Count the number of images in each class directory.
    
    Args:
        directory: Path to the dataset directory containing class subdirectories
        
    Returns:
        Dictionary mapping class names to image counts
    """
    class_counts = {}
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            # Count valid image files (jpg, jpeg, png)
            valid_extensions = ['.jpg', '.jpeg', '.png']
            count = sum(1 for f in os.listdir(class_dir) if 
                        os.path.isfile(os.path.join(class_dir, f)) and 
                        any(f.lower().endswith(ext) for ext in valid_extensions))
            class_counts[class_name] = count
    return class_counts


def analyze_dataset(config: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], int]:
    """
    Analyze the dataset structure and distribution.
    
    Args:
        config: Configuration dictionary containing dataset parameters
        
    Returns:
        Tuple of (train_counts, test_counts, val_counts, max_count)
    """
    base_dir = config["dataset"]["base_dir"]
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    val_dir = os.path.join(base_dir, 'val')
    
    train_counts = count_images_in_directory(train_dir)
    test_counts = count_images_in_directory(test_dir)
    val_counts = count_images_in_directory(val_dir)
    
    print("Training set class distribution:")
    for class_name, count in train_counts.items():
        print(f"  {class_name}: {count} images")
    
    print("\nTest set class distribution:")
    for class_name, count in test_counts.items():
        print(f"  {class_name}: {count} images")
    
    print("\nValidation set class distribution:")
    for class_name, count in val_counts.items():
        print(f"  {class_name}: {count} images")
    
    max_count = max(train_counts.values())
    min_count = min(train_counts.values())
    print(f"\nImbalance in training set: max={max_count}, min={min_count}, ratio={max_count/min_count:.2f}")
    
    return train_counts, test_counts, val_counts, max_count


def create_augmentation_generator(config: Dict[str, Any]) -> ImageDataGenerator:
    """
    Create an image data generator with augmentation settings.
    
    Args:
        config: Configuration dictionary containing augmentation parameters
        
    Returns:
        Configured ImageDataGenerator for augmentation
    """
    aug_config = config["augmentation"]
    
    return ImageDataGenerator(
        rotation_range=aug_config["rotation_range"],
        width_shift_range=aug_config["width_shift_range"],
        height_shift_range=aug_config["height_shift_range"],
        shear_range=aug_config["shear_range"],
        zoom_range=aug_config["zoom_range"],
        horizontal_flip=aug_config["horizontal_flip"],
        vertical_flip=aug_config["vertical_flip"],
        fill_mode=aug_config["fill_mode"],
        brightness_range=aug_config["brightness_range"],
        channel_shift_range=aug_config["channel_shift_range"]
    )


def generate_augmented_images(config: Dict[str, Any], train_counts: Dict[str, int], target_count: int) -> str:
    """
    Generate augmented images to balance classes.
    
    Args:
        config: Configuration dictionary
        train_counts: Dictionary of class counts in training set
        target_count: Target count for each class after augmentation
        
    Returns:
        Path to the augmented dataset directory
    """
    base_dir = config["dataset"]["base_dir"]
    train_dir = os.path.join(base_dir, 'train')
    augmented_dir = os.path.join(base_dir, 'augmented_train')
    img_height = config["dataset"]["img_height"]
    img_width = config["dataset"]["img_width"]
    
    print("\nGenerating augmented images to balance classes...")
    
    # Create directory for augmented data if it doesn't exist
    if os.path.exists(augmented_dir):
        shutil.rmtree(augmented_dir)
    os.makedirs(augmented_dir)
    
    # Copy the original directory structure
    for class_name in train_counts.keys():
        os.makedirs(os.path.join(augmented_dir, class_name), exist_ok=True)
        
        # First, copy all original images
        source_dir = os.path.join(train_dir, class_name)
        dest_dir = os.path.join(augmented_dir, class_name)
        
        image_files = [f for f in os.listdir(source_dir) if 
                    os.path.isfile(os.path.join(source_dir, f)) and 
                    any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])]
        
        for img_file in image_files:
            shutil.copy2(os.path.join(source_dir, img_file), 
                        os.path.join(dest_dir, img_file))
        
        # Calculate how many more images we need
        current_count = train_counts[class_name]
        if current_count >= target_count:
            print(f"  {class_name}: Already has {current_count} images, no augmentation needed")
            continue
        
        num_to_generate = target_count - current_count
        print(f"  {class_name}: Generating {num_to_generate} additional images")
        
        # Set up image data generator for this class
        aug_gen = create_augmentation_generator(config)
        
        # Create a temporary generator to access a few images
        temp_gen = ImageDataGenerator().flow_from_directory(
            train_dir,
            target_size=(img_height, img_width),
            batch_size=len(image_files),
            classes=[class_name],
            shuffle=True,
            class_mode='categorical'
        )
        
        # Get all original images
        original_images, _ = next(temp_gen)
        
        # Generate and save augmented images
        aug_img_idx = 0
        pbar = tqdm(total=num_to_generate)
        
        while aug_img_idx < num_to_generate:
            # Choose a random image from originals
            img_idx = random.randint(0, len(original_images) - 1)
            img = original_images[img_idx:img_idx+1]
            
            # Get an augmented version
            augmented = next(aug_gen.flow(
                img,
                batch_size=1,
                shuffle=True
            ))
            
            # Save the augmented image
            aug_filename = f"aug_{aug_img_idx}_{os.path.basename(image_files[img_idx % len(image_files)])}"
            tf.keras.preprocessing.image.save_img(
                os.path.join(dest_dir, aug_filename),
                augmented[0]
            )
            
            aug_img_idx += 1
            pbar.update(1)
        
        pbar.close()
    
    return augmented_dir


def create_data_generators(config: Dict[str, Any], use_augmented: bool = True) -> Tuple:
    """
    Create data generators for training, validation, and testing.
    
    Args:
        config: Configuration dictionary
        use_augmented: Whether to use augmented training data
        
    Returns:
        Tuple of (train_generator, validation_generator, test_generator)
    """
    base_dir = config["dataset"]["base_dir"]
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    val_dir = os.path.join(base_dir, 'val')
    augmented_dir = os.path.join(base_dir, 'augmented_train')
    
    img_height = config["dataset"]["img_height"]
    img_width = config["dataset"]["img_width"]
    batch_size = config["dataset"]["batch_size"]
    
    # For training data - with augmentation during training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # For validation and test data - just rescaling
    test_val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Choose the appropriate training directory
    train_dir_to_use = augmented_dir if use_augmented and os.path.exists(augmented_dir) else train_dir
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir_to_use,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = test_val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_val_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator