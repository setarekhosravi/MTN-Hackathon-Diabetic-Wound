#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 15 for Hakathon
    @author: STRH
    Generate data with augmentation
"""

import os
import cv2
import albumentations as A
from tqdm import tqdm

# Paths to your dataset
IMAGE_DIR = "Datasets/azh_wound_care_center_dataset_patches/train/images"
MASK_DIR = "Datasets/azh_wound_care_center_dataset_patches/train/labels"
OUTPUT_IMAGE_DIR = "Pytorch-UNet/data/imgs"
OUTPUT_MASK_DIR = "Pytorch-UNet/data/masks"

# Ensure output directories exist
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# Define augmentation pipeline
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5)
])

# Function to apply augmentations
def augment_and_save(image_path, mask_path, output_image_dir, output_mask_dir, num_augmentations=5):
    # Load image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Assuming mask is grayscale

    base_name = os.path.basename(image_path).split('.')[0]

    for i in range(num_augmentations):
        # Apply augmentations
        augmented = augmentation_pipeline(image=image, mask=mask)
        aug_image = augmented["image"]
        aug_mask = augmented["mask"]

        # Save augmented image and mask
        aug_image_path = os.path.join(output_image_dir, f"{base_name}_aug_{i}.png")
        aug_mask_path = os.path.join(output_mask_dir, f"{base_name}_aug_{i}.png")
        cv2.imwrite(aug_image_path, aug_image)
        cv2.imwrite(aug_mask_path, aug_mask)

# Loop through dataset and augment
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
for image_file in tqdm(image_files, desc="Augmenting dataset"):
    image_path = os.path.join(IMAGE_DIR, image_file)
    mask_path = os.path.join(MASK_DIR, image_file)  # Assuming mask has the same filename
    augment_and_save(image_path, mask_path, OUTPUT_IMAGE_DIR, OUTPUT_MASK_DIR)
