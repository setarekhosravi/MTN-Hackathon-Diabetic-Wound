#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 15 for Hakathon
    @author: STRH
    preprocess the images
"""
import os
import cv2
from natsort import natsorted

# Paths to directories
image_path = "Hackathon Official Data/Cropped"
save_path = "Hackathon Official Data/Preprocessed"

# Ensure output directory exists
os.makedirs(save_path, exist_ok=True)

# Parameters for preprocessing
FILTER_KERNEL_SIZE = 5  # Kernel size for denoising filters

# Preprocessing function for images
def preprocess_image(image):
    # Apply Gaussian blur for denoising
    image = cv2.GaussianBlur(image, (FILTER_KERNEL_SIZE, FILTER_KERNEL_SIZE), 0)

    # Apply CLAHE for contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image

def apply_to_save():
    imgs_list = [f for f in os.listdir(image_path) if f.endswith('.png') or 
                        f.endswith('.PNG') or 
                        f.endswith('.webp') or 
                        f.endswith('.jpg') or 
                        f.endswith('jpeg')]
    imgs_list = natsorted(imgs_list)
    for path in imgs_list:
        img = cv2.imread(os.path.join(image_path, path))
        img = preprocess_image(img)
        cv2.imwrite(f"{save_path}/{path}", img)


if __name__=="__main__":
    apply_to_save()