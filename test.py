#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 15 for Hakathon
    @author: STRH, Ebadzadeh
    final test
"""

import cv2
import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from joblib import load
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from Unet.unet import UNet
from Unet.utils.data_loading import BasicDataset
from Unet.utils.utils import plot_img_and_mask
from deepskin import wound_segmentation
import umap.umap_ as umap
from sklearn.cluster import KMeans
from crop_images import find_contour, crop
from preprocess import preprocess_image

###############################################
# GLOBALS & CONFIGURATIONS
###############################################
COLOR_RANGES = {
    'red': [(0, 50, 50), (10, 255, 255)],
    'dark red': [(170, 50, 50), (180, 255, 255)],
    'yellow': [(20, 100, 100), (30, 255, 255)],
    'black': [(0, 0, 0), (180, 255, 50)],
    'pink': [(160, 50, 50), (170, 255, 255)],
    'white': [(0, 0, 200), (180, 30, 255)],
    'brown': [(10, 50, 50), (20, 255, 200)],
    'purple': [(130, 50, 50), (160, 255, 255)]
}

REFINED_COLOR_RANGES = {
    'yellow (light)': [(20, 100, 150), (40, 255, 255)],
    'yellow (dark)': [(20, 80, 80), (40, 100, 150)],
    'brown (light)': [(10, 50, 100), (30, 255, 200)],
    'brown (dark)': [(10, 30, 50), (30, 50, 100)],
    'pink (light)': [(160, 100, 150), (170, 255, 255)],
    'pink (dark)': [(160, 50, 100), (170, 100, 150)],
    'purple (light)': [(130, 100, 150), (160, 255, 255)],
    'purple (dark)': [(130, 50, 100), (160, 100, 150)]
}

ALL_COLOR_RANGES = {**COLOR_RANGES, **REFINED_COLOR_RANGES}

###############################################
# U-NET/DeepSkin PREDICTION
###############################################
def initial_unet(args):
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    return mask_values, net

def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')

        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = (torch.sigmoid(output) > out_threshold)

    return mask[0].long().squeeze().numpy()

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

def infer(args, img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask_values, net = initial_unet(args)
    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=args.scale,
                       out_threshold=args.mask_threshold,
                       device=device)
    mask = mask_to_image(mask, mask_values)
    return img, mask

def extract_wound(args, img, mask):
    # Create a masked image showing only the wound area
    if args.model.lower()!="deepskin":
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    wound_image = cv2.bitwise_and(img, img, mask=mask)

    return wound_image

def draw_countour(args, img, mask):
    if args.model.lower() != "deepskin":
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contour = img.copy()
    image_with_contour = cv2.drawContours(image_with_contour, contours, -1, (0, 255, 0), 2)
    return image_with_contour

def dilate(mask, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8) 
    dilated_mask = cv2.dilate(mask, kernel, iterations=1) 

    return dilated_mask

def skin_extraxtor(args, img, mask):
    dilated = dilate(mask, 25)
    skin_mask = dilated - mask
    around_skin = extract_wound(args, img, skin_mask)

    return  around_skin

def show_wound_parts(args, img):
    wound_list = ["Original Image", "Wound Mask", "Wound Part", "Wound", "Outer Layer"]
    if args.model.lower()=="deepskin":
        mask, wound = deepskin(img)

    else:
        img, mask = infer(args, img)
        # convert to numpy array
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        mask = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR)
        wound = extract_wound(args, img, mask)
    
    draw_wound = draw_countour(args, img, mask)
    around_skin = skin_extraxtor(args, img, mask)

    image_list = [img, mask,
                  draw_wound, wound, around_skin]
    
    plt.figure(figsize=(15, len(image_list)))
    for i in range(len(image_list)):
        plt.subplot(1, len(image_list), i + 1)
        plt.imshow(cv2.cvtColor(image_list[i], cv2.COLOR_BGR2RGB))
        plt.title(wound_list[i])
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def deepskin(img):
    segmentation = wound_segmentation(
        img=img[..., ::-1],
        tol=0.95,
        verbose=True,
    )
    wound_mask, body_mask, bg_mask = cv2.split(segmentation)
    wound = cv2.bitwise_and(img, img, mask=wound_mask)
    return wound_mask, wound

###############################################
# COLOR ANALYSIS & SEGMENTATION
###############################################
def analyze_colors(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_percentages = {}
    for color, (lower, upper) in COLOR_RANGES.items():
        lower_bound = np.array(lower, dtype="uint8")
        upper_bound = np.array(upper, dtype="uint8")
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        percentage = (cv2.countNonZero(mask) / mask.size) * 100
        color_percentages[color] = percentage
    return color_percentages

def plot_color_histogram(color_percentages):
    colors = list(color_percentages.keys())
    percentages = list(color_percentages.values())
    bar_colors = ['red', 'darkred', 'yellow', 'black', 'pink', 'white', 'brown', 'purple']

    plt.figure(figsize=(10, 6), facecolor="lightgray")
    ax = plt.gca()
    ax.set_facecolor("lightgray")
    bars = plt.bar(colors, percentages, color=bar_colors[:len(colors)], alpha=0.8)
    plt.title("Wound Color Percentage Histogram")
    plt.xlabel("Colors")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 100)

    for bar, percentage in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5,
                 f'{percentage:.2f}%', ha='center', va='bottom', color='lightgray', fontsize=10)

    plt.show()

def segment_wound_parts(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    segmented_parts = {}
    for color_name, (lower, upper) in ALL_COLOR_RANGES.items():
        lower_bound = np.array(lower, dtype="uint8")
        upper_bound = np.array(upper, dtype="uint8")
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        segmented_part = cv2.bitwise_and(image, image, mask=mask)
        segmented_parts[color_name] = segmented_part
    return segmented_parts

def display_segmented_parts(segmented_parts):
    num_colors = len(segmented_parts)
    plt.figure(figsize=(15, num_colors))
    for i, (color_name, segmented_image) in enumerate(segmented_parts.items()):
        plt.subplot(1, num_colors, i + 1)
        plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
        plt.title(color_name)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

###############################################
# K-MEANS CLUSTER PREDICTION FUNCTION
###############################################
def extract_deep_features_for_kmeans(image_bgr):
    # Match EXACT preprocessing as in K-Means training code
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    img = cv2.resize(image_bgr, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype('float32')
    img = preprocess_input(img)
    features = base_model.predict(img, verbose=0)
    return features.flatten()

def predict_cluster_for_wound(image_bgr, kmeans_model_path, umap_model_path):
    """
    Predict which cluster the wound image belongs to using the trained K-Means model and UMAP.
    
    Steps:
    1. Extract deep features from the image (same as training).
    2. Load UMAP model and transform features.
    3. Load the trained K-Means model and predict the cluster.
    """
    features = extract_deep_features_for_kmeans(image_bgr).reshape(1, -1)

    # Load UMAP and transform (just like at training)
    umap_reducer = load(umap_model_path)
    umap_embedding = umap_reducer.transform(features)

    # Load K-Means and predict cluster
    kmeans = load(kmeans_model_path)
    cluster_label = kmeans.predict(umap_embedding)
    return cluster_label[0]

###############################################
# MAIN & ARGUMENTS
###############################################
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify model file or "deepskin" for that model.')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', required=True,
                        help='Filenames of input images')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for input images')
    parser.add_argument('--bilinear', action='store_true', default=False,
                        help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--kmeans-model', default='Hackathon Official Data/Results/kmeans_model.joblib',
                        help='Path to the trained KMeans model')
    parser.add_argument('--umap-model', default='Hackathon Official Data/Results/umap_model.joblib',
                        help='Path to the UMAP model')
    return parser.parse_args()

def main():
    args = get_args()

    for image_path in args.input:
        print(f"\nPre-Processing image...")
        original_img = cv2.imread(image_path)
        bounding_box = find_contour(original_img)
        img = crop(image=original_img, bbox=bounding_box)
        img_for_clustering = preprocess_image(img)
        print(f"Pre-Processing Done!")

        print(f"\nProcessing image...")

        # Show wound parts
        show_wound_parts(args, img)

        # Analyze colors and plot histogram
        color_percentages = analyze_colors(img)
        plot_color_histogram(color_percentages)

        # Segment and display parts
        parts = segment_wound_parts(img)
        display_segmented_parts(parts)

        # Predict cluster using K-Means (with UMAP)
        cluster_id = predict_cluster_for_wound(img_for_clustering, args.kmeans_model, args.umap_model)
        print(f"Cluster prediction for {image_path}: {cluster_id}")

if __name__ == '__main__':
    main()