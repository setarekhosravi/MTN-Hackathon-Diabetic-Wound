#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 15 for Hakathon
    @author: STRH
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

from Unet.unet import UNet
from Unet.utils.data_loading import BasicDataset
from Unet.utils.utils import plot_img_and_mask
from sklearn.cluster import KMeans


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
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

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

def infer(args):
    # print(args.input)
    img = Image.open(args.input[0])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask_values, net = initial_unet(args)
    mask = predict_img(net=net,
                        full_img=img,
                        scale_factor=args.scale,
                        out_threshold=args.mask_threshold,
                        device=device)
    
    mask = mask_to_image(mask, mask_values)
    

    return img, mask

def extract_wound(img, mask):
    # Create a masked image showing only the wound area
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    wound_image = cv2.bitwise_and(img, img, mask=mask)

    return wound_image

def draw_countour(img, mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contour = img.copy()

    # Draw the contour on the image
    image_with_contour = cv2.drawContours(image_with_contour, contours, -1, (0, 255, 0), 2)
    return image_with_contour

def analyze_colors(image):
    # Convert BGR to HSV for better color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define HSV ranges for wound-specific colors
    color_ranges = {
        'red': [(0, 50, 50), (10, 255, 255)],  # Bright red
        'dark red': [(170, 50, 50), (180, 255, 255)],  # Dark red (close to purple)
        'yellow': [(20, 100, 100), (30, 255, 255)],  # Yellow (infection or slough)
        'black': [(0, 0, 0), (180, 255, 50)],  # Necrotic tissue
        'pink': [(160, 50, 50), (170, 255, 255)],  # Healthy granulation tissue
        'white': [(0, 0, 200), (180, 30, 255)],  # Exposed bone or tissue
        'brown': [(10, 50, 50), (20, 255, 200)],  # Eschar or dried blood
        'purple': [(130, 50, 50), (160, 255, 255)],  # Bruising or vascular discoloration
    }
    
    color_percentages = {}
    for color, (lower, upper) in color_ranges.items():
        # Convert bounds to numpy arrays for cv2
        lower_bound = np.array(lower, dtype="uint8")
        upper_bound = np.array(upper, dtype="uint8")
        
        # Create mask and calculate percentage
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        percentage = (cv2.countNonZero(mask) / mask.size) * 100
        color_percentages[color] = percentage
    
    return color_percentages

def plot_color_histogram(color_percentages):
    # Extract colors and their percentages
    colors = list(color_percentages.keys())
    percentages = list(color_percentages.values())
    bar_colors = ['red', 'darkred', 'yellow', 'black', 'pink', 'white', 'brown', 'purple']
    
    # Create a bar chart
    plt.figure(figsize=(10, 6), facecolor="lightgray")
    ax = plt.gca()  # Get current axes
    ax.set_facecolor("lightgray") 
    bars = plt.bar(colors, percentages, color=bar_colors, alpha=0.8)
    plt.title("Wound Color Percentage Histogram")
    plt.xlabel("Colors")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 100)
    
    # Add percentages on top of the bars
    for bar, percentage in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5, 
                 f'{percentage:.2f}%', ha='center', va='bottom', color='lightgray', fontsize=10)
    
    plt.show()

def segment_wound_parts(image):
    # Convert BGR to HSV for color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define HSV ranges for wound-specific colors
    color_ranges = {
        'red': [(0, 50, 50), (10, 255, 255)],  # Bright red
        'dark red': [(170, 50, 50), (180, 255, 255)],  # Dark red
        'yellow': [(20, 100, 100), (30, 255, 255)],  # Yellow (infection or slough)
        'black': [(0, 0, 0), (180, 255, 50)],  # Necrotic tissue
        'pink': [(160, 50, 50), (170, 255, 255)],  # Healthy granulation tissue
        'brown': [(10, 50, 50), (20, 255, 200)],  # Eschar or dried blood
        'purple': [(130, 50, 50), (160, 255, 255)],  # Bruising or vascular discoloration
        
        'yellow (light)': [(20, 100, 150), (40, 255, 255)],  # Light yellow
        'yellow (dark)': [(20, 80, 80), (40, 100, 150)],  # Dark yellow
        'white': [(0, 0, 200), (180, 30, 255)],  # White (exposed tissue)
        
        'brown (light)': [(10, 50, 100), (30, 255, 200)],  # Light brown
        'brown (dark)': [(10, 30, 50), (30, 50, 100)],  # Dark brown
        
        'pink (light)': [(160, 100, 150), (170, 255, 255)],  # Light pink
        'pink (dark)': [(160, 50, 100), (170, 100, 150)],  # Dark pink
        
        'purple (light)': [(130, 100, 150), (160, 255, 255)],  # Light purple
        'purple (dark)': [(130, 50, 100), (160, 100, 150)],  # Dark purple
    }
    
    # Create an empty dictionary to store segmented parts
    segmented_parts = {}
    
    for color_name, (lower, upper) in color_ranges.items():
        # Convert bounds to numpy arrays for cv2
        lower_bound = np.array(lower, dtype="uint8")
        upper_bound = np.array(upper, dtype="uint8")
        
        # Create a mask for the color
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Segment the part of the image corresponding to the color
        segmented_part = cv2.bitwise_and(image, image, mask=mask)
        
        # Save the segmented part in the dictionary
        segmented_parts[color_name] = segmented_part
        
    return segmented_parts

def display_segmented_parts(segmented_parts):
    # Plot each segmented part
    num_colors = len(segmented_parts)
    plt.figure(figsize=(15, num_colors))
    
    for i, (color_name, segmented_image) in enumerate(segmented_parts.items()):
        plt.subplot(1, num_colors, i + 1)
        plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
        plt.title(color_name)
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()


def check(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    # show segmentation mask
    img, mask = infer(args)
    # convert to numpy array
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    mask = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR)
    draw_wound = draw_countour(img, mask)
    wound = extract_wound(img, mask)
    # print(analyze_colors(img))
    plot_color_histogram(analyze_colors(img))
    display_segmented_parts(segment_wound_parts(img))
