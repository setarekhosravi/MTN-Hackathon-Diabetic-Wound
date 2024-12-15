#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 15 for Hakathon
    @author: STRH
    Dataset analysis
"""

import os
from PIL import Image
import matplotlib.pyplot as plt

# Set the directory containing the images
image_dir = 'Final Data'

# Create lists to store the image widths and heights
widths = []
heights = []

# Loop through the images and extract the dimensions
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)
        width, height = image.size
        widths.append(width)
        heights.append(height)

# Create the figure and histogram plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the histogram for widths
ax1.hist(widths, bins=30)
ax1.set_title('Image Width Distribution')
ax1.set_xlabel('Width (pixels)')
ax1.set_ylabel('Frequency')

# Plot the histogram for heights
ax2.hist(heights, bins=30)
ax2.set_title('Image Height Distribution')
ax2.set_xlabel('Height (pixels)')
ax2.set_ylabel('Frequency')

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.4)

# Save the plot to a file
plt.savefig('image_size_histogram.png', dpi=300)

print('Plot saved as image_size_histogram.png')