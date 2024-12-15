#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 15 for Hakathon
    @author: STRH
    apply kmeans
"""

import os
import numpy as np
import cv2
import tensorflow as tf
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from joblib import dump

# Paths to directories
IMAGE_DIR = "Hackathon Official Data/Preprocessed"  # Path to preprocessed images
OUTPUT_DIR = "Hackathon Official Data/Results"  # Path to save cluster assignments

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters for KMeans
N_CLUSTERS = 5  # Final clusters for KMeans
PCA_COMPONENTS = 50  # Number of PCA components for dimensionality reduction
FILTER_KERNEL_SIZE = 5  # Kernel size for denoising filters

# Load ResNet50 pre-trained model (without top layer)
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to extract deep features using ResNet50
def extract_deep_features(image_path):
    img = cv2.imread(image_path)  # Load the image
    img = cv2.resize(img, (224, 224))  # Resize to 224x224 for ResNet50
    img = img/255
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.astype('float32')
    img = preprocess_input(img)  # Preprocess the image (ResNet-specific)
    
    # Extract features using ResNet50
    features = base_model.predict(img)
    return features.flatten()  # Flatten to 1D array for clustering

# Step 1: Extract features from all images
print("Extracting features using DenseNet121...")
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
features = []

for image_file in image_files:
    image_path = os.path.join(IMAGE_DIR, image_file)
    feature = extract_deep_features(image_path)
    features.append(feature)

features = np.array(features)

# Step 2: Dimensionality reduction with PCA
print("Reducing dimensionality with PCA...")
pca = PCA(n_components=PCA_COMPONENTS)
reduced_features = pca.fit_transform(features)

# Save PCA model for future use
pca_model_path = os.path.join(OUTPUT_DIR, "pca_model.joblib")
dump(pca, pca_model_path)
print(f"PCA model saved at {pca_model_path}")

# Step 3: Apply K-Means clustering
print(f"Applying K-Means to form {N_CLUSTERS} clusters...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
kmeans_labels = kmeans.fit_predict(reduced_features)

# Step 4: Evaluate with Silhouette Score
sil_score = silhouette_score(reduced_features, kmeans_labels)
print(f"Silhouette Score for K-Means clustering: {sil_score}")

# Step 5: Save cluster assignments
final_cluster_results = {image_files[i]: int(kmeans_labels[i]) for i in range(len(image_files))}
cluster_results_path = os.path.join(OUTPUT_DIR, "final_cluster_assignments.txt")
with open(cluster_results_path, "w") as f:
    for image, cluster in final_cluster_results.items():
        f.write(f"{image}: Cluster {cluster}\n")
print(f"Cluster assignments saved at {cluster_results_path}")

# Step 6: Visualize clusters in 2D space
print("Visualizing clusters...")
plt.figure(figsize=(10, 8))
for cluster in range(N_CLUSTERS):
    cluster_points = reduced_features[kmeans_labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")
plt.title(f"K-Means Clustering - {N_CLUSTERS} Clusters (Deep Features)")
plt.legend(loc="upper right")
visualization_path = os.path.join(OUTPUT_DIR, "kmeans_clusters_visualization.png")
plt.savefig(visualization_path)
plt.show()
print(f"Cluster visualization saved at {visualization_path}")
