#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 15 for Hakathon
    @author: STRH
    apply kmeans
"""

import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from joblib import dump, load
import matplotlib.pyplot as plt
from tqdm import tqdm

# Paths to directories
IMAGE_DIR = "Hackathon Official Data/Preprocessed"  # Path to preprocessed images
OUTPUT_DIR = "Hackathon Official Data/Results"  # Path to save cluster assignments

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
N_CLUSTERS = 5  # Number of clusters
COLOR_SPACE = "HSV"  # Choose between 'RGB' or 'HSV'
HIST_BINS = 32  # Number of bins for histogram
PCA_COMPONENTS = 50  # Number of PCA components

# Function to extract color histogram as features
def extract_color_histogram(image_path, color_space="RGB", bins=32):
    image = cv2.imread(image_path)
    if color_space == "HSV":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Compute histogram and normalize
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Step 1: Extract features
print("Extracting features from images...")
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
features = []
for image_file in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(IMAGE_DIR, image_file)
    hist = extract_color_histogram(image_path, color_space=COLOR_SPACE, bins=HIST_BINS)
    features.append(hist)

features = np.array(features)

# Step 2: Dimensionality reduction with PCA
print("Reducing dimensionality with PCA...")
pca = PCA(n_components=PCA_COMPONENTS)
reduced_features = pca.fit_transform(features)

# Save the PCA model for future use
pca_model_path = os.path.join(OUTPUT_DIR, "pca_model.joblib")
dump(pca, pca_model_path)
print(f"PCA model saved at {pca_model_path}")

# Step 3: Apply K-Means clustering
print(f"Clustering into {N_CLUSTERS} clusters using K-Means...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
cluster_labels = kmeans.fit_predict(reduced_features)

# Save the K-Means model for future use
kmeans_model_path = os.path.join(OUTPUT_DIR, "kmeans_model.joblib")
dump(kmeans, kmeans_model_path)
print(f"K-Means model saved at {kmeans_model_path}")

# Step 4: Evaluate clustering with Silhouette Score
sil_score = silhouette_score(reduced_features, cluster_labels)
print(f"Silhouette Score for {N_CLUSTERS} clusters: {sil_score}")

# Step 5: Save cluster assignments
cluster_results = {image_files[i]: int(cluster_labels[i]) for i in range(len(image_files))}
cluster_results_path = os.path.join(OUTPUT_DIR, "cluster_assignments.txt")
with open(cluster_results_path, "w") as f:
    for image, cluster in cluster_results.items():
        f.write(f"{image}: Cluster {cluster}\n")
print(f"Cluster assignments saved at {cluster_results_path}")

# Step 6: Visualize clusters in 2D space
print("Visualizing clusters...")
plt.figure(figsize=(10, 8))
for cluster in range(N_CLUSTERS):
    cluster_points = reduced_features[cluster_labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")
plt.title("Cluster Visualization (PCA Reduced)")
plt.legend()
visualization_path = os.path.join(OUTPUT_DIR, "clusters_visualization.png")
plt.savefig(visualization_path)
plt.show()
print(f"Cluster visualization saved at {visualization_path}")
