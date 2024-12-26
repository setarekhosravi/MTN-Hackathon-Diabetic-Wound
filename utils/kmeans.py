#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 15 for Hakathon
    @author: Ebadzadeh, STRH
    kmeans code
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from joblib import dump
import matplotlib.pyplot as plt
import umap.umap_ as umap

# Paths
IMAGE_DIR = "Hackathon Official Data/Preprocessed"
OUTPUT_DIR = "Hackathon Official Data/Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CLUSTERS = 5

# Load ResNet50 base model
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_deep_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = base_model.predict(img, verbose=0)
    return features.flatten()

# Feature extraction
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
features = []
for image_file in image_files:
    fpath = os.path.join(IMAGE_DIR, image_file)
    feat = extract_deep_features(fpath)
    features.append(feat)
features = np.array(features)

# UMAP dimensionality reduction
print("Reducing dimensions with UMAP...")
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.0001, n_components=2, random_state=42)
umap_embedding = umap_reducer.fit_transform(features)

umap_model_path = os.path.join(OUTPUT_DIR, "umap_model.joblib")
dump(umap_reducer, umap_model_path)
print(f"UMAP model saved at {umap_model_path}")

# K-Means clustering
print(f"Applying K-Means with {N_CLUSTERS} clusters on UMAP embedding...")
kmeans = KMeans(n_clusters=N_CLUSTERS, init='k-means++', n_init=50, max_iter=1000, random_state=42)
labels = kmeans.fit_predict(umap_embedding)

# Save KMeans model
kmeans_model_path = os.path.join(OUTPUT_DIR, "kmeans_model.joblib")
dump(kmeans, kmeans_model_path)
print(f"KMeans model saved at {kmeans_model_path}")

# Evaluate
sil_score = silhouette_score(umap_embedding, labels)
ch_score = calinski_harabasz_score(umap_embedding, labels)
db_score = davies_bouldin_score(umap_embedding, labels)
inertia = kmeans.inertia_

print(f"Silhouette Score: {sil_score:.4f}")
print(f"Inertia: {inertia:.4f}")
print(f"Calinski-Harabasz Score: {ch_score:.4f}")
print(f"Davies-Bouldin Score: {db_score:.4f}")

# Save assignments
final_cluster_results = {image_files[i]: int(labels[i]) for i in range(len(image_files))}
cluster_results_path = os.path.join(OUTPUT_DIR, "final_cluster_assignments.txt")
with open(cluster_results_path, "w") as f:
    f.write("Image File\tCluster\n")
    for img, cluster_id in final_cluster_results.items():
        f.write(f"{img}\t{cluster_id}\n")
print(f"Cluster assignments saved at {cluster_results_path}")

metrics_path = os.path.join(OUTPUT_DIR, "cluster_metrics.txt")
with open(metrics_path, "w") as f:
    f.write("Clustering Performance Metrics (UMAP+KMeans):\n")
    f.write(f"Number of Clusters: {N_CLUSTERS}\n")
    f.write(f"Silhouette Score: {sil_score:.4f}\n")
    f.write(f"Inertia: {inertia:.4f}\n")
    f.write(f"Calinski-Harabasz Score: {ch_score:.4f}\n")
    f.write(f"Davies-Bouldin Score: {db_score:.4f}\n")
print(f"Metrics saved at {metrics_path}")

# Visualization
print("Visualizing UMAP clusters...")
plt.figure(figsize=(10, 8))
for c in range(N_CLUSTERS):
    pts = umap_embedding[labels == c]
    plt.scatter(pts[:,0], pts[:,1], label=f"Cluster {c}", alpha=0.7, s=30)
plt.title(f"K-Means Clustering - {N_CLUSTERS} Clusters (UMAP Embedding)")
plt.legend(loc="upper right")
visualization_path = os.path.join(OUTPUT_DIR, "kmeans_clusters_umap.png")
plt.savefig(visualization_path, dpi=300)
plt.close()  # Change plt.show() to plt.close() to prevent display in non-interactive environments
print(f"Cluster visualization saved at {visualization_path}")