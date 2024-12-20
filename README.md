# Computer Vision Hackathon


## Table of Contents
1. [Diabetic Wound Analysis Application](#diabetic-wound-analysis-application)
2. [Features](#features)
3. [Requirements](#requirements)
   - [Dependencies](#dependencies)
   - [Model Files](#model-files)
4. [Installation](#installation)
5. [Technical Details](#technical-details)
   - [Color Analysis](#color-analysis)
     - [Basic Colors](#basic-colors)
     - [Refined Color Ranges](#refined-color-ranges)
   - [Image Processing Pipeline](#image-processing-pipeline)
     - [Preprocessing](#preprocessing)
     - [Segmentation](#segmentation)
     - [Analysis](#analysis)
     - [Classification](#classification)
6. [Command Line Usage](#command-line-usage)
   - [Command Line Arguments](#command-line-arguments)
7. [File Structure](#file-structure)
8. [Output](#output)
9. [Notes](#notes)
10. [Contributing](#contributing)

## Diabetic Wound Analysis Application

A desktop application for analyzing diabetic wounds through image processing, segmentation, and machine learning techniques. The application provides automated wound assessment, color analysis, and generates detailed medical reports.

### Features

- Interactive GUI built with Tkinter
- Image preprocessing and wound area detection
- Wound segmentation using U-Net and DeepSkin models
- Multi-color analysis with refined color ranges
- Automated wound clustering using UMAP and ResNet50
- AI-powered wound assessment report generation
- Step-by-step analysis workflow

### Requirements

#### Dependencies
```
python >= 3.8
tkinter
PIL (Pillow)
opencv-python (cv2)
torch
transformers
numpy
scikit-learn
matplotlib
tensorflow
joblib
umap-learn
```

#### Model Files
The application requires pre-trained models in the following locations:
- `Hackathon Official Data/Results/kmeans_model.joblib`: Pre-trained K-means clustering model
- `Hackathon Official Data/Results/umap_model.joblib`: Pre-trained UMAP dimensionality reduction model
- U-Net model file (`.pth` format)

### Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install pillow opencv-python torch transformers scikit-learn matplotlib tensorflow joblib umap-learn
```
3. Ensure you have the required model files in the correct directory structure

### Technical Details

#### Color Analysis

The application analyzes wounds using the following color ranges in HSV format:

##### Basic Colors
- Red: `(0-10, 50-255, 50-255)`
- Dark Red: `(170-180, 50-255, 50-255)`
- Yellow: `(20-30, 100-255, 100-255)`
- Black: `(0-180, 0-255, 0-50)`
- Pink: `(160-170, 50-255, 50-255)`
- White: `(0-180, 0-30, 200-255)`
- Brown: `(10-20, 50-255, 50-200)`
- Purple: `(130-160, 50-255, 50-255)`

##### Refined Color Ranges
- Light/Dark Yellow
- Light/Dark Brown
- Light/Dark Pink
- Light/Dark Purple

#### Image Processing Pipeline

1. **Preprocessing**:
   - Contour detection and cropping
   - Image normalization
   - ResNet50 feature extraction

2. **Segmentation**:
   - U-Net or DeepSkin model segmentation (Download checkpoints from [here](https://drive.google.com/drive/folders/1IFMbiQjTgPVK8HGQ_rIaQXU_mcFwZ0eN?usp=sharing))
   - Mask generation and wound extraction
   - Contour drawing and dilation

3. **Analysis**:
   - Color percentage calculation
   - Multi-part segmentation
   - Histogram visualization

4. **Classification**:
   - Feature extraction using ResNet50
   - UMAP dimensionality reduction
   - K-means clustering prediction

### Command Line Usage

The application can also be run from the command line:

```bash
python test.py --model MODEL.pth --input IMAGE_PATH [OPTIONS]
```

#### Command Line Arguments
- `--model, -m`: Model file path or "deepskin"
- `--input, -i`: Input image path(s)
- `--mask-threshold, -t`: Mask probability threshold (default: 0.5)
- `--scale, -s`: Input image scale factor (default: 0.5)
- `--bilinear`: Use bilinear upsampling
- `--classes, -c`: Number of classes (default: 2)

### File Structure

- `wound_analysis_app.py`: Main GUI application
- `test.py`: Core analysis functions
- `crop_images.py`: Image preprocessing utilities
- `preprocess.py`: Image preprocessing functions
- `Unet/`: U-Net model implementation
- `deepskin.py`: DeepSkin model implementation

### Output

The application generates:
1. Segmented wound images
2. Color distribution histograms
3. Multi-part segmentation visualization
4. Cluster prediction
5. Detailed wound assessment report

### Notes

- The application uses ResNet50 for feature extraction
- UMAP is used for dimensionality reduction before clustering
- Color analysis uses HSV color space for better accuracy
- Both U-Net and DeepSkin models are supported for segmentation

### Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
