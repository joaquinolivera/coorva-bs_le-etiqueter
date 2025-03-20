# coorva-bs_le-etiqueter
Python project for start a label studio project, upload images to it and add annotations bbox.

## Overview

This repository contains two Python scripts designed to integrate with Label Studio and YOLOv8 for the detection, annotation, and preparation of datasets involving object detection. The scripts are tailored to identify specific objects (e.g., posts and lights) within images, annotate these detections, and export datasets formatted specifically for YOLOv8 model training.

---

## Scripts Description

### 1. **`export_training_dataset.py`**

#### Purpose

Exports annotated image data from Label Studio into a YOLOv8-compatible dataset.

#### Functionality

- Connects to Label Studio using API credentials.
- Retrieves project information and annotations.
- Downloads images (including handling base64 encoded images).
- Converts Label Studio annotations into YOLOv8 format.
- Splits the dataset into training and validation sets.
- Generates `data.yaml` required by YOLOv8 for training.

#### Usage

```bash
python export_training_dataset.py
```

Ensure you have a valid `config.yaml` file specifying your Label Studio API credentials and project details.

### 2. **`lum_and_post_deteccion.py`**

#### Purpose

Automates detection and annotation of specific objects (such as posts and lights) within images, integrating directly with Label Studio.

#### Functionality

- Loads a custom-trained YOLO model (`best.pt`) for detecting objects.
- Processes images by resizing them to a standard dimension, maintaining aspect ratios.
- Runs detection on images, identifying specific object classes (`postacion`, `luminaria`).
- Creates bounding box annotations for detected objects.
- Automatically uploads images and annotations to Label Studio.
- Offers parallel processing capabilities for faster throughput.
- Saves progress periodically to facilitate resuming tasks.

#### Usage

Command-line execution with optional parameters:

```bash
python lum_and_post_deteccion.py --config config.yaml --visualize
```

Parameters include confidence thresholds, batch sizes, visualization options, and resume functionality.

---

## Dependencies

Make sure the following Python libraries are installed:

```bash
pip install requests pyyaml numpy opencv-python ultralytics torch
```

---

## Configuration

Both scripts require a YAML configuration file (`config.yaml`) containing:

```yaml
api_token: YOUR_LABEL_STUDIO_TOKEN
base_url: http://labelstudio.yourdomain.com
project_name: YOUR_PROJECT_NAME
model_path: path_to_your_yolo_model/best.pt
base_folder: path_to_images_folder
confidence_threshold: 0.3
```

Adjust these settings according to your environment.

---

## Workflow

1. **Setup Label Studio**: Prepare your labeling project.
2. **Run Detection Script**: Use `lum_and_post_deteccion.py` to process images and upload annotations.
3. **Export Dataset**: Execute `export_training_dataset.py` to download and format labeled data for YOLOv8.
4. **Train YOLOv8 Model**: Train your model using the prepared dataset:

```bash
yolo train model=yolov8n.pt data=path_to_dataset/data.yaml
```

---

## Troubleshooting

- Verify your Label Studio API token and URL.
- Confirm the existence and correctness of your YOLO model file.
- Check for correct Python package installations and versions.

---

## Security Notice

This repository may contain sensitive information (e.g., API tokens). Handle files with appropriate security measures and do not share publicly without removing sensitive credentials.

---

For any additional information or issues, consult the script's logs and console outputs.

