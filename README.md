
# EuroSAT Image Classification and Visual Object Analysis

This project applies deep learning techniques to the EuroSAT(land use and land cover classification dataset) that is based on Sentinel-2 satelite imagery images.


The main task is **image classification** using three models:

1. Custom CNN trained from scratch
2. ResNet18 with transfer learning
3. EfficientNet-B0 with transfer learning

Additional visual analysis extensions are also included:

- pseudo semantic segmentation
- pseudo object detection
- YOLOv8 pseudo object detection pipeline

## Important Note

EuroSAT is an image classification dataset. It does not provide official segmentation masks or bounding-box annotations. Therefore, the segmentation and object detection parts are implemented as **pseudo visual analysis extensions**, not as fully supervised segmentation/detection benchmarks.

## Project Structure

```text
data/                       EuroSAT images after download
models/                     saved trained model weights
results/                    curves, metrics, confusion matrices
src/                        source code
yolo_pseudo_dataset/         generated YOLO pseudo dataset
README.md
REPORT.md
requirements.txt
RUN_ORDER_WINDOWS.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Download Dataset

```bash
cd src
python download_eurosat_dataset.py
```

## Train Models

```bash
python train_cnn.py
python train_resnet18.py
python train_efficientnet_b0.py
```

## Compare Models

```bash
python compare_results.py
```

Outputs:

```text
results/model_comparison.csv
results/model_comparison_accuracy.png
```

## Prediction

```bash
python predict_single_image.py --model resnet18_transfer --class_name Forest
```

## Pseudo Semantic Segmentation

```bash
python pseudo_semantic_segmentation.py
```

## Pseudo Object Detection

```bash
python pseudo_object_detection.py
```

## YOLOv8 Pseudo Object Detection

Create pseudo YOLO labels:

```bash
python yolo_create_pseudo_dataset.py
```

Train YOLO:

```bash
python train_yolo_pseudo_detection.py
```

Run YOLO prediction:

```bash
python yolo_predict_pseudo_detection.py --class_name Highway
```

## Expected Results

Typical results on EuroSAT:

| Model | Expected Accuracy |
|---|---:|
| Custom CNN | 88–91% |
| ResNet18 Transfer Learning | 94–96% |
| EfficientNet-B0 Transfer Learning | 95–98% |

## Outputs for Report

The code automatically saves:

- loss curves
- accuracy curves
- confusion matrices
- normalized confusion matrices
- classification reports
- test metrics
- comparison table
- pseudo object detection images
- pseudo segmentation images
- YOLO runs/predictions
=======
# Deep-Learning-techniques-to-the-Eurosat-satetelite-image-dataset-
Remote sensing image analysis on EuroSAT using CNNs, ResNet18, EfficientNet-B0, YOLOv8 pseudo object detection, and pseudo semantic segmentation with comprehensive performance evaluation.
>>>>>>> dabe258756d527f94d8d042a7ec978e3e96bdb94
