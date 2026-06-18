
# EuroSAT Image Classification and Visual Object Analysis

This projeinvestigates the aplicattion of deep learning techniques  to the EuroSAT(land use and land cover classification dataset) that is based on Sentinel-2 satelite imagery images.


The main task is **image classification** using three models:

1. A Custom convotutional neural network (CNN) trained from scratch.
2. ResNet18 with transfer learning.
3. EfficientNet-B0 with transfer learning.

Additional visual analysis extensions are also included:

- pseudo semantic segmentation
- pseudo object detection
- YOLOv8 pseudo object detection pipeline

## Important Note

EuroSAT is an image classification dataset. It does not provide official segmentation masks or bounding-box annotations. Therefore, the segmentation and object detection parts are implemented as **pseudo visual analysis extensions**, not as fully supervised segmentation/detection benchmarks.Since EuroSAT does not provide official segmentation masks or object detection annotations, pseudo labels were generated automatically to enable qualitative visual analysis.

## Dataset Description

The project uses the EuroSAT RGB dataset, which consists of Sentinel-2 satellite image patches categorized into ten land use and land cover classes.

### Dataset Characteristics

- Image size: 64 × 64 pixels
- Number of classes: 10
- Data source: Sentinel-2 satellite imagery
- Task: Image classification

### EuroSAT Classes

| Class ID | Class Name |
|-----------|-------------|
| 0 | AnnualCrop |
| 1 | Forest |
| 2 | HerbaceousVegetation |
| 3 | Highway |
| 4 | Industrial |
| 5 | Pasture |
| 6 | PermanentCrop |
| 7 | Residential |
| 8 | River |
| 9 | SeaLake |

Dataset source:

https://zenodo.org/records/7711810
---
##Implemented Models

## Custom CNN
A custom Convolutional Neural Network (CNN) was designed and trained from scratch.

Architecture components:

- Convolutional layers
- Batch normalization
- ReLU activation
- Max pooling
- Dropout
- Fully connected layers

This model serves as the baseline approach.

## ResNet18 Transfer Learning
ResNet was Intiliazed with pre-trained weigthts.
Transfer Learning strategy:
- Replace the final classification layer.
- Fine-tune the model on EuroSAT.

## EfficientNet-B0 Transfer Learning
EfficientNet-B0 was initialized with ImageNet pre-trained weights and fine-tuned on EuroSAT.

Advantages:

- Efficient architecture
- Strong feature extraction capability
- High classification performance

---

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
