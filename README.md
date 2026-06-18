
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

## Additional Visual Analysis

### Pseudo Semantic Segmentation

EuroSAT does not provide official segmentation masks.

Pseudo segmentation is implemented using:

- Color thresholding
- Binary mask generation
- Overlay visualization

### Pseudo Object Detection

EuroSAT does not provide official bounding-box annotations.

Pseudo object detection is implemented by:

- Generating masks
- Identifying regions of interest
- Creating pseudo bounding boxes

### YOLOv8 Pseudo Object Detection

Pseudo annotations are converted into YOLO format to train a YOLOv8 model.

The pipeline includes:

1. Pseudo label generation
2. YOLO dataset creation
3. YOLOv8 training
4. YOLOv8 prediction

---

## Project Structure

```text
Deep-Learning-techniques-to-the-Eurosat-satetelite-image-dataset-/

├── data/
├── models/
├── notebooks/
├── results/
├── src/
├── yolo_pseudo_dataset/

├── README.md
├── REPORT.pdf
├── requirements.txt
├── RUN_ORDER_WINDOWS.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/demetrapappa14-geospatial/Deep-Learning-techniques-to-the-Eurosat-satetelite-image-dataset-.git

cd Deep-Learning-techniques-to-the-Eurosat-satetelite-image-dataset-
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
## Evaluation Metrics 
Classsification models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Normalized Confusion Matrix

YOLOv8 pseudo object detection is evaluated using:

- Precision
- Recall
- mAP@50
- mAP@50-95

Pseudo semantic segmentation is evaluated qualitatively through visual inspection.

---

## Expected Results
| Model | Training Strategy | Test Accuracy |
|-------|-------------------|---------------:|
| Custom CNN | Training from scratch | 93.58% |
| ResNet18 | Transfer learning | 94.74% |
| EfficientNet-B0 | Transfer learning | 97.70% |

### YOLOv8 Pseudo Detection Results

| Metric | Value |
|---------|------:|
| Precision | 0.94 |
| Recall | 0.91 |
| mAP@50 | 0.937 |
| mAP@50-95 | 0.900 |


## Limitations 

-Eurosat Dataset is desdigned  fro image classification only 
-No official segmetation masks are available
