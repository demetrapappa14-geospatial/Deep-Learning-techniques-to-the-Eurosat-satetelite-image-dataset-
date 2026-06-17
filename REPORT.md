# Report: EuroSAT Image Classification and Visual Detection Analysis

## 1. Problem Description

The purpose of this project is to classify satellite image patches from the EuroSAT dataset into 10 land-use and land-cover classes.

The selected primary task is **image classification**, which is appropriate for EuroSAT because the dataset provides one class label per image.

## 2. Dataset

EuroSAT contains RGB satellite image patches of size 64×64 pixels. The 10 classes are:

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

Some classes are visually similar, such as AnnualCrop, Pasture and HerbaceousVegetation, which makes the problem meaningful for deep learning evaluation.

## 3. Methodology

Three models are implemented:

### 3.1 Custom CNN from Scratch

A convolutional neural network was designed and trained from scratch. This model acts as a baseline.

### 3.2 ResNet18 Transfer Learning

A pretrained ResNet18 model was loaded with ImageNet weights and its final classification layer was replaced with a 10-class EuroSAT classifier.

### 3.3 EfficientNet-B0 Transfer Learning

A pretrained EfficientNet-B0 model was also fine-tuned for the same classification task. EfficientNet uses a more modern scaling strategy and is expected to achieve strong accuracy.

## 4. Evaluation Metrics

The project evaluates models using:

- test accuracy
- macro F1-score
- weighted F1-score
- classification report
- confusion matrix
- normalized confusion matrix
- loss and accuracy curves

## 5. Expected Comparison

The CNN baseline is expected to perform well but lower than pretrained architectures. ResNet18 and EfficientNet-B0 benefit from transfer learning and usually produce higher validation and test accuracy.

## 6. Pseudo Semantic Segmentation

EuroSAT does not provide pixel-level segmentation masks. Therefore, the semantic segmentation part is implemented as pseudo visualization using heuristic image-processing masks.

This is useful for visual analysis, but it is not a fully supervised semantic segmentation benchmark.

## 7. Pseudo Object Detection

EuroSAT does not provide bounding box annotations. A pseudo object detection script was implemented using:

- color thresholding
- binary masking
- region localization
- bounding box extraction

This produces visual bounding boxes around likely regions of interest, such as rivers, roads or vegetation.

## 8. YOLOv8 Pseudo Object Detection

A YOLOv8 pipeline is added by automatically converting heuristic masks into YOLO-format pseudo bounding boxes.

This allows training YOLOv8 on generated pseudo annotations. However, the resulting detector should be interpreted as a methodological extension and visualization tool, not as a true supervised object detection system.

## 9. Limitations

- EuroSAT images are low resolution.
- Some land-cover classes are visually similar.
- Object detection and segmentation labels are not official.
- YOLO annotations are generated automatically and may contain noise.

## 10. Conclusion

The project satisfies the deep learning image classification requirements using multiple models, quantitative evaluation, visual analysis, and model comparison. Transfer learning models are expected to outperform the custom CNN due to pretrained feature extraction.
