# YOLOv8 Pseudo Object Detection Guide

EuroSAT does not include official bounding boxes. To include a YOLO pipeline, this project creates pseudo bounding boxes automatically using image processing.

## Steps

```bash
cd src
python yolo_create_pseudo_dataset.py
python train_yolo_pseudo_detection.py
python yolo_predict_pseudo_detection.py --class_name Highway
```

## Academic explanation

This is pseudo object detection. The boxes are generated with heuristic masks and are not human-annotated ground truth.
