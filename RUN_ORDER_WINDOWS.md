# Run Order on Windows / VS Code

Open PowerShell in the project root:

```text
eurosat_optimized_grade10_project
```

## 1. Create and activate venv

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

## 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## 3. Go to source folder

```powershell
cd .\src\
```

## 4. Download dataset

```powershell
python download_eurosat_dataset.py
```

## 5. Train classification models

```powershell
python train_cnn.py
python train_resnet18.py
python train_efficientnet_b0.py
```

## 6. Compare results

```powershell
python compare_results.py
```

## 7. Predict one image

```powershell
python predict_single_image.py --model resnet18_transfer --class_name Forest
```

## 8. Extra pseudo visualizations

```powershell
python pseudo_semantic_segmentation.py
python pseudo_object_detection.py
```

## 9. YOLO pseudo object detection

```powershell
python yolo_create_pseudo_dataset.py
python train_yolo_pseudo_detection.py
python yolo_predict_pseudo_detection.py --class_name Highway
```
