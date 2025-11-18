# Object Detection CV Project

This project implements object detection using YOLOv8 on the PASCAL VOC dataset. It includes data preparation, model training, improvement (with augmentation and larger model), evaluation for overfitting, and testing on new images. The project was developed locally on PC with RTX 3060, temporarily switched to Google Colab due to GPU issues, and resumed locally after fixing CUDA.

## Project Overview
- **Goal**: Train a YOLO model for object detection on 20 classes from VOC dataset (aeroplane, bicycle, etc.).
- **Dataset**: PASCAL VOC0712 from Kaggle (~16k images, converted XML annotations to TXT, custom splits: train/val/test).
- **Model**: Started with YOLOv8m, improved to YOLOv8l with augmentation (mosaic=1.0, mixup=0.2, copy_paste=0.1), 100 epochs, batch=8, imgsz=416.
- **Key Challenges**: Slow local training due to CUDA/GPU issues (fixed with torch reinstall and NVIDIA settings); OOM errors (fixed with PYTORCH_CUDA_ALLOC_CONF).
- **Results**: Best mAP50=0.966, mAP50-95=0.848 (no overfitting, tested on 10 new images with accurate detections and minor errors).

## Steps and Development Process
### 1. Initial Local Setup on PC
- Downloaded VOC dataset from Kaggle.
- Converted XML annotations to YOLO TXT format using custom script.
- Created custom splits (train/val/test) and dataset.yaml with paths.
- Set up venv and installed dependencies (ultralytics, torch with CUDA 13.0, etc.).
- Attempted training on RTX 3060, but faced slow performance (low GPU utilization, CPU fallback).

### 2. Switch to Google Colab
- Due to local GPU issues, moved to Colab for faster training (free T4 GPU).
- Uploaded data via Kaggle API, adapted paths to /content/.
- Fixed errors: Updated Ultralytics settings (datasets_dir=''), absolute paths in yaml, YAML formatting.
- Trained initial model (YOLOv8m, 50 epochs) with mAP50-95=0.65.
- Downloaded results (runs, best.pt, voc_yolo_model.pt) to PC.

### 3. Return to Local PC and Fix CUDA/GPU Issues
- Reinstalled torch with CUDA support: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130.
- Fixed GPU utilization: Added python.exe to NVIDIA Control Panel (High-performance), ensured CUDA toolkit installed.
- Adapted code/paths for local (S:\... paths in yaml).
- Handled OOM: Added os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.5' and torch.cuda.empty_cache().
- Resumed training with resume=True from last.pt.

### 4. Model Improvement
- Switched to larger model (yolov8l.pt) for better accuracy.
- Added augmentation: mosaic=1.0, mixup=0.2, copy_paste=0.1.
- Trained for 100 epochs (with patience=20, but continued to 100 as mAP improved).
- Reduced params for 6GB VRAM: batch=8, imgsz=416.
- Final training: Resumed from checkpoints, total ~10-12 hours on RTX 3060.

### 5. Overfitting Check
- Loaded best.pt and ran model.val() on val set: Val mAP50-95=0.848, loss stable and close to train (plotted from results.csv using pandas/matplotlib) — no overfitting (val metrics high and consistent, no drop compared to train).
- Per class: Balanced performance (e.g., cat=0.962, person=0.811), confirming good generalization.

### 6. Testing on New Data
- Downloaded 10 new unseen images (with VOC classes like person, car, dog) from Google to img_for_test/ folder.
- Predicted: model.predict('img_for_test/', conf=0.5, save=True) — accurate detections (e.g., person with conf=0.9, car=0.85; minor errors on background/similar objects, but overall correct classes and boxes).
- Results saved in runs/predict/ — model generalizes well, no overfitting signs.

## Installation
1. Clone the repo: `git clone https://github.com/ErebusAbyss/object-detection-cv.git`
2. Navigate to the project folder: `cd object-detection-cv/content` (note: content is a subfolder for path compatibility).
3. Create and activate venv: `python -m venv od_venv; od_venv\Scripts\activate` (Windows).
4. Install dependencies: `pip install -r requirements.txt`

## Usage
- **Train/Resume**: Run cells in object_detection.ipynb (includes resume=True for continuing from last.pt).
- **Evaluate**: `model = YOLO('runs/detect/voc_improved_1/weights/best.pt'); metrics = model.val()`
- **Predict on new images**: `results = model.predict('img_for_test/', save=True)`
- **Full code**: See object_detection.ipynb for all steps.

## Results
- Best model: runs/detect/voc_improved_1/weights/best.pt (mAP50=0.966, mAP50-95=0.848).
- Per class: High accuracy (e.g., cat=0.962, train=0.924).
- No overfitting; tested on unseen data with successful detections (minor background errors).

## License
MIT