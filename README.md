# YOLOv12 Setup and Training

This repository provides a setup guide and training instructions for using YOLOv12 for object detection tasks, specifically applied to brain tumor detection.

## Prerequisites
Ensure you have Google Colab with access to Google Drive.

## Installation

Follow these steps to set up YOLOv12:

### 1. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Navigate to Working Directory
```sh
%cd /content/drive/MyDrive/AI/YOLO
```

### 3. Clone YOLOv12 Repository
```sh
!git clone https://github.com/sunsmarterjie/yolov12.git
%cd yolov12
```

### 4. Install Dependencies
```sh
!wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
!pip install -r requirements.txt
!pip install -e .
```

## Running YOLOv12

### 1. Load a Pre-trained Model
```python
from ultralytics import YOLO
model = YOLO('yolov12n.pt')
results = model('/content/drive/MyDrive/AI/YOLO/image1.jpg')
results[0].show()
```

### 2. Train the Model
Modify `data.yaml` to specify the dataset location before running training.
```python
model = YOLO('yolov12n.yaml')
results = model.train(
  data='../Brain tumor/data.yaml',
  epochs=20,
  batch=64,
  imgsz=640,
  scale=0.5,
  mosaic=1.0,
  mixup=0.0,
  copy_paste=0.1,
  device="0")
```

### 3. Perform Inference
```python
model = YOLO('/content/drive/MyDrive/AI/YOLO/yolov12/runs/detect/train/weights/best.pt')
results1 = model('/content/drive/MyDrive/AI/YOLO/b.jpg')
results1[0].show()

results2 = model('/content/drive/MyDrive/AI/YOLO/bt.jpeg')
results2[0].show()
```

## Notes
- Ensure that `data.yaml` is correctly configured with the dataset path.
- Adjust hyperparameters as needed to optimize model performance.

## License
This project is released under an open-source license. Please check the original YOLOv12 repository for details.
