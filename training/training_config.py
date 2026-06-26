"""
Configuration for YOLO segmentation training.

This module is intentionally separate from the prediction / reporting pipeline.
Edit the values here or override them from the command line in train_yolo.py.
"""

from pathlib import Path

# Path to your YOLO segmentation data.yaml
DATA_YAML = Path(r"data.yaml")

# Folder where YOLO training runs will be stored
TRAINING_OUTPUT_ROOT = Path(r"out")

# Device:
#   "auto" = cuda if available, otherwise cpu
#   "cpu"  = CPU only
#   "0"    = first CUDA GPU
DEVICE = "auto"

# Grid search models, add more if necessary
MODELS = [
    "yolov8n-seg.pt",
    #"yolov8s-seg.pt",
   # "yolo11m-seg.pt",
    #"yolo11s-seg.pt",
   # "yolo26m-seg.pt",
   # "yolo26s-seg.pt",
]

IMG_SIZES = [1024]
EPOCHS_LIST = [150]
OPTIMIZERS = ["AdamW"]
LR0_VALUES = [0.001]
WEIGHT_DECAY_VALUES = [0.0005]

# YOLO training settings
BATCH = -1
WORKERS = 0
PATIENCE = 80

# Segmentation/root-friendly augmentation settings.
# These are conservative because the root images have a fixed geometry.
AUGMENTATION = {
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "close_mosaic": 0,
}
