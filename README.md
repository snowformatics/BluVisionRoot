# 🌾 BluVision Root — Fusarium Root Severity Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLO](https://img.shields.io/badge/YOLO-segmentation-brightgreen)
![OpenCV](https://img.shields.io/badge/OpenCV-image%20analysis-orange)
![Status](https://img.shields.io/badge/status-research%20tool-purple)

BluVision Root is a Python pipeline for automated **Fusarium root infection analysis** from macrobot images. It preprocesses images, runs YOLO segmentation, calculates object- and lane-level severity metrics, and generates CSV and HTML reports.

<img src="BluVisionRoot.png" width="1000">

## ✨ Features

- Image preprocessing and percentile white balance
- YOLO segmentation
- MaxRGB-based severity measurements
- Object-level metrics for Fusarium infection and root tissue
- Automatic assignment to 4 horizontal lanes
- CSV summaries and visual HTML reports
- Batch processing of complete experiments

## 📦 Installation

```bash
git clone https://github.com/snowformatics/BluVisionRoot.git
cd BluVisionRoot

conda create -n fusvision python=3.10 -y
conda activate fusvision

pip install ultralytics opencv-python numpy torch torchvision
```

### Optional GPU support

For NVIDIA GPUs, install the PyTorch build matching your CUDA setup from the official PyTorch installation guide:

https://pytorch.org/get-started/locally/

Check GPU availability:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 🧪 Test the pipeline

A ready-to-use test dataset is included in:

```text
test_images/
```

`config.py` is already configured for this test. After installation, simply run:

```bash
python main.py
```

## ⚙️ Configuration

Edit `config.py` for your own dataset:

```python
MODEL_PATH = r"yolov8s-seg_img1024_ep150_AdamW_lr0.0005_best.pt"
ROOT_FOLDER = r"MB0504"
OUTPUT_ROOT = r"MB0504_yolov8s"

EXPERIMENT_NAME = "MB0504"
INPUT_HAS_EXPERIMENT_LEVEL = False

IMG_SIZE = 1024
DEVICE = "cpu"          # "0" for GPU
FILE_SUFFIX = "preview.tif"
N_LANES = 4
```

For a root folder containing multiple experiments:

```python
INPUT_HAS_EXPERIMENT_LEVEL = True
```

## 📁 Input structure

Single experiment:

```text
ROOT_FOLDER/
├── 1dai/
│   └── plate_folder/
│       └── image_preview.tif
└── 2dai/
    └── plate_folder/
        └── image_preview.tif
```

Multiple experiments:

```text
ROOT_FOLDER/
├── MB0504/
│   └── 1dai/
│       └── plate_folder/
│           └── image_preview.tif
└── MB0505/
    └── 1dai/
        └── plate_folder/
            └── image_preview.tif
```

## ▶️ Run

```bash
python main.py
```

The pipeline performs preprocessing, segmentation, MaxRGB analysis, lane assignment, CSV export, and HTML report generation.

## 📤 Outputs

Typical output:

```text
OUTPUT_ROOT/
├── MB0504_summary.csv
├── MB0504_lane_summary.csv
├── MB0504_report_index.html
└── 1dai/
    ├── MB0504_1dai_all_plates_report.html
    └── plate_folder/
        ├── image_preview_wb.png
        ├── image_preview_boxes.png
        ├── image_preview_poly.png
        └── image_preview_maxrgb.png
```

The reports use:

| Class | Label |
|---|---|
| `0` | Fusarium infection |
| `1` | Root tissue |

Class names can be changed in `config.py`:

```python
CLASS_DISPLAY_NAMES = {
    0: "Fusarium infection",
    1: "Root tissue",
}
```

## 🔬 Severity metric

MaxRGB keeps the dominant RGB channel at each pixel and calculates the mean intensity inside each segmentation mask:

```python
maximumRGB_intensity = maximum_R + maximum_G + maximum_B
mean_intensity = np.mean(maximumRGB_intensity[mask])
```

The pipeline also reports the red channel after MaxRGB filtering for a stricter red-dominant signal.

## ⚡ Benchmark

The full end-to-end pipeline was benchmarked on **2,268 images** using the CPU.

| Metric             | Result |
|--------------------|---|
| CPU                | 2 × Intel Xeon E5-2643 v3 |
| CPU resources      | 8 cores / 16 logical processors |
| RAM                | 128 GB |
| GPU (off)          | NVIDIA GeForce RTX 3090 |
| Dataset            | 2,268 images |
| Total CPU runtime  | 10,167.98 s (2 h 49 min 28 s) |
| Average runtime    | 4.48 s/image |
| Throughput         | ~0.223 images/s (~13.38 images/min) |
| Observed RAM usage | ~11 GB |

Runtime includes **image loading, preprocessing, inference, post-processing, and output generation**, not inference alone. RAM usage is an indicative Task Manager observation rather than a run-wide average.

## 🏋️ YOLO training

Standalone training utilities are available in:

```text
training/
├── training_config.py
├── training_overview.py
└── train_yolo.py
```

Run training:

```bash
cd training
python train_yolo.py
```

Create only the training overview:

```bash
python train_yolo.py --destination_path PATH_TO_RUNS --overview_only
```

This generates:

```text
model_training_overview.csv
```

## 🧹 Remove environment

```bash
conda deactivate
conda env remove -n fusvision
```

## 📜 License

GNU General Public License v3.0
