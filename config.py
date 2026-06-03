"""
Configuration for macrobot Fusarium prediction & severity analysis.
Adjust paths and options here; the rest of the code imports from this module.
"""

from pathlib import Path

# -------------------- user settings --------------------

#MODEL_PATH = r"yolov8s-seg_img1024_ep150_AdamW_lr0.0005_best.pt"
#MODEL_PATH = r"D:\stefanie\fus_root_high_02\yolov8s-seg_img1024_ep150_AdamW_lr0.0005\weights\best.pt"
MODEL_PATH = r"D:/stefanie/02_split_model/yolov8s-seg_img1024_ep150_AdamW_lr0.0005/weights/best.pt"
# Input root:
#   ROOT_FOLDER / MB0001 / 10dai / 20250727_154933__F35-4 / ...preview.tif
#ROOT_FOLDER = r"\\psg-09\Mikroskop\Exchange\!to_analyze\macrobot\fusarium\MB0361"
#ROOT_FOLDER = r"\\hsm\AGR-BIM\macrobot\MB0560"
ROOT_FOLDER = r"D:\stefanie\screening"
# Output root – same structure (experiment / timepoint / plate),
# but filenames are clearly marked as predictions.
#OUTPUT_ROOT = r"\\psg-09\Mikroskop\Images\FusVision\MB0361_yolov8s_test"
#OUTPUT_ROOT = r"D:\stefanie\MB0560_002"
OUTPUT_ROOT = r"D:\stefanie\screening\out"
# Folder layout of ROOT_FOLDER.
# False means: ROOT_FOLDER / timepoint / plate_id / image
# True means:  ROOT_FOLDER / experiment / timepoint / plate_id / image
# Your current MB0504 path uses False, because MB0504 is already the experiment root.
INPUT_HAS_EXPERIMENT_LEVEL = True
EXPERIMENT_NAME = Path(ROOT_FOLDER).name

# ONLY process these (experiment, timepoint) combinations.
# Example: ONLY_COMBINATIONS = [("MB0001", "10dai"), ("MB0002", "11dai")]
ONLY_COMBINATIONS = []   # empty = process everything

# YOLO inference image size
IMG_SIZE = 1024

# "cpu" or "0" for first GPU
DEVICE = "cpu"

# Display YOLO window?
SHOW = False

# True = also let YOLO save into runs/...
SAVE_YOLO_RUNS = False

# Images to process must end with this suffix
FILE_SUFFIX = "preview.tif"

# For white-balance helper (percentile clipping)
CLIP_PERCENT = 0.05

# Number of vertical lanes to assign by x-position.
N_LANES = 4

# Class colors (BGR).
# Here: class 0 = Fusarium/red overlay, class 1 = control/green overlay.
CLASS_COLORS = {
    1: (0, 255, 0),   # green for class 1
    0: (0, 0, 255),   # red for class 0 (fusarium)
}

# Which class is Fusarium for severity calculation?
FUSARIUM_CLASS_ID = 0

# Human-readable class names used in HTML reports.
# Adjust if your model class order changes.
CLASS_DISPLAY_NAMES = {
    0: "Fusarium infection",
    1: "Root tissue",
}

# Output CSV column names for per-object / per-polygon table.
# NOTE: fusarium_severity_maxrgb_intensity is now comparable to the comparison script:
#       it is mean(maximum_R + maximum_G + maximum_B) inside the polygon.
CSV_COLUMNS = [
    "experiment",
    "timepoint",
    "plate_id",
    "image",
    "bbox_index",
    "class_id",
    "class_name",
    "class_display_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "box_center_x",
    "box_center_y",
    "lane_id",
    "lane_x_start",
    "lane_x_end",
    "relative_x_position",
    "polygon_area_px",
    "pixels_fusarium_infection",
    "pixels_root_tissue",
    "pixels_class0",
    "pixels_class1",
    "fusarium_severity_maxrgb_intensity",
    "fusarium_severity_maxrgb_red",
    "control_severity_maxrgb_intensity",
    "control_severity_maxrgb_red",
]

# Output CSV column names for per-lane / infection-percent summary.
LANE_SUMMARY_COLUMNS = [
    "experiment",
    "timepoint",
    "plate_id",
    "image",
    "lane_id",
    "lane_x_start",
    "lane_x_end",
    "lane_polygon_count_total",
    "lane_polygon_area_total_px",
    "fusarium_infection_polygon_count",
    "fusarium_infection_polygon_percent",
    "fusarium_infection_area_px",
    "fusarium_infection_area_percent",
    "fusarium_infection_mean_maxrgb_intensity",
    "fusarium_infection_mean_maxrgb_red",
    "root_tissue_polygon_count",
    "root_tissue_polygon_percent",
    "root_tissue_area_px",
    "root_tissue_area_percent",
    "root_tissue_mean_maxrgb_intensity",
    "root_tissue_mean_maxrgb_red",
    # Backward-compatible raw class columns.
    "class0_polygon_count",
    "class0_polygon_percent",
    "class0_area_px",
    "class0_area_percent",
    "class0_mean_maxrgb_intensity",
    "class0_mean_maxrgb_red",
    "class1_polygon_count",
    "class1_polygon_percent",
    "class1_area_px",
    "class1_area_percent",
    "class1_mean_maxrgb_intensity",
    "class1_mean_maxrgb_red",
]

# -------------------------------------------------------

def ensure_paths_exist():
    """Convenience helper for validating paths at runtime if needed."""
    root = Path(ROOT_FOLDER)
    out = Path(OUTPUT_ROOT)
    return root, out
