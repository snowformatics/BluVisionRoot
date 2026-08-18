
"""
Entry point for running the macrobot Fusarium prediction pipeline.

Usage (from this folder):
    python main.py
"""

from config import (
    CLASS_COLORS,
    CLIP_PERCENT,
    CONF_THRESHOLD,
    DEVICE,
    FILE_SUFFIX,
    FUSARIUM_CLASS_ID,
    IMG_SIZE,
    IOU_THRESHOLD,
    N_LANES,
    MODEL_PATH,
    OUTPUT_ROOT,
    ROOT_FOLDER,
    SAVE_YOLO_RUNS,
    SHOW,
)
from predictor import MacrobotPredictor
from html_report import generate_html_reports


def main():
    predictor = MacrobotPredictor(
        model_path=MODEL_PATH,
        imgsz=IMG_SIZE,
        device=DEVICE,
        show=SHOW,
        save_yolo_runs=SAVE_YOLO_RUNS,
        clip_percent=CLIP_PERCENT,
        class_colors=CLASS_COLORS,
        fusarium_class_id=FUSARIUM_CLASS_ID,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        n_lanes=N_LANES,
    )
    predictor.predict_folder(ROOT_FOLDER, OUTPUT_ROOT, FILE_SUFFIX)
    generate_html_reports(OUTPUT_ROOT, fusarium_class_id=FUSARIUM_CLASS_ID)


if __name__ == "__main__":
    main()


