"""
Core prediction logic:
  - Walk folders
  - Preprocess images
  - Run YOLO model
  - Save visualizations (white-balanced, boxes, polygons, MaxRGB)
  - Accumulate object metrics and lane infection summaries
"""

import cv2
from ultralytics import YOLO

from config import CLASS_COLORS, N_LANES
from image_processing import preprocess_image
from io_utils import (
    build_output_paths,
    create_experiment_rows_dict,
    iter_preview_images,
    parse_metadata,
    write_experiment_csvs,
)
from metrics import compute_bbox_metrics, create_lane_summary
from visualization import draw_boxes_on_image, draw_polygons_on_image


class MacrobotPredictor:
    def __init__(self,
                 model_path: str,
                 imgsz: int = 1024,
                 device: str = "cpu",
                 show: bool = False,
                 save_yolo_runs: bool = False,
                 clip_percent: float = 0.05,
                 class_colors=None,
                 fusarium_class_id: int = 0,
                 n_lanes: int = N_LANES):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.device = device
        self.show = show
        self.save_yolo_runs = save_yolo_runs
        self.clip_percent = clip_percent
        self.class_colors = class_colors if class_colors is not None else CLASS_COLORS
        self.fusarium_class_id = fusarium_class_id
        self.n_lanes = n_lanes

    def predict_folder(self,
                       root_folder: str,
                       output_root: str,
                       file_suffix: str = "preview.tif"):
        """
        Walk the folder structure, preprocess images, run YOLO prediction,
        save images, and create per-experiment CSV files.
        """
        all_images = list(iter_preview_images(root_folder, file_suffix))
        if not all_images:
            print(f"No '*{file_suffix}' images found under: {root_folder}")
            return

        print(f"Found {len(all_images)} images ending with '{file_suffix}' under:")
        print(f"  {root_folder}")
        print("Starting prediction...\n")

        experiment_rows = create_experiment_rows_dict()

        for idx, img_path in enumerate(sorted(all_images)):
            try:
                print(f"[{idx + 1}/{len(all_images)}] Processing: {img_path}")

                experiment, timepoint, plate_id = parse_metadata(img_path, root_folder)
                image_name = img_path.name

                preprocessed_bgr, maxrgb_bgr, maxrgb_red = preprocess_image(img_path)

                out_boxes_path, out_poly_path, out_maxrgb_path, preprocessed_bgr_path = build_output_paths(
                    img_path, root_folder, output_root
                )

                if not cv2.imwrite(str(out_maxrgb_path), maxrgb_bgr):
                    print(f"  !! Could not write MaxRGB image to: {out_maxrgb_path}")
                else:
                    print(f"  -> saved MaxRGB image to: {out_maxrgb_path}")

                if not cv2.imwrite(str(preprocessed_bgr_path), preprocessed_bgr):
                    print(f"  !! Could not write white-balanced image to: {preprocessed_bgr_path}")
                else:
                    print(f"  -> saved white-balanced image to: {preprocessed_bgr_path}")

                results = self.model.predict(
                    source=preprocessed_bgr,
                    imgsz=self.imgsz,
                    device=self.device,
                    show=self.show,
                    save=self.save_yolo_runs,
                    verbose=False,
                )

                if not results:
                    print("  -> no result returned from model")
                    continue

                r = results[0]
                n_boxes = len(r.boxes) if hasattr(r, "boxes") and r.boxes is not None else 0
                n_masks = r.masks.data.shape[0] if hasattr(r, "masks") and r.masks is not None else 0
                print(f"  -> {n_boxes} boxes detected, {n_masks} masks found")

                img_boxes = draw_boxes_on_image(preprocessed_bgr, r, self.class_colors)
                img_poly = draw_polygons_on_image(preprocessed_bgr, r, self.class_colors)

                if not cv2.imwrite(str(out_boxes_path), img_boxes):
                    print(f"  !! Could not write boxes image to: {out_boxes_path}")
                else:
                    print(f"  -> saved boxes image to: {out_boxes_path}")

                if not cv2.imwrite(str(out_poly_path), img_poly):
                    print(f"  !! Could not write polygon image to: {out_poly_path}")
                else:
                    print(f"  -> saved polygon image to: {out_poly_path}")

                rows = compute_bbox_metrics(
                    result=r,
                    maxrgb_bgr=maxrgb_bgr,
                    experiment=experiment,
                    timepoint=timepoint,
                    plate_id=plate_id,
                    image_name=image_name,
                    fusarium_class_id=self.fusarium_class_id,
                    n_lanes=self.n_lanes,
                )
                if rows:
                    experiment_rows[experiment].extend(rows)

            except Exception as e:
                print(f"ERROR processing {img_path}: {e}")

        lane_summary_by_experiment = {
            experiment: create_lane_summary(rows)
            for experiment, rows in experiment_rows.items()
        }

        write_experiment_csvs(experiment_rows, output_root, lane_summary_by_experiment)
