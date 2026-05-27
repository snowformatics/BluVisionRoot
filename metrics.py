"""
Computing per-object and per-lane metrics:
  - Pixel counts per class inside polygon
  - Mean MaxRGB intensity inside polygon, compatible with the comparison script
  - Lane assignment by horizontal x-position
  - Infection percentage per class within each lane
"""

from collections import defaultdict
from typing import Dict, List

import cv2
import numpy as np

from config import CLASS_DISPLAY_NAMES


def assign_lane_from_x(x_center: float, image_width: int, n_lanes: int = 4) -> int:
    """Assign object to lane 1..n_lanes based on bbox center x-position."""
    if image_width <= 0:
        return 1
    lane = int((x_center / image_width) * n_lanes) + 1
    return max(1, min(n_lanes, lane))


def get_lane_boundaries(image_width: int, lane_id: int, n_lanes: int = 4):
    """Return x-start and x-end for a lane."""
    x_start = int((lane_id - 1) * image_width / n_lanes)
    x_end = int(lane_id * image_width / n_lanes)
    return x_start, x_end


def _safe_mean(values):
    vals = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(vals)) if vals else 0.0


def compute_bbox_metrics(result,
                         maxrgb_bgr: np.ndarray,
                         experiment: str,
                         timepoint: str,
                         plate_id: str,
                         image_name: str,
                         fusarium_class_id: int,
                         n_lanes: int = 4) -> List[dict]:
    """
    For each YOLO object/mask compute object-level metrics.

    Two MaxRGB metrics are calculated inside every polygon:
      1) maxrgb_intensity = maximum_B + maximum_G + maximum_R
         This matches the comparison script's maximumRGB_intensity_mean.
      2) maxrgb_red = maximum_R only
         This keeps the older red-channel-only severity as an additional parameter.
    """
    rows: List[dict] = []

    if not hasattr(result, "boxes") or result.boxes is None:
        return rows

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
    cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else None
    names = result.names

    masks = None
    if hasattr(result, "masks") and result.masks is not None and result.masks.data is not None:
        masks = result.masks.data.cpu().numpy()  # (N, H, W), float 0/1

    h_img, w_img = maxrgb_bgr.shape[:2]

    # Comparable to maximumRGB_intensity_mean in the comparison script.
    maxrgb_float = maxrgb_bgr.astype(np.float32)
    maxrgb_intensity = np.sum(maxrgb_float, axis=2)

    # Additional red-channel-only metric after MaxRGB filtering.
    # maxrgb_bgr is BGR, therefore channel index 2 is red.
    maxrgb_red_only = maxrgb_float[:, :, 2]

    for idx, box in enumerate(xyxy):
        x1, y1, x2, y2 = box.astype(int)
        box_center_x = float((x1 + x2) / 2.0)
        box_center_y = float((y1 + y2) / 2.0)
        lane_id = assign_lane_from_x(box_center_x, w_img, n_lanes=n_lanes)
        lane_x_start, lane_x_end = get_lane_boundaries(w_img, lane_id, n_lanes=n_lanes)

        cls_id = cls_ids[idx] if cls_ids is not None and idx < len(cls_ids) else -1
        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        class_display_name = CLASS_DISPLAY_NAMES.get(cls_id, cls_name)

        pixels_class0 = 0
        pixels_class1 = 0
        pixels_fusarium_infection = 0
        pixels_root_tissue = 0
        polygon_area_px = 0
        fusarium_severity = 0.0
        fusarium_severity_red = 0.0
        control_severity = 0.0
        control_severity_red = 0.0
        mean_maxrgb_intensity = 0.0
        mean_maxrgb_red = 0.0

        if masks is not None and idx < masks.shape[0]:
            mask_bool = masks[idx] > 0.5

            if mask_bool.shape != maxrgb_intensity.shape:
                mask_bool = cv2.resize(
                    mask_bool.astype(np.uint8),
                    (w_img, h_img),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            polygon_area_px = int(np.count_nonzero(mask_bool))

            if cls_id == 0:
                pixels_class0 = polygon_area_px
                pixels_fusarium_infection = polygon_area_px
            elif cls_id == 1:
                pixels_class1 = polygon_area_px
                pixels_root_tissue = polygon_area_px

            if polygon_area_px > 0:
                intensity_values = maxrgb_intensity[mask_bool]
                red_values = maxrgb_red_only[mask_bool]
                mean_maxrgb_intensity = float(np.mean(intensity_values))
                mean_maxrgb_red = float(np.mean(red_values))

                if cls_id == fusarium_class_id:
                    fusarium_severity = mean_maxrgb_intensity
                    fusarium_severity_red = mean_maxrgb_red
                else:
                    control_severity = mean_maxrgb_intensity
                    control_severity_red = mean_maxrgb_red

        rows.append({
            "experiment": experiment,
            "timepoint": timepoint,
            "plate_id": plate_id,
            "image": image_name,
            "bbox_index": idx,
            "class_id": cls_id,
            "class_name": cls_name,
            "class_display_name": class_display_name,
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "box_center_x": box_center_x,
            "box_center_y": box_center_y,
            "lane_id": lane_id,
            "lane_x_start": lane_x_start,
            "lane_x_end": lane_x_end,
            "relative_x_position": box_center_x / w_img if w_img > 0 else 0.0,
            "polygon_area_px": polygon_area_px,
            "pixels_fusarium_infection": pixels_fusarium_infection,
            "pixels_root_tissue": pixels_root_tissue,
            "pixels_class0": pixels_class0,
            "pixels_class1": pixels_class1,
            "fusarium_severity_maxrgb_intensity": fusarium_severity,
            "fusarium_severity_maxrgb_red": fusarium_severity_red,
            "control_severity_maxrgb_intensity": control_severity,
            "control_severity_maxrgb_red": control_severity_red,
            # Internal helpers used for summary; removed if not listed in CSV columns.
            "mean_maxrgb_intensity": mean_maxrgb_intensity,
            "mean_maxrgb_red": mean_maxrgb_red,
        })

    return rows


def create_lane_summary(rows: List[dict]) -> List[dict]:
    """
    Create lane-level infection summaries.

    For each image/lane:
      - polygon percent per class = class polygon count / total polygons in lane * 100
      - area percent per class = class polygon area / total polygon area in lane * 100
      - class mean MaxRGB intensity = mean of object intensities for that class in lane
    """
    grouped: Dict[tuple, List[dict]] = defaultdict(list)

    for row in rows:
        key = (
            row.get("experiment", ""),
            row.get("timepoint", ""),
            row.get("plate_id", ""),
            row.get("image", ""),
            row.get("lane_id", 0),
        )
        grouped[key].append(row)

    summary_rows: List[dict] = []

    for key, group_rows in sorted(grouped.items()):
        experiment, timepoint, plate_id, image_name, lane_id = key
        lane_area = float(sum(float(r.get("polygon_area_px", 0) or 0) for r in group_rows))
        lane_count = len(group_rows)

        summary = {
            "experiment": experiment,
            "timepoint": timepoint,
            "plate_id": plate_id,
            "image": image_name,
            "lane_id": lane_id,
            "lane_x_start": group_rows[0].get("lane_x_start", 0),
            "lane_x_end": group_rows[0].get("lane_x_end", 0),
            "lane_polygon_count_total": lane_count,
            "lane_polygon_area_total_px": lane_area,
        }

        class_prefixes = {
            0: ("class0", "fusarium_infection"),
            1: ("class1", "root_tissue"),
        }

        for class_id in (0, 1):
            class_rows = [r for r in group_rows if int(r.get("class_id", -1)) == class_id]
            class_count = len(class_rows)
            class_area = float(sum(float(r.get("polygon_area_px", 0) or 0) for r in class_rows))
            object_percent = 100.0 * class_count / lane_count if lane_count else 0.0
            area_percent = 100.0 * class_area / lane_area if lane_area > 0 else 0.0
            mean_intensity = _safe_mean([
                float(r.get("mean_maxrgb_intensity", 0.0) or 0.0) for r in class_rows
            ])
            mean_red = _safe_mean([
                float(r.get("mean_maxrgb_red", 0.0) or 0.0) for r in class_rows
            ])

            for prefix in class_prefixes[class_id]:
                summary[f"{prefix}_polygon_count"] = class_count
                summary[f"{prefix}_polygon_percent"] = object_percent
                summary[f"{prefix}_area_px"] = class_area
                summary[f"{prefix}_area_percent"] = area_percent
                summary[f"{prefix}_mean_maxrgb_intensity"] = mean_intensity
                summary[f"{prefix}_mean_maxrgb_red"] = mean_red

        summary_rows.append(summary)

    return summary_rows
