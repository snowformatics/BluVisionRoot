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
                         hsv_v: np.ndarray,
                         experiment: str,
                         timepoint: str,
                         plate_id: str,
                         image_name: str,
                         fusarium_class_id: int,
                         n_lanes: int = 4) -> List[dict]:
    """
    For each YOLO object/mask compute object-level metrics.

    MaxRGB and HSV-V metrics are calculated inside every polygon:
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
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
    names = result.names

    masks = None
    if hasattr(result, "masks") and result.masks is not None:
        if result.masks.data is not None:
            masks = result.masks.data.cpu().numpy()  # (N, H, W), float 0/1

    h_img, w_img = maxrgb_bgr.shape[:2]
    if hsv_v.shape[:2] != (h_img, w_img):
        hsv_v = cv2.resize(hsv_v, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
    hsv_v_float = hsv_v.astype(np.float32)

    # Comparable to maximumRGB_intensity_mean in the comparison script.
    maxrgb_float = maxrgb_bgr.astype(np.float32)
    maxrgb_intensity = np.sum(maxrgb_float, axis=2)

    # Additional red-channel-only metric after MaxRGB filtering.
    # maxrgb_bgr is BGR, therefore channel index 2 is red.
    maxrgb_red_only = maxrgb_float[:, :, 2]

    # Normalize all masks to the preprocessed image size once, so overlap can be
    # calculated consistently between polygons. For each object below,
    # overlap_area_px = pixels where this object's mask overlaps any other mask.
    # overlap_percent = overlap_area_px / this object's polygon_area_px * 100.
    mask_bool_list = []
    if masks is not None:
        for mask in masks:
            mask_bool = mask > 0.5
            if mask_bool.shape != maxrgb_intensity.shape:
                mask_bool = cv2.resize(
                    mask_bool.astype(np.uint8),
                    (w_img, h_img),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            mask_bool_list.append(mask_bool)

    if mask_bool_list:
        mask_stack = np.stack(mask_bool_list, axis=0)
        mask_overlap_counts = np.sum(mask_stack, axis=0)
    else:
        mask_overlap_counts = None

    # Bounding-box overlaps are useful for detecting the visual overlaps you see
    # in the report images. Mask/polygon overlap can be 0 when YOLO produces
    # non-overlapping instance masks, even though the enclosing boxes overlap.
    bbox_overlap_area_by_idx = []
    bbox_overlap_percent_by_idx = []
    bbox_overlap_indices_by_idx = []
    for i, b in enumerate(xyxy):
        ax1, ay1, ax2, ay2 = b.astype(int)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        union_overlap_mask = np.zeros((h_img, w_img), dtype=bool)
        overlapping_indices = []
        for j, other in enumerate(xyxy):
            if i == j:
                continue
            bx1, by1, bx2, by2 = other.astype(int)
            ix1 = max(ax1, bx1, 0)
            iy1 = max(ay1, by1, 0)
            ix2 = min(ax2, bx2, w_img)
            iy2 = min(ay2, by2, h_img)
            if ix2 > ix1 and iy2 > iy1:
                union_overlap_mask[iy1:iy2, ix1:ix2] = True
                overlapping_indices.append(int(j))
        bbox_overlap_area = int(np.count_nonzero(union_overlap_mask))
        bbox_overlap_percent = float(100.0 * bbox_overlap_area / area_a) if area_a > 0 else 0.0
        bbox_overlap_area_by_idx.append(bbox_overlap_area)
        bbox_overlap_percent_by_idx.append(bbox_overlap_percent)
        bbox_overlap_indices_by_idx.append(",".join(map(str, sorted(set(overlapping_indices)))))

    for idx, box in enumerate(xyxy):
        x1, y1, x2, y2 = box.astype(int)
        bbox_width = int(max(0, x2 - x1))
        bbox_height = int(max(0, y2 - y1))
        confidence = float(confs[idx]) if confs is not None and idx < len(confs) else 0.0
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
        mask_overlap_area_px = 0
        mask_overlap_percent = 0.0
        bbox_overlap_area_px = bbox_overlap_area_by_idx[idx] if idx < len(bbox_overlap_area_by_idx) else 0
        bbox_overlap_percent = bbox_overlap_percent_by_idx[idx] if idx < len(bbox_overlap_percent_by_idx) else 0.0
        overlap_object_indices = bbox_overlap_indices_by_idx[idx] if idx < len(bbox_overlap_indices_by_idx) else ""

        # Backward-compatible overlap columns now represent BBOX overlap,
        # because these are the overlaps visible in the boxes report image.
        overlap_area_px = bbox_overlap_area_px
        overlap_percent = bbox_overlap_percent
        fusarium_severity = 0.0
        fusarium_severity_red = 0.0
        control_severity = 0.0
        control_severity_red = 0.0
        fusarium_severity_hsv_v = 0.0
        control_severity_hsv_v = 0.0
        mean_maxrgb_intensity = 0.0
        mean_maxrgb_red = 0.0
        mean_hsv_v = 0.0
        polygon_centroid_x = 0.0
        polygon_centroid_y = 0.0
        polygon_x_min = 0
        polygon_y_min = 0
        polygon_x_max = 0
        polygon_y_max = 0

        mask_bool = None
        if idx < len(mask_bool_list):
            mask_bool = mask_bool_list[idx]

            polygon_area_px = int(np.count_nonzero(mask_bool))

            # Derive polygon position directly from the resized binary mask.
            # This is more robust than result.masks.xy because masks.xy can be
            # empty or not aligned with the preprocessed image in some YOLO versions.
            if polygon_area_px > 0:
                ys, xs = np.nonzero(mask_bool)
                polygon_x_min = int(xs.min())
                polygon_y_min = int(ys.min())
                polygon_x_max = int(xs.max())
                polygon_y_max = int(ys.max())
                polygon_centroid_x = float(xs.mean())
                polygon_centroid_y = float(ys.mean())

            if polygon_area_px > 0 and mask_overlap_counts is not None:
                mask_overlap_area_px = int(np.count_nonzero(mask_bool & (mask_overlap_counts > 1)))
                mask_overlap_percent = float(100.0 * mask_overlap_area_px / polygon_area_px)

            if cls_id == 0:
                pixels_class0 = polygon_area_px
                pixels_fusarium_infection = polygon_area_px
            elif cls_id == 1:
                pixels_class1 = polygon_area_px
                pixels_root_tissue = polygon_area_px

            if polygon_area_px > 0:
                intensity_values = maxrgb_intensity[mask_bool]
                red_values = maxrgb_red_only[mask_bool]
                hsv_v_values = hsv_v_float[mask_bool]
                mean_maxrgb_intensity = float(np.mean(intensity_values))
                mean_maxrgb_red = float(np.mean(red_values))
                mean_hsv_v = float(np.mean(hsv_v_values))

                if cls_id == fusarium_class_id:
                    fusarium_severity = mean_maxrgb_intensity
                    fusarium_severity_red = mean_maxrgb_red
                    fusarium_severity_hsv_v = mean_hsv_v
                else:
                    control_severity = mean_maxrgb_intensity
                    control_severity_red = mean_maxrgb_red
                    control_severity_hsv_v = mean_hsv_v

        # Safety fallback: if an old/empty polygon coordinate extraction would leave
        # coordinates at 0 despite a detected object, use the bbox extent so the
        # report never shows impossible 0,0 positions for real detections.
        # With valid masks, the values above are real mask-derived positions.
        if polygon_area_px > 0 and polygon_x_min == 0 and polygon_y_min == 0 and polygon_x_max == 0 and polygon_y_max == 0:
            polygon_x_min = int(x1)
            polygon_y_min = int(y1)
            polygon_x_max = int(x2)
            polygon_y_max = int(y2)
            polygon_centroid_x = float((x1 + x2) / 2.0)
            polygon_centroid_y = float((y1 + y2) / 2.0)

        rows.append({
            "experiment": experiment,
            "timepoint": timepoint,
            "plate_id": plate_id,
            "image": image_name,
            "bbox_index": idx,
            "class_id": cls_id,
            "class_name": cls_name,
            "class_display_name": class_display_name,
            "confidence": confidence,
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "bbox_width": bbox_width,
            "bbox_height": bbox_height,
            "box_center_x": box_center_x,
            "box_center_y": box_center_y,
            "lane_id": lane_id,
            "lane_x_start": lane_x_start,
            "lane_x_end": lane_x_end,
            "relative_x_position": box_center_x / w_img if w_img > 0 else 0.0,
            "polygon_centroid_x": polygon_centroid_x,
            "polygon_centroid_y": polygon_centroid_y,
            "polygon_x_min": polygon_x_min,
            "polygon_y_min": polygon_y_min,
            "polygon_x_max": polygon_x_max,
            "polygon_y_max": polygon_y_max,
            "polygon_area_px": polygon_area_px,
            "overlap_area_px": overlap_area_px,
            "overlap_percent": overlap_percent,
            "bbox_overlap_area_px": bbox_overlap_area_px,
            "bbox_overlap_percent": bbox_overlap_percent,
            "mask_overlap_area_px": mask_overlap_area_px,
            "mask_overlap_percent": mask_overlap_percent,
            "overlap_object_indices": overlap_object_indices,
            "pixels_fusarium_infection": pixels_fusarium_infection,
            "pixels_root_tissue": pixels_root_tissue,
            "pixels_class0": pixels_class0,
            "pixels_class1": pixels_class1,
            "fusarium_severity_maxrgb_intensity": fusarium_severity,
            "fusarium_severity_maxrgb_red": fusarium_severity_red,
            "fusarium_severity_hsv_v": fusarium_severity_hsv_v,
            "control_severity_maxrgb_intensity": control_severity,
            "control_severity_maxrgb_red": control_severity_red,
            "control_severity_hsv_v": control_severity_hsv_v,
            # Internal helpers used for summary; removed if not listed in CSV columns.
            "mean_maxrgb_intensity": mean_maxrgb_intensity,
            "mean_maxrgb_red": mean_maxrgb_red,
            "mean_hsv_v": mean_hsv_v,
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
            mean_hsv_v = _safe_mean([
                float(r.get("mean_hsv_v", 0.0) or 0.0) for r in class_rows
            ])

            for prefix in class_prefixes[class_id]:
                summary[f"{prefix}_polygon_count"] = class_count
                summary[f"{prefix}_polygon_percent"] = object_percent
                summary[f"{prefix}_area_px"] = class_area
                summary[f"{prefix}_area_percent"] = area_percent
                summary[f"{prefix}_mean_maxrgb_intensity"] = mean_intensity
                summary[f"{prefix}_mean_maxrgb_red"] = mean_red
                summary[f"{prefix}_mean_hsv_v"] = mean_hsv_v

        summary_rows.append(summary)

    return summary_rows
