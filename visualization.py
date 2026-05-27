
"""
Visualization helpers:
  - Drawing bounding boxes
  - Drawing polygons from segmentation masks
"""

from typing import Dict

import cv2
import numpy as np


def draw_boxes_on_image(img_bgr: np.ndarray, result, class_colors: Dict[int, tuple]):
    """
    Draw bounding boxes (and labels) from a YOLO result onto an image.

    Args:
        img_bgr: BGR image to draw on.
        result: single result object from YOLOv8.
        class_colors: mapping class_id -> BGR color tuple.

    Returns:
        Image with bounding boxes drawn (copy of input).
    """
    img_draw = img_bgr.copy()

    if not hasattr(result, "boxes") or result.boxes is None:
        return img_draw

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
    cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else None
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
    names = result.names  # dict: id -> class name

    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = box.astype(int)

        cls_id = cls_ids[i] if cls_ids is not None and i < len(cls_ids) else -1
        color = class_colors.get(cls_id, (255, 255, 255))  # default white

        # draw rectangle
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

        # label text
        label_parts = []
        if cls_ids is not None and i < len(cls_ids):
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            label_parts.append(cls_name)
        if confs is not None and i < len(confs):
            label_parts.append(f"{confs[i]:.2f}")
        label = " ".join(label_parts)

        if label:
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img_draw, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
            cv2.putText(img_draw, label, (x1, y1 - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    return img_draw


def draw_polygons_on_image(img_bgr: np.ndarray, result, class_colors: Dict[int, tuple]):
    """
    Draw polygon contours from YOLO masks onto a copy of the image.

    Args:
        img_bgr: BGR image to draw on.
        result: single result object from YOLOv8 (with segmentation masks).
        class_colors: mapping class_id -> BGR color tuple.

    Returns:
        Image with polygon outlines (copy of input).
    """
    img_draw = img_bgr.copy()

    if not hasattr(result, "masks") or result.masks is None:
        return img_draw

    masks_xy = result.masks.xy  # list of [N_polys x 2] arrays
    boxes = result.boxes
    cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else None
    names = result.names

    for i, poly_xy in enumerate(masks_xy):
        cls_id = cls_ids[i] if cls_ids is not None and i < len(cls_ids) else -1
        color = class_colors.get(cls_id, (255, 255, 255))

        pts = np.array(poly_xy, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_draw, [pts], isClosed=True, color=color, thickness=2)

        # optional: class label near first polygon point
        if len(pts) > 0 and names is not None:
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            cv2.putText(img_draw, cls_name,
                        (int(pts[0][0][0]), int(pts[0][0][1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    return img_draw
