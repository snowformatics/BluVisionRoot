
"""
Image preprocessing utilities:
  - White balance (percentile clipping)
  - MaxRGB filter
  - HSV value channel extraction
  - Overall preprocessing for YOLO & severity analysis
"""

from pathlib import Path
import cv2
import numpy as np


def maxrgb_filter(bgr_img: np.ndarray):
    """
    Apply MaxRGB filter:
    For each pixel, keep only the channel that has the maximum intensity,
    set the other two channels to zero.

    Args:
        bgr_img: BGR image (uint8).

    Returns:
        feature_image (BGR), red_channel (2D array) after MaxRGB filtering.
    """
    # Split BGR correctly
    b, g, r = cv2.split(bgr_img)

    # Max per pixel
    m = np.maximum(np.maximum(r, g), b)

    b2 = b.copy()
    g2 = g.copy()
    r2 = r.copy()

    b2[b2 < m] = 0
    g2[g2 < m] = 0
    r2[r2 < m] = 0

    feature_image = cv2.merge([b2, g2, r2])

    return feature_image, r2


def hsv_value_channel(bgr_img: np.ndarray) -> np.ndarray:
    """
    Extract the V (value/brightness) channel from HSV.

    Args:
        bgr_img: preprocessed BGR image (uint8).

    Returns:
        hsv_v: single-channel uint8 value/brightness image, where 0 = dark
               and 255 = bright.
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    _h, _s, v = cv2.split(hsv)
    return v


def convert_to_8bit(img):
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)
    return img


def fix_channels_safe(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def crop_fixed_border(img, top, bottom, left, right):
    h, w = img.shape[:2]
    return img[top:h - bottom, left:w - right]


def wb_helper(channel, perc=0.05):
    low, high = np.percentile(channel, (perc, 100 - perc))
    channel = np.clip(channel, low, high)
    channel = ((channel - low) / (high - low + 1e-6)) * 255
    return channel.astype(np.uint8)


def white_balance_percentile(image, perc=0.05):
    image_split = np.dsplit(image, image.shape[-1])
    return np.dstack([
        wb_helper(image_split[0], perc),
        wb_helper(image_split[1], perc),
        wb_helper(image_split[2], perc)
    ]).astype(np.uint8)


def mild_contrast_normalization(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=1.2,
        tileGridSize=(8, 8)
    )

    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))

    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def preprocess_image(img_path):
    CROP_TOP = 150
    CROP_BOTTOM = 150
    CROP_LEFT = 100
    CROP_RIGHT = 100

    WB_PERCENTILE = 0.05
    APPLY_CONTRAST = False
    img = cv2.imread(str(img_path))

    # ============================================================
    # PREPROCESSING
    # ============================================================

    img = convert_to_8bit(img)
    img = fix_channels_safe(img)

    img = crop_fixed_border(
        img,
        top=CROP_TOP,
        bottom=CROP_BOTTOM,
        left=CROP_LEFT,
        right=CROP_RIGHT
    )

    bgr_wb = white_balance_percentile(img, perc=WB_PERCENTILE)

    if APPLY_CONTRAST:
        bgr_wb = mild_contrast_normalization(bgr_wb)

    maxrgb_bgr, maxrgb_red = maxrgb_filter(bgr_wb)
    hsv_v = hsv_value_channel(bgr_wb)

    return bgr_wb, maxrgb_bgr, maxrgb_red, hsv_v

    return img
    # """
    # Load image (BGR), convert to RGB, resize by 0.5, white balance each channel,
    # merge and convert back to BGR (for YOLO/OpenCV). Also computes the MaxRGB
    # image and red channel after MaxRGB.
    #
    # Args:
    #     img_path: Path to input image.
    #     clip_percent: percentile clipping amount for white balance.
    #
    # Returns:
    #     bgr_wb: preprocessed BGR image (for YOLO).
    #     maxrgb_bgr: MaxRGB BGR image (for saving / visualization).
    #     maxrgb_red: single-channel red image after MaxRGB (for severity).
    # """
    # bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    # if bgr is None:
    #     raise RuntimeError(f"Could not read image: {img_path}")
    #
    # # Convert BGR -> RGB for processing
    # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    #
    # # resize 0.5 x 0.5
    # image = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    #
    # # Split the 3-channel image into separate channels
    # image_split = np.dsplit(image, image.shape[-1])
    #
    # # Apply white balance to each channel and merge them back into an image
    # image_wb = np.dstack([
    #     wb_helper(image_split[0], clip_percent),
    #     wb_helper(image_split[1], clip_percent),
    #     wb_helper(image_split[2], clip_percent),
    # ])
    #
    # # Convert back to BGR for YOLO / OpenCV
    # bgr_wb = cv2.cvtColor(image_wb.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # MaxRGB filter for severity & saving
    maxrgb_bgr, maxrgb_red = maxrgb_filter(bgr_wb)
    hsv_v = hsv_value_channel(bgr_wb)

    return bgr_wb, maxrgb_bgr, maxrgb_red, hsv_v
