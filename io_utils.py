"""
I/O utilities for:
  - Traversing the folder structure
  - Building output paths
  - Writing per-experiment CSV summary files
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from config import CSV_COLUMNS, EXPERIMENT_NAME, INPUT_HAS_EXPERIMENT_LEVEL, LANE_SUMMARY_COLUMNS, ONLY_COMBINATIONS


def parse_metadata(img_path: Path, root_folder: str) -> Tuple[str, str, str]:
    """
    Extract metadata from the image path.

    Supported layouts:
      1) INPUT_HAS_EXPERIMENT_LEVEL = False
         ROOT_FOLDER / timepoint / plate_id / image

      2) INPUT_HAS_EXPERIMENT_LEVEL = True
         ROOT_FOLDER / experiment / timepoint / plate_id / image

    The current MB0504 setup uses layout 1, so the image filename is never
    interpreted as plate_id.
    """
    root = Path(root_folder)
    rel = img_path.relative_to(root)
    parts = rel.parts

    if INPUT_HAS_EXPERIMENT_LEVEL:
        experiment = parts[0] if len(parts) > 0 else EXPERIMENT_NAME
        timepoint = parts[1] if len(parts) > 1 else ""
        plate_id = parts[2] if len(parts) > 2 else ""
    else:
        experiment = EXPERIMENT_NAME
        timepoint = parts[0] if len(parts) > 0 else ""
        plate_id = parts[1] if len(parts) > 1 else ""

    return experiment, timepoint, plate_id


def iter_preview_images(root_folder: str, file_suffix: str) -> Iterable[Path]:
    """Recursively find all *<file_suffix> files from root_folder downward."""
    root = Path(root_folder)
    for path in root.rglob(f"*{file_suffix}"):
        if not path.is_file():
            continue
        experiment, timepoint, _ = parse_metadata(path, root_folder)
        if ONLY_COMBINATIONS and (experiment, timepoint) not in ONLY_COMBINATIONS:
            continue
        yield path


def build_output_paths(img_path: Path, root_folder: str, output_root: str):
    """
    Maintain same folder structure as ROOT_FOLDER, under OUTPUT_ROOT.

    With INPUT_HAS_EXPERIMENT_LEVEL=False:
        OUT / timepoint / plate_id / <stem>_wb.png

    With INPUT_HAS_EXPERIMENT_LEVEL=True:
        OUT / experiment / timepoint / plate_id / <stem>_wb.png
    """
    root = Path(root_folder)
    out_root = Path(output_root)
    rel = img_path.relative_to(root)
    out_dir = out_root / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = rel.stem

    out_boxes = out_dir / f"{stem}_boxes.png"
    out_poly = out_dir / f"{stem}_poly.png"
    out_maxrgb = out_dir / f"{stem}_maxrgb.png"
    out_hsv_v = out_dir / f"{stem}_hsv_v.png"
    out_preprocessed_bgr = out_dir / f"{stem}_wb.png"
    return out_boxes, out_poly, out_maxrgb, out_hsv_v, out_preprocessed_bgr


def experiment_csv_dir(output_root: str, experiment: str) -> Path:
    """Return where summary CSVs should be written for the configured layout."""
    out_root = Path(output_root)
    return out_root / experiment if INPUT_HAS_EXPERIMENT_LEVEL else out_root


def create_experiment_rows_dict() -> Dict[str, List[dict]]:
    return defaultdict(list)


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  -> wrote CSV: {path}")


def write_experiment_csvs(experiment_rows, output_root: str, lane_summary_rows=None):
    """Write one object CSV and one lane-summary CSV per experiment."""
    out_root = Path(output_root)
    print("\nWriting per-experiment CSV files...")

    lane_summary_rows = lane_summary_rows or {}

    for experiment, rows in experiment_rows.items():
        if not rows:
            continue
        exp_dir = experiment_csv_dir(output_root, experiment)
        object_csv = exp_dir / f"{experiment}_summary.csv"
        _write_csv(object_csv, rows, CSV_COLUMNS)

        summary = lane_summary_rows.get(experiment, [])
        if summary:
            lane_csv = exp_dir / f"{experiment}_lane_summary.csv"
            _write_csv(lane_csv, summary, LANE_SUMMARY_COLUMNS)

    print("\nDone!")
    print("Check:")
    print(f"  - '{output_root}' for wb, boxes, polygons, MaxRGB, HSV-V images and CSVs")
