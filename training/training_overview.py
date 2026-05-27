"""
Create a compact overview table from YOLO training runs.

The overview scans for results.csv files below a training output directory and
extracts the best segmentation-mask mAP run statistics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


MASK_MAP_COL = "metrics/mAP50-95(M)"
MASK_MAP50_COL = "metrics/mAP50(M)"


def create_training_overview(results_root: str | Path) -> Optional[pd.DataFrame]:
    """
    Scan YOLO training results and write model_training_overview.csv.

    Parameters
    ----------
    results_root:
        Folder containing YOLO training run folders.

    Returns
    -------
    pandas.DataFrame or None
        Sorted overview dataframe, or None if no valid results were found.
    """
    results_root = Path(results_root)
    summary_rows = []

    for results_csv in sorted(results_root.rglob("results.csv")):
        run_dir = results_csv.parent
        run_name = run_dir.name

        try:
            df = pd.read_csv(results_csv)
        except Exception as exc:
            print(f"Could not read {results_csv}: {exc}")
            continue

        df.columns = [column.strip() for column in df.columns]

        if MASK_MAP_COL not in df.columns:
            print(f"Skipping {run_name}: no mask mAP column found.")
            continue

        best_idx = df[MASK_MAP_COL].idxmax()
        best_row = df.loc[best_idx]

        summary_rows.append({
            "run": run_name,
            "best_epoch": int(best_row.get("epoch", best_idx)),
            "best_mAP50_95_mask": best_row.get(MASK_MAP_COL, None),
            "best_mAP50_mask": best_row.get(MASK_MAP50_COL, None),
            "precision_mask": best_row.get("metrics/precision(M)", None),
            "recall_mask": best_row.get("metrics/recall(M)", None),
            "box_mAP50_95": best_row.get("metrics/mAP50-95(B)", None),
            "box_mAP50": best_row.get("metrics/mAP50(B)", None),
            "train_seg_loss": best_row.get("train/seg_loss", None),
            "val_seg_loss": best_row.get("val/seg_loss", None),
            "results_csv": str(results_csv),
            "run_dir": str(run_dir),
        })

    if not summary_rows:
        print("No valid training results found.")
        return None

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by="best_mAP50_95_mask",
        ascending=False,
    )

    out_file = results_root / "model_training_overview.csv"
    summary_df.to_csv(out_file, index=False)

    print("\nModel training overview:")
    print(summary_df)
    print(f"\nSaved overview to: {out_file}")

    return summary_df
