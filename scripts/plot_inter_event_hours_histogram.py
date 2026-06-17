#!/usr/bin/env python3
"""
Histogram of time between consecutive events on MEDS training data.

Reads parquet files from ``{data_dir}/{split}/`` (same layout as MEDSDataset),
sorts by ``subject_id`` and ``time``, takes within-subject consecutive diffs,
and plots gaps in **hours** or **minutes** (``--unit``).

Two panels:
  left  — linear x and y
  right — log₁₀ x-axis (positive gaps only)

Examples
--------
  python scripts/plot_inter_event_hours_histogram.py --config configs/ehr_config.yaml
  python scripts/plot_inter_event_hours_histogram.py --config configs/ehr_config.yaml --unit minutes
  python scripts/plot_inter_event_hours_histogram.py --data_dir /path/to/clean_meds \\
      --split train --max-files 20 --unit hours --out /tmp/inter_event.png
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from data.meds_parser import _sorted_parquet_files


def _load_config_data_dir(config_path: str) -> str:
    from main import load_config

    cfg = load_config(config_path)
    d = cfg.get("data", {}).get("data_dir")
    if not d:
        raise SystemExit(f"No data.data_dir in {config_path}")
    return str(d)


def collect_inter_event_seconds(
    data_dir: str,
    split: str,
    max_files: int | None,
) -> np.ndarray:
    import polars as pl

    files = _sorted_parquet_files(data_dir, split)
    if max_files is not None:
        files = files[: max(1, max_files)]

    lf = pl.scan_parquet(files).select(
        [
            pl.col("subject_id"),
            pl.col("time"),
        ]
    )
    df = lf.collect()
    # Parquet may store ``time`` as string or datetime depending on export.
    if df["time"].dtype in (pl.Utf8, pl.String):
        df = df.with_columns(pl.col("time").str.to_datetime(strict=False))
    elif df["time"].dtype != pl.Datetime:
        df = df.with_columns(
            pl.col("time").cast(pl.Datetime(time_unit="us"), strict=False)
        )
    df = df.filter(pl.col("time").is_not_null())
    df = df.sort(["subject_id", "time"])

    delta = pl.col("time").diff().over("subject_id").alias("delta")
    out = df.with_columns(delta).filter(pl.col("delta").is_not_null())

    # Polars Duration → seconds
    secs = out.select(pl.col("delta").dt.total_seconds()).to_series().to_numpy()
    secs = secs.astype(np.float64)
    secs = secs[np.isfinite(secs)]
    return secs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inter-event time histogram (hours or minutes)",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="YAML config; uses data.data_dir if --data_dir not set",
    )
    ap.add_argument("--data_dir", default=None, help="MEDS root (contains train/ etc.)")
    ap.add_argument("--split", default="train", help="Subfolder name, e.g. train")
    ap.add_argument(
        "--unit",
        choices=("hours", "minutes"),
        default="hours",
        help="Time unit for the x-axis (default: hours)",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Read only the first N parquet files (ordered by name) for a quick plot",
    )
    ap.add_argument(
        "--xmax-linear",
        type=float,
        default=None,
        help="Max value on linear panel x-axis in the chosen --unit (default: 99th percentile)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: logs/inter_event_<unit>_histogram.png)",
    )
    args = ap.parse_args()

    if args.out is None:
        args.out = os.path.join(
            ROOT, "logs", f"inter_event_{args.unit}_histogram.png"
        )

    data_dir = args.data_dir
    if data_dir is None:
        if args.config is None:
            ap.error("Provide --data_dir or --config")
        data_dir = _load_config_data_dir(args.config)

    print(f"[data] data_dir={data_dir}  split={args.split}  unit={args.unit}")
    secs = collect_inter_event_seconds(data_dir, args.split, args.max_files)
    scale = 3600.0 if args.unit == "hours" else 60.0
    x = secs / scale
    unit_short = "h" if args.unit == "hours" else "min"
    unit_long = "hours" if args.unit == "hours" else "minutes"

    print(f"[data] n gaps (finite): {len(x):,}")
    if len(x) == 0:
        raise SystemExit("No inter-event gaps computed — check time column and split.")

    import matplotlib.pyplot as plt

    xmax_lin = args.xmax_linear
    if xmax_lin is None:
        xmax_lin = float(np.percentile(x, 99))
    xmax_lin = max(xmax_lin, 1e-9)

    pos = x[x > 0]
    if len(pos) == 0:
        pos = x.copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Linear x, linear y
    ax = axes[0]
    ax.hist(
        np.clip(x, 0, xmax_lin),
        bins=80,
        range=(0, xmax_lin),
        color="steelblue",
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_xlabel(f"{unit_long.capitalize()} between consecutive events (same subject)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Linear scale (x clipped to [0, {xmax_lin:.4g}] {unit_short}, 99p cap)"
    )
    ax.grid(True, alpha=0.3)

    # Log₁₀ x (positive gaps only)
    ax = axes[1]
    lo = float(np.nanpercentile(pos, 0.1))
    hi = float(np.nanpercentile(pos, 99.9))
    lo = max(lo, 1e-12)
    hi = max(hi, lo * 10)
    bins = np.logspace(np.log10(lo), np.log10(hi), 80)
    ax.hist(pos, bins=bins, color="darkorange", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel(f"{unit_long.capitalize()} between events (log10 x)")
    ax.set_ylabel("Count")
    ax.set_title("Log-scaled x (positive gaps; wide dynamic range)")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Inter-event times ({unit_long}) — {args.split} split — n={len(x):,} gaps",
        fontsize=11,
    )
    fig.tight_layout()

    out_abs = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_abs, dpi=150)
    print(f"[plot] Wrote {out_abs}")


if __name__ == "__main__":
    main()

