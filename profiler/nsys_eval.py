#!/usr/bin/env python3
"""
Generate Nsight Systems launch-gap plots for CoBaLI
===================================================

This script expects two SQLite files produced by:

  nsys export -t sqlite -o seq_profile  seq_profile.nsys-rep
  nsys export -t sqlite -o cont_profile cont_profile.nsys-rep

It:
  - Reads CUDA runtime launch events
  - Computes launch gaps (next_start - end)
  - Produces:
      1) Histogram of gaps (seq)
      2) Histogram of gaps (cont)
      3) Bar chart of mean / P90 / P95 gap
      4) Timeline of gaps (seq, sampled)
      5) Timeline of gaps (cont, sampled)
"""

import os
import sqlite3
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------
# Data loading utilities
# -----------------------

def load_launches(sqlite_path: Path) -> pd.DataFrame:
    """
    Load CUDA runtime launch events from an Nsight Systems SQLite export.

    We use:
      - StringIds: maps id -> API name
      - CUPTI_ACTIVITY_KIND_RUNTIME: runtime API events with start/end + nameId

    We then filter rows whose API name contains "Launch".
    """
    conn = sqlite3.connect(sqlite_path)

    # Map nameId -> API name
    names = pd.read_sql("SELECT id, value FROM StringIds", conn)

    # Runtime API events
    rt = pd.read_sql(
        """
        SELECT start, end, (end - start) AS dur, nameId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        ORDER BY start
        """,
        conn,
    )

    conn.close()

    # Attach the function names
    rt = rt.merge(names, left_on="nameId", right_on="id", how="left")
    rt.rename(columns={"value": "api_name"}, inplace=True)

    # Keep only launch calls
    launches = rt[rt["api_name"].str.contains("Launch", case=False, na=False)].copy()

    # Compute gap to next launch
    launches["next_start"] = launches["start"].shift(-1)
    launches["gap"] = launches["next_start"] - launches["end"]

    return launches


def build_stats(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute basic statistics on launch gaps and API durations.

    gap, dur are in nanoseconds.
    """
    gap = df["gap"].dropna()
    dur = df["dur"].dropna()

    stats = {
        "count": len(df),
        "gap_mean_ns": gap.mean(),
        "gap_median_ns": gap.median(),
        "gap_p75_ns": gap.quantile(0.75),
        "gap_p90_ns": gap.quantile(0.90),
        "gap_p95_ns": gap.quantile(0.95),
        "gap_max_ns": gap.max(),
        "gap_min_ns": gap.min(),
        "dur_mean_ns": dur.mean(),
        "dur_median_ns": dur.median(),
    }
    return stats


def launch_rate(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Compute launches per second and the total span (seconds).
    Uses min(start) to max(end) as the window.
    """
    span_ns = df["end"].max() - df["start"].min()
    span_s = span_ns / 1e9
    rate = len(df) / span_s if span_s > 0 else float("nan")
    return rate, span_s


# -----------------------
# Plotting helpers
# -----------------------

def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sample_series(series: pd.Series, max_points: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Down-sample a series to at most max_points points for plotting timelines.
    """
    series = series.reset_index(drop=True)
    n = len(series)
    if n == 0:
        return np.array([]), np.array([])
    if n <= max_points:
        return np.arange(n), series.to_numpy()

    idx = np.linspace(0, n - 1, max_points).astype(int)
    return idx, series.iloc[idx].to_numpy()


def plot_histogram(
    gaps_us: np.ndarray,
    title: str,
    out_path: Path,
    bins: int = 100,
    max_us: float = 200.0,
) -> None:
    """
    Plot a histogram of gaps (µs), capped at max_us for visualization.
    """
    if len(gaps_us) == 0:
        print(f"[warn] No data for {title}, skipping histogram.")
        return

    clipped = np.clip(gaps_us, None, max_us)

    plt.figure(figsize=(6, 4))
    plt.hist(clipped, bins=bins)
    plt.title(f"{title} (gaps capped at {max_us:.0f} µs)")
    plt.xlabel("Gap (µs)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[info] Saved histogram: {out_path}")


def plot_bar_stats(
    seq_stats: Dict[str, float],
    cont_stats: Dict[str, float],
    out_path: Path,
) -> None:
    """
    Bar chart of mean, P90, P95 gap in µs for seq vs cont.
    """
    labels = ["Mean", "P90", "P95"]
    seq_vals = [
        seq_stats["gap_mean_ns"] / 1e3,
        seq_stats["gap_p90_ns"] / 1e3,
        seq_stats["gap_p95_ns"] / 1e3,
    ]
    cont_vals = [
        cont_stats["gap_mean_ns"] / 1e3,
        cont_stats["gap_p90_ns"] / 1e3,
        cont_stats["gap_p95_ns"] / 1e3,
    ]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6, 4))
    # plt.bar(x - width / 2, seq_vals, width, label="Sequential")
    plt.bar(x - width / 2, seq_vals, width, label="Continuous without Prefill")
    # plt.bar(x + width / 2, cont_vals, width, label="Continuous")
    plt.bar(x + width / 2, cont_vals, width, label="Continuous with Prefill")

    plt.xticks(x, labels)
    plt.ylabel("Gap (µs)")
    plt.title("Mean / P90 / P95 Launch Gaps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[info] Saved bar chart: {out_path}")


def plot_timeline(
    gaps_us: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """
    Timeline plot of gaps (µs) vs sampled launch index.
    """
    if len(gaps_us) == 0:
        print(f"[warn] No data for {title}, skipping timeline.")
        return

    x = np.arange(len(gaps_us))

    plt.figure(figsize=(6, 4))
    plt.plot(x, gaps_us)
    plt.title(title)
    plt.xlabel("Launch index (sampled)")
    plt.ylabel("Gap (µs)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[info] Saved timeline: {out_path}")


# -----------------------
# Main
# -----------------------

def main() -> None:
    # Adjust these paths if needed
    seq_sqlite = Path("seq_profile.sqlite")
    cont_sqlite = Path("cont_prefill_nosplit.sqlite")
    # cont_sqlite = Path("cont_profile.sqlite")
    # cont_sqlite = Path("cont_prefill_split.sqlite")
    outdir = Path("plots/seq_vs_cont")
    # outdir = Path("plots/cont_noprefill_vs_cont_prefill")

    ensure_outdir(outdir)

    # Load data
    print("[info] Loading launches from SQLite exports...")
    seq_launches = load_launches(seq_sqlite)
    cont_launches = load_launches(cont_sqlite)

    # Compute stats
    seq_stats = build_stats(seq_launches)
    cont_stats = build_stats(cont_launches)

    seq_rate, seq_span = launch_rate(seq_launches)
    cont_rate, cont_span = launch_rate(cont_launches)

    print("[info] Sequential stats:")
    for k, v in seq_stats.items():
        print(f"  {k}: {v:.3f}")
    print(f"  launch_rate: {seq_rate:.3f} launches/s, span: {seq_span:.3f} s")

    print("[info] Continuous stats:")
    for k, v in cont_stats.items():
        print(f"  {k}: {v:.3f}")
    print(f"  launch_rate: {cont_rate:.3f} launches/s, span: {cont_span:.3f} s")

    # Convert gaps to µs for plotting
    seq_gaps_us = (seq_launches["gap"].dropna() / 1e3).to_numpy()
    cont_gaps_us = (cont_launches["gap"].dropna() / 1e3).to_numpy()

    # Figures 1 & 2: histograms (caps at 200 µs)
    plot_histogram(
        gaps_us=seq_gaps_us,
        title="Sequential Mode: Launch Gaps",
        # title="Continuous Mode without Prefill: Launch Gaps",
        out_path=outdir / "seq_gaps_hist.png",
        # out_path=outdir / "cont_no_prefill_gaps_hist.png",
        bins=100,
        max_us=200.0,
    )

    plot_histogram(
        gaps_us=cont_gaps_us,
        title="Continuous Mode: Launch Gaps",
        # title="Continuous Mode with Prefill: Launch Gaps",
        out_path=outdir / "cont_gaps_hist.png",
        # out_path=outdir / "cont_with_prefill_gaps_hist.png",
        bins=100,
        max_us=200.0,
    )

    # Figure 3: bar chart of mean / P90 / P95
    plot_bar_stats(seq_stats, cont_stats, outdir / "gap_stats_bar.png")

    # Figures 4 & 5: timelines (sampled)
    seq_idx, seq_sample = sample_series(seq_launches["gap"].dropna() / 1e3, max_points=2000)
    cont_idx, cont_sample = sample_series(cont_launches["gap"].dropna() / 1e3, max_points=2000)

    plot_timeline(
        gaps_us=seq_sample,
        title="Sequential Mode: Launch Gap Timeline (sampled)",
        # title="Continuous without Prefill mode: Launch Gap Timeline (sampled)",
        out_path=outdir / "seq_gap_timeline.png",
        # out_path=outdir / "cont_no_prefill_gap_timeline.png",
    )

    plot_timeline(
        gaps_us=cont_sample,
        title="Continuous Mode: Launch Gap Timeline (sampled)",
        # title="Continuous with Prefill Mode: Launch Gap Timeline (sampled)",
        out_path=outdir / "cont_gap_timeline.png",
        # out_path=outdir / "cont_with_prefill_gap_timeline.png",
    )

    print(f"[info] All plots saved under: {outdir.resolve()}")


if __name__ == "__main__":
    main()
