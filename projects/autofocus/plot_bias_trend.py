#!/usr/bin/env python3
"""Plot Bok 90Prime bias-level trends vs time across multiple nights.

Two data sources, four panels in one figure:
  1. Bias frames (bs.ZERO.*.fits)              → mean and std
  2. OBJECT science frames, BIASSEC overscan   → mean and std

Output: bias_trend.{png,csv} (csv contains both sources, marked by `source`).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

DEFAULT_FOLDERS = [
    "/Users/Jenny/projects/observation/bok/20250427",
    "/Users/Jenny/projects/observation/bok/20251023",
    "/Users/Jenny/projects/observation/bok/20251110_90prime",
    "/Users/Jenny/projects/observation/bok/20251111",
]

BIAS_PATTERNS = ("bs.ZERO.*.fits", "zero.*.fits", "ZERO.*.fits")
_SEC_RE = re.compile(r"\[\s*(\d+)\s*:\s*(\d+)\s*,\s*(\d+)\s*:\s*(\d+)\s*\]")
_AMP_NUM_RE = re.compile(r"(\d+)")


def _amp_number(extname: str) -> int | None:
    if not extname:
        return None
    m = _AMP_NUM_RE.findall(str(extname))
    return int(m[-1]) if m else None


# ---------- header helpers ----------

def parse_obs_time(header) -> datetime | None:
    date = header.get("DATE-OBS")
    if not date:
        return None
    if "T" in date:
        try:
            return datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    ut = header.get("UT") or header.get("TIME-OBS") or header.get("UTSTART")
    if ut is None:
        return None
    try:
        return datetime.fromisoformat(f"{date}T{ut}").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def is_object_frame(header) -> bool:
    return any("object" in str(header.get(k, "")).strip().lower()
               for k in ("IMAGETYP", "OBSTYPE"))


def parse_section(s: str) -> tuple[slice, slice] | None:
    """IRAF '[x1:x2,y1:y2]' (1-based, inclusive) → numpy (y, x) slices."""
    if not s:
        return None
    m = _SEC_RE.search(s)
    if not m:
        return None
    x1, x2, y1, y2 = map(int, m.groups())
    return (slice(y1 - 1, y2), slice(x1 - 1, x2))


# ---------- per-frame measurements ----------

def measure_full_bias(path: str, amp: int | None = None):
    """Sigma-clipped mean/std over bias frame data.

    If *amp* is given, restrict to the HDU whose EXTNAME ends in that number
    (e.g. ``IM4``); otherwise pool all 8 amps.
    """
    try:
        with fits.open(path, memmap=False) as hdul:
            samples = []
            for h in hdul:
                if h.data is None or h.data.ndim != 2:
                    continue
                if amp is not None and _amp_number(h.header.get("EXTNAME")) != amp:
                    continue
                arr = h.data.astype(np.float32)
                if arr.size > 1_000_000:
                    step = int(np.sqrt(arr.size / 1_000_000)) + 1
                    arr = arr[::step, ::step]
                samples.append(arr.ravel())
            if not samples:
                return None
            stacked = np.concatenate(samples)
            mean, _, std = sigma_clipped_stats(stacked, sigma=3.0, maxiters=3)
            return float(mean), float(std)
    except Exception as exc:
        print(f"  ! {os.path.basename(path)}: {exc}")
        return None


def measure_overscan(path: str, amp: int | None = None):
    """Sigma-clipped mean/std over BIASSEC region in an OBJECT frame.

    If *amp* is given, restrict to the HDU whose EXTNAME ends in that number.
    """
    try:
        with fits.open(path, memmap=False) as hdul:
            chunks = []
            for h in hdul:
                if h.data is None or h.data.ndim != 2:
                    continue
                if amp is not None and _amp_number(h.header.get("EXTNAME")) != amp:
                    continue
                sec = parse_section(h.header.get("BIASSEC", ""))
                if sec is None:
                    continue
                ny, nx = h.data.shape
                ys, xs = sec
                ys = slice(max(ys.start, 0), min(ys.stop, ny))
                xs = slice(max(xs.start, 0), min(xs.stop, nx))
                if ys.stop <= ys.start or xs.stop <= xs.start:
                    continue
                chunks.append(h.data[ys, xs].astype(np.float32).ravel())
            if not chunks:
                return None
            stacked = np.concatenate(chunks)
            mean, _, std = sigma_clipped_stats(stacked, sigma=3.0, maxiters=3)
            return float(mean), float(std)
    except Exception as exc:
        print(f"  ! {os.path.basename(path)}: {exc}")
        return None


# ---------- collection ----------

def collect(folders, amp: int | None = None):
    bias_rows: list[dict] = []
    over_rows: list[dict] = []

    for folder in folders:
        if not os.path.isdir(folder):
            print(f"skip (missing): {folder}")
            continue

        # --- bias frames ---
        bias_files: list[str] = []
        for pat in BIAS_PATTERNS:
            bias_files.extend(glob.glob(os.path.join(folder, pat)))
        bias_files = sorted(set(bias_files))

        for f in bias_files:
            try:
                hdr = fits.getheader(f)
            except Exception as exc:
                print(f"  ! header {os.path.basename(f)}: {exc}")
                continue
            t = parse_obs_time(hdr)
            if t is None:
                continue
            stats = measure_full_bias(f, amp=amp)
            if stats is None:
                continue
            mean, std = stats
            bias_rows.append({"folder": os.path.basename(folder),
                              "file": os.path.basename(f),
                              "time": t, "mean": mean, "std": std,
                              "camtemp": _to_float(hdr.get("CAMTEMP"))})

        # --- object frames (overscan) ---
        all_files = sorted(glob.glob(os.path.join(folder, "*.fits")))
        n_obj = 0
        for f in all_files:
            try:
                hdr = fits.getheader(f)
            except Exception:
                continue
            if not is_object_frame(hdr):
                continue
            n_obj += 1
            t = parse_obs_time(hdr)
            if t is None:
                continue
            stats = measure_overscan(f, amp=amp)
            if stats is None:
                continue
            mean, std = stats
            over_rows.append({"folder": os.path.basename(folder),
                              "file": os.path.basename(f),
                              "time": t, "mean": mean, "std": std,
                              "camtemp": _to_float(hdr.get("CAMTEMP"))})

        print(f"{folder}: {len(bias_files)} bias  /  {n_obj} object frames")

    bias_rows.sort(key=lambda r: r["time"])
    over_rows.sort(key=lambda r: r["time"])
    return bias_rows, over_rows


# ---------- plotting ----------

def _scatter(ax, rows, key, colors):
    folders = sorted({r["folder"] for r in rows})
    for folder in folders:
        sub = [r for r in rows if r["folder"] == folder]
        ax.plot([r["time"] for r in sub], [r[key] for r in sub],
                "o", ms=4, color=colors[folder], label=folder, alpha=0.8)


def plot(bias_rows, over_rows, outpath: str) -> None:
    if not bias_rows and not over_rows:
        print("No data to plot.")
        return

    all_folders = sorted({r["folder"] for r in (*bias_rows, *over_rows)})
    cmap = plt.get_cmap("tab10")
    colors = {f: cmap(i % 10) for i, f in enumerate(all_folders)}

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), constrained_layout=True)
    ax_bm, ax_bs, ax_om, ax_os = axes

    _scatter(ax_bm, bias_rows, "mean", colors)
    _scatter(ax_bs, bias_rows, "std",  colors)
    _scatter(ax_om, over_rows, "mean", colors)
    _scatter(ax_os, over_rows, "std",  colors)

    panels = [
        (ax_bm, "Bias frame — mean (ADU)"),
        (ax_bs, "Bias frame — std (ADU)"),
        (ax_om, "Overscan (BIASSEC, OBJECT) — mean (ADU)"),
        (ax_os, "Overscan (BIASSEC, OBJECT) — std (ADU)"),
    ]
    for ax, ylabel in panels:
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=timezone.utc))
        if ax.get_lines():
            ax.legend(loc="best", fontsize=9)

    axes[-1].set_xlabel("UTC time")
    fig.suptitle("Bok 90Prime — bias level trends vs time", fontsize=14)
    fig.savefig(outpath, dpi=130)
    print(f"saved plot → {outpath}")


def write_csv(bias_rows, over_rows, outpath: str) -> None:
    with open(outpath, "w") as f:
        f.write("source,folder,file,time_utc,mean,std\n")
        for src, rows in (("bias", bias_rows), ("overscan", over_rows)):
            for r in rows:
                f.write(f"{src},{r['folder']},{r['file']},"
                        f"{r['time'].isoformat()},{r['mean']:.4f},{r['std']:.4f}\n")
    print(f"saved csv  → {outpath}")


def plot_bias_per_folder(bias_rows, over_rows, amp: int | None, out_dir: str) -> None:
    """One figure per folder: bias mean/std + overscan mean/std vs time."""
    if not bias_rows and not over_rows:
        print("No data to plot.")
        return
    os.makedirs(out_dir, exist_ok=True)
    folders = sorted({r["folder"] for r in (*bias_rows, *over_rows)})
    amp_tag = f"IM{amp}" if amp is not None else "all-amps"
    for folder in folders:
        b = [r for r in bias_rows if r["folder"] == folder]
        o = [r for r in over_rows if r["folder"] == folder]
        if not b and not o:
            continue

        def _format(ax, ylabel):
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%H:%M", tz=timezone.utc))
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)

        # --- bias figure ---
        if b:
            fig, (ax_m, ax_s) = plt.subplots(2, 1, figsize=(11, 6),
                                              sharex=True, constrained_layout=True)
            tb = [r["time"] for r in b]
            ax_m.plot(tb, [r["mean"] for r in b], "o-", ms=4, color="C0")
            ax_s.plot(tb, [r["std"]  for r in b], "o-", ms=4, color="C1")
            _format(ax_m, "Bias mean (ADU)")
            _format(ax_s, "Bias std (ADU)")
            ax_s.set_xlabel("UTC time")
            fig.suptitle(f"Bok 90Prime bias trend — {folder} — {amp_tag}",
                         fontsize=13)
            outpath = os.path.join(out_dir, f"bias_trend_{folder}_{amp_tag}.png")
            fig.savefig(outpath, dpi=130)
            plt.close(fig)
            print(f"saved plot → {outpath}  ({len(b)} bias frames)")

        # --- overscan figure ---
        if o:
            fig, (ax_m, ax_s) = plt.subplots(2, 1, figsize=(11, 6),
                                              sharex=True, constrained_layout=True)
            to = [r["time"] for r in o]
            ax_m.plot(to, [r["mean"] for r in o], "o-", ms=4, color="C2")
            ax_s.plot(to, [r["std"]  for r in o], "o-", ms=4, color="C3")
            _format(ax_m, "Overscan mean (ADU)")
            _format(ax_s, "Overscan std (ADU)")
            ax_s.set_xlabel("UTC time")
            fig.suptitle(f"Bok 90Prime overscan trend — {folder} — {amp_tag}",
                         fontsize=13)
            outpath = os.path.join(out_dir, f"overscan_trend_{folder}_{amp_tag}.png")
            fig.savefig(outpath, dpi=130)
            plt.close(fig)
            print(f"saved plot → {outpath}  ({len(o)} object frames)")


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def plot_vs_temperature(bias_rows, over_rows, amp: int | None, out_dir: str) -> None:
    """Two figures: bias and overscan mean/std vs CAMTEMP, all folders combined."""
    os.makedirs(out_dir, exist_ok=True)
    amp_tag = f"IM{amp}" if amp is not None else "all-amps"
    all_folders = sorted({r["folder"] for r in (*bias_rows, *over_rows)})
    cmap = plt.get_cmap("tab10")
    colors = {f: cmap(i % 10) for i, f in enumerate(all_folders)}

    for label, rows, fname_prefix in (
        ("Bias", bias_rows, "bias_vs_camtemp"),
        ("Overscan", over_rows, "overscan_vs_camtemp"),
    ):
        rows_t = [r for r in rows if r.get("camtemp") is not None]
        if not rows_t:
            print(f"No {label.lower()} CAMTEMP data — skipping.")
            continue
        fig, (ax_m, ax_s) = plt.subplots(2, 1, figsize=(11, 7),
                                          sharex=True, constrained_layout=True)
        for folder in all_folders:
            sub = [r for r in rows_t if r["folder"] == folder]
            if not sub:
                continue
            ts = [r["camtemp"] for r in sub]
            ax_m.plot(ts, [r["mean"] for r in sub], "o", ms=4,
                      color=colors[folder], label=folder, alpha=0.8)
            ax_s.plot(ts, [r["std"]  for r in sub], "o", ms=4,
                      color=colors[folder], label=folder, alpha=0.8)
        for ax, ylabel in ((ax_m, f"{label} mean (ADU)"),
                            (ax_s, f"{label} std (ADU)")):
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax_m.legend(loc="best", fontsize=9)
        ax_s.set_xlabel("CAMTEMP")
        fig.suptitle(f"Bok 90Prime {label.lower()} vs CAMTEMP — {amp_tag}",
                     fontsize=13)
        outpath = os.path.join(out_dir, f"{fname_prefix}_{amp_tag}.png")
        fig.savefig(outpath, dpi=130)
        plt.close(fig)
        print(f"saved plot → {outpath}  ({len(rows_t)} points)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folders", nargs="+", default=DEFAULT_FOLDERS)
    p.add_argument("--out-plot", default="bias_trend.png")
    p.add_argument("--out-csv",  default="bias_trend.csv")
    p.add_argument("--amp", type=int, default=None,
                   help="Restrict to a single amp (EXTNAME suffix), e.g. 4 for IM4.")
    p.add_argument("--per-folder", action="store_true",
                   help="Generate one bias-only plot per folder (no overscan).")
    p.add_argument("--vs-temp", action="store_true",
                   help="Also generate bias/overscan mean+std vs CAMTEMP plots "
                        "(combined across folders).")
    p.add_argument("--out-dir", default=".",
                   help="Output directory for per-folder plots.")
    args = p.parse_args()

    bias_rows, over_rows = collect(args.folders, amp=args.amp)
    print(f"\nUsable bias frames:    {len(bias_rows)}")
    print(f"Usable OBJECT frames:  {len(over_rows)}")
    write_csv(bias_rows, over_rows, args.out_csv)
    if args.per_folder:
        plot_bias_per_folder(bias_rows, over_rows, args.amp, args.out_dir)
    else:
        plot(bias_rows, over_rows, args.out_plot)
    if args.vs_temp:
        plot_vs_temperature(bias_rows, over_rows, args.amp, args.out_dir)


if __name__ == "__main__":
    main()
