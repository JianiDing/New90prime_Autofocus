#!/usr/bin/env python3
"""Plot Bok 90Prime bias-level trends vs time across multiple nights, per amplifier.

For each of the 8 amps (IM1..IM8) writes one PNG with 4 panels:
  1. Bias frame (bs.ZERO.*.fits) mean
  2. Bias frame std
  3. Overscan (BIASSEC, OBJECT) mean
  4. Overscan (BIASSEC, OBJECT) std

Outputs:
  bias_trend_<AMP>.png       (× 8)
  bias_trend_temp_<AMP>.png  (× 8, if a temperature keyword is available)
  bias_trend.csv             (long-format: source, amp, folder, file, time, mean, std, temp)
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

DEFAULT_FOLDERS: list[str] = []

BIAS_PATTERNS = ("bs.ZERO.*.fits", "img.ZERO.*.fits", "zero.*.fits", "ZERO.*.fits", "*.ZERO.*.fits")
_SEC_RE = re.compile(r"\[\s*(\d+)\s*:\s*(\d+)\s*,\s*(\d+)\s*:\s*(\d+)\s*\]")
_AMP_NUM_RE = re.compile(r"(\d+)")
_MASK_CACHE: dict[tuple[str, int], np.ndarray | None] = {}


# ---------- bad pixel mask ----------

def amp_number_from_name(name: str) -> int | None:
    """Extract trailing integer from amp name, e.g. 'IM3' -> 3, 'R5' -> 5."""
    if not name:
        return None
    m = _AMP_NUM_RE.findall(str(name))
    return int(m[-1]) if m else None


def load_mask_for_amp(mask_dir: str | None, amp_num: int) -> np.ndarray | None:
    """Load bad-pixel mask for a given amp number.

    Looks for files named ``bad_pixel_mask_amp_<BAND><N>.npy`` in *mask_dir*
    (any band), and returns the logical-OR across all bands found (bad columns
    are detector-level, so band-independent for bias diagnostics).
    Returns boolean array where True == bad pixel, or None if no mask found.
    Cached per (mask_dir, amp_num).
    """
    if not mask_dir:
        return None
    key = (mask_dir, amp_num)
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]
    pattern = os.path.join(mask_dir, f"bad_pixel_mask_amp_*{amp_num}.npy")
    files = glob.glob(pattern)
    # filter to filenames that end with the digit immediately (avoid amp 1 catching 11)
    files = [f for f in files
             if re.search(rf"bad_pixel_mask_amp_[A-Za-z]+{amp_num}\.npy$", os.path.basename(f))]
    if not files:
        _MASK_CACHE[key] = None
        return None
    mask = None
    for fp in files:
        try:
            arr = np.load(fp).astype(bool)
        except Exception as exc:
            print(f"  ! failed to load mask {fp}: {exc}")
            continue
        mask = arr if mask is None else (mask | arr)
    _MASK_CACHE[key] = mask
    return mask


# ---------- header helpers ----------

def parse_obs_time(header):
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


def parse_temperature(header, preferred_key="CAMTEMP"):
    """Return (temperature, key) from the primary header, or (None, None)."""
    keys = [preferred_key, "CAMTEMP", "CCDTEMP1", "CCDTEMP", "DEWTEMP", "DOME_DEW"]
    seen = set()
    for key in keys:
        if not key or key in seen:
            continue
        seen.add(key)
        if key not in header:
            continue
        try:
            return float(header[key]), key
        except (TypeError, ValueError):
            continue
    return None, None


def is_object_frame(header):
    return any("object" in str(header.get(k, "")).strip().lower()
               for k in ("IMAGETYP", "OBSTYPE"))


def parse_section(s):
    if not s:
        return None
    m = _SEC_RE.search(s)
    if not m:
        return None
    x1, x2, y1, y2 = map(int, m.groups())
    return (slice(y1 - 1, y2), slice(x1 - 1, x2))


def amp_name(hdu, idx):
    return (hdu.header.get("EXTNAME")
            or hdu.header.get("AMPNAME")
            or hdu.name
            or f"IM{idx}")


# ---------- per-amp measurements ----------

def measure_per_amp_full(path, mask_dir=None):
    """Return {amp: (mean, std)} over the entire data array of each amp.

    If *mask_dir* is given, masks bad pixels (True in mask) before stats.
    """
    out = {}
    try:
        with fits.open(path, memmap=False) as hdul:
            for i, h in enumerate(hdul):
                if h.data is None or h.data.ndim != 2:
                    continue
                aname = amp_name(h, i)
                arr = h.data.astype(np.float32)
                # apply bad-pixel mask if available (must match full-frame shape)
                anum = amp_number_from_name(aname)
                bad = load_mask_for_amp(mask_dir, anum) if anum is not None else None
                masked = bad is not None and bad.shape == arr.shape
                if masked:
                    arr = np.where(bad, np.nan, arr)
                # subsample large frames for speed
                if arr.size > 1_000_000:
                    step = int(np.sqrt(arr.size / 1_000_000)) + 1
                    arr = arr[::step, ::step]
                flat = arr.ravel()
                flat = flat[np.isfinite(flat)]
                if flat.size == 0:
                    continue
                if masked:
                    # mask already removes the outlier columns -> raw stats
                    m = float(np.median(flat))
                    s = float(np.std(flat))
                else:
                    m, _, s = sigma_clipped_stats(flat, sigma=5.0, maxiters=3)
                    m, s = float(m), float(s)
                out[aname] = (m, s)
    except Exception as exc:
        print(f"  ! {os.path.basename(path)}: {exc}")
    return out


def measure_per_amp_overscan(path, mask_dir=None):
    """Return {amp: (mean, std)} over the BIASSEC region of each amp.

    The BIASSEC region is the serial-overscan strip and has no science pixels,
    so the bad-pixel mask is generally not applied; *mask_dir* is accepted for
    a uniform call signature and used only if the mask happens to cover the
    overscan columns too.
    """
    out = {}
    try:
        with fits.open(path, memmap=False) as hdul:
            for i, h in enumerate(hdul):
                if h.data is None or h.data.ndim != 2:
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
                aname = amp_name(h, i)
                region = h.data[ys, xs].astype(np.float32)
                anum = amp_number_from_name(aname)
                bad = load_mask_for_amp(mask_dir, anum) if anum is not None else None
                masked = bad is not None and bad.shape == h.data.shape
                if masked:
                    region = np.where(bad[ys, xs], np.nan, region)
                flat = region.ravel()
                flat = flat[np.isfinite(flat)]
                if flat.size == 0:
                    continue
                if masked:
                    m = float(np.median(flat))
                    s = float(np.std(flat))
                else:
                    m, _, s = sigma_clipped_stats(flat, sigma=5.0, maxiters=3)
                    m, s = float(m), float(s)
                out[aname] = (m, s)
    except Exception as exc:
        print(f"  ! {os.path.basename(path)}: {exc}")
    return out


# ---------- collection ----------

def collect(folders, mask_dir=None, temp_key="CAMTEMP"):
    """Return (bias_rows, over_rows). Each row has amp, folder, file, time, mean, std."""
    bias_rows: list[dict] = []
    over_rows: list[dict] = []

    for folder in folders:
        if not os.path.isdir(folder):
            print(f"skip (missing): {folder}")
            continue
        fname = os.path.basename(folder)

        # bias files
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
            temp, used_temp_key = parse_temperature(hdr, temp_key)
            for amp, (mean, std) in measure_per_amp_full(f, mask_dir).items():
                bias_rows.append({"amp": amp, "folder": fname,
                                  "file": os.path.basename(f),
                                  "time": t, "mean": mean, "std": std,
                                  "temp": temp, "temp_key": used_temp_key})

        # object frames
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
            temp, used_temp_key = parse_temperature(hdr, temp_key)
            for amp, (mean, std) in measure_per_amp_overscan(f, mask_dir).items():
                over_rows.append({"amp": amp, "folder": fname,
                                  "file": os.path.basename(f),
                                  "time": t, "mean": mean, "std": std,
                                  "temp": temp, "temp_key": used_temp_key})

        print(f"{folder}: {len(bias_files)} bias  /  {n_obj} object frames")

    bias_rows.sort(key=lambda r: r["time"])
    over_rows.sort(key=lambda r: r["time"])
    return bias_rows, over_rows


# ---------- plotting ----------

def _scatter(ax, rows, key, colors):
    folders = sorted({r["folder"] for r in rows})
    for folder in folders:
        sub = [r for r in rows if r["folder"] == folder]
        if not sub:
            continue
        ax.plot([r["time"] for r in sub], [r[key] for r in sub],
                "o", ms=4, color=colors[folder], label=folder, alpha=0.8)


def plot_per_amp(amp, bias_rows, over_rows, colors, outpath):
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
    fig.suptitle(f"Bok 90Prime — bias level trends vs time  —  amp {amp}",
                 fontsize=14)
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"saved → {outpath}")


def _scatter_temp(ax, rows, key, colors):
    folders = sorted({r["folder"] for r in rows})
    for folder in folders:
        sub = [r for r in rows if r["folder"] == folder and r.get("temp") is not None]
        if not sub:
            continue
        ax.plot([r["temp"] for r in sub], [r[key] for r in sub],
                "o", ms=4, color=colors[folder], label=folder, alpha=0.8)


def plot_per_amp_temperature(amp, bias_rows, over_rows, colors, outpath):
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), constrained_layout=True)
    ax_bm, ax_bs, ax_om, ax_os = axes

    _scatter_temp(ax_bm, bias_rows, "mean", colors)
    _scatter_temp(ax_bs, bias_rows, "std",  colors)
    _scatter_temp(ax_om, over_rows, "mean", colors)
    _scatter_temp(ax_os, over_rows, "std",  colors)

    temp_keys = sorted({r.get("temp_key") for r in (*bias_rows, *over_rows) if r.get("temp_key")})
    xlabel = "Temperature (C)"
    if temp_keys:
        xlabel = f"Temperature ({'/'.join(temp_keys)}, C)"

    panels = [
        (ax_bm, "Bias frame — mean (ADU)"),
        (ax_bs, "Bias frame — std (ADU)"),
        (ax_om, "Overscan (BIASSEC, OBJECT) — mean (ADU)"),
        (ax_os, "Overscan (BIASSEC, OBJECT) — std (ADU)"),
    ]
    for ax, ylabel in panels:
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if ax.get_lines():
            ax.legend(loc="best", fontsize=9)

    axes[-1].set_xlabel(xlabel)
    fig.suptitle(f"Bok 90Prime — bias level trends vs temperature  —  amp {amp}",
                 fontsize=14)
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"saved → {outpath}")


def write_csv(bias_rows, over_rows, outpath):
    with open(outpath, "w") as f:
        f.write("source,amp,folder,file,time_utc,mean,std,temp_c,temp_key\n")
        for src, rows in (("bias", bias_rows), ("overscan", over_rows)):
            for r in rows:
                temp = "" if r.get("temp") is None else f"{r['temp']:.4f}"
                temp_key = r.get("temp_key") or ""
                f.write(f"{src},{r['amp']},{r['folder']},{r['file']},"
                        f"{r['time'].isoformat()},{r['mean']:.4f},{r['std']:.4f},"
                        f"{temp},{temp_key}\n")
    print(f"saved csv → {outpath}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folders", nargs="+", required=True,
                   help="One or more directories containing FITS files (bias and/or object).")
    p.add_argument("--out-prefix", default="bias_trend",
                   help="Prefix for per-amp PNG files (e.g. 'bias_trend' → bias_trend_IM1.png).")
    p.add_argument("--out-csv", default="bias_trend.csv")
    p.add_argument("--mask-dir", default=None,
                   help="Directory of bad-pixel masks (bad_pixel_mask_amp_<BAND><N>.npy). "
                        "If given, bad pixels are excluded from the bias stats.")
    p.add_argument("--temp-key", default="CAMTEMP",
                   help="Preferred primary-header temperature keyword for temperature plots.")
    p.add_argument("--temperature-only", action="store_true",
                   help="Only write the CSV and temperature plots; do not write time-trend PNGs.")
    args = p.parse_args()

    bias_rows, over_rows = collect(args.folders, mask_dir=args.mask_dir, temp_key=args.temp_key)
    print(f"\nUsable bias rows:    {len(bias_rows)}")
    print(f"Usable overscan rows: {len(over_rows)}")

    # If a mask is used and the user kept the default names, suffix outputs
    # with "_masked" so we don't overwrite the sigma-clipped versions.
    suffix = "_masked" if args.mask_dir else ""
    out_prefix = args.out_prefix
    out_csv = args.out_csv
    if suffix:
        if out_prefix == "bias_trend":
            out_prefix = "bias_trend_masked"
        if out_csv == "bias_trend.csv":
            out_csv = "bias_trend_masked.csv"

    write_csv(bias_rows, over_rows, out_csv)

    if not bias_rows and not over_rows:
        print("Nothing to plot.")
        return

    all_folders = sorted({r["folder"] for r in (*bias_rows, *over_rows)})
    cmap = plt.get_cmap("tab10")
    colors = {f: cmap(i % 10) for i, f in enumerate(all_folders)}

    amps = sorted({r["amp"] for r in (*bias_rows, *over_rows)})
    print(f"Amps: {amps}")
    for amp in amps:
        b = [r for r in bias_rows if r["amp"] == amp]
        o = [r for r in over_rows if r["amp"] == amp]
        if not args.temperature_only:
            outpath = f"{out_prefix}_{amp}.png"
            plot_per_amp(amp, b, o, colors, outpath)
        if any(r.get("temp") is not None for r in (*b, *o)):
            temp_outpath = f"{out_prefix}_vs_temperature_{amp}.png"
            plot_per_amp_temperature(amp, b, o, colors, temp_outpath)


if __name__ == "__main__":
    main()
