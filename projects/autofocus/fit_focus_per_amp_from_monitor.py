#!/usr/bin/env python3
"""
Fit one focus curve per amplifier from realtime_focus_monitor outputs.

This script reads the catalog and time-series products already produced by
focus_pipeline.py / realtime_focus_monitor.py:

  * focus_sources.fits
  * focus_time_series.ecsv

For each science exposure and amplifier, it computes the median FWHM of
GMM-selected stars.  It then fits

    FWHM = A * (focus_position - h)^2 + k

separately for each amplifier.  The fitted vertex h is the best focus for
that amplifier.  The script also reports the average best focus across amps.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from focus_pipeline import compute_gmm_labels, fit_parabola_vertex_form


IMAGE_REGEX = re.compile(r"(\d+)")


def expand_numbers(tokens: Optional[Iterable[str]]) -> Optional[set]:
    """Expand tokens like 150 151-156 into a set of image numbers."""
    if not tokens:
        return None
    out = set()
    for raw in tokens:
        token = str(raw).strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(token))
    return out


def image_number(filename: str) -> Optional[int]:
    matches = IMAGE_REGEX.findall(Path(str(filename)).stem)
    return int(matches[-1]) if matches else None


def parse_subset_id(value) -> Tuple[Optional[int], Optional[int]]:
    text = value.decode() if isinstance(value, (bytes, bytearray)) else str(value)
    match = re.fullmatch(r"(\d+)_amp(\d+)", text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def quality_mask(
    tbl: Table,
    fwhm_max: float,
    flag_max: int,
    flux_ratio_min: float,
) -> np.ndarray:
    mask = np.ones(len(tbl), dtype=bool)
    if "FWHM" in tbl.colnames:
        fwhm = np.array(tbl["FWHM"], dtype=float)
        mask &= np.isfinite(fwhm) & (fwhm < fwhm_max)
    if "flag" in tbl.colnames:
        flag = np.array(tbl["flag"], dtype=int)
        mask &= flag <= flag_max
    if "flux_ratio" in tbl.colnames:
        fr = np.array(tbl["flux_ratio"], dtype=float)
        mask &= np.isfinite(fr) & (fr > flux_ratio_min)
    return mask


def build_per_amp_points(
    sources: Table,
    time_series: Table,
    amps: List[int],
    fit_numbers: Optional[set],
    fwhm_max: float,
    flag_max: int,
    flux_ratio_min: float,
) -> Table:
    """Return one median-FWHM row per exposure and amplifier."""
    if "subset_id" not in sources.colnames:
        raise ValueError("focus_sources.fits must contain a subset_id column")
    if "focus_position" not in time_series.colnames:
        raise ValueError("focus_time_series.ecsv must contain focus_position")

    subset_pairs = [parse_subset_id(v) for v in sources["subset_id"]]
    file_idx_arr = np.array([p[0] if p[0] is not None else -1 for p in subset_pairs], dtype=int)
    amp_arr = np.array([p[1] if p[1] is not None else -1 for p in subset_pairs], dtype=int)

    rows = []
    for file_idx, row in enumerate(time_series):
        img_num = image_number(str(row["sci_file"]))
        if fit_numbers is not None and img_num not in fit_numbers:
            continue

        in_file = file_idx_arr == file_idx
        if not np.any(in_file):
            continue
        file_tbl = sources[in_file]
        file_amp_arr = amp_arr[in_file]

        good = quality_mask(file_tbl, fwhm_max, flag_max, flux_ratio_min)
        cut = file_tbl[good]
        cut_amp_arr = file_amp_arr[good]
        if len(cut) < 2:
            continue

        labels = compute_gmm_labels(cut)
        fwhm = np.array(cut["FWHM"], dtype=float)
        ellip = np.array(cut["e"], dtype=float) if "e" in cut.colnames else np.full(len(cut), np.nan)

        for amp in amps:
            sel = (cut_amp_arr == amp) & (labels == 0) & np.isfinite(fwhm)
            vals = fwhm[sel]
            if vals.size == 0:
                med_fwhm = np.nan
                med_e = np.nan
                n_stars = 0
            else:
                med_fwhm = float(np.nanmedian(vals))
                med_e = float(np.nanmedian(ellip[sel]))
                n_stars = int(vals.size)

            rows.append(
                {
                    "sci_file": str(row["sci_file"]),
                    "image_number": img_num if img_num is not None else -1,
                    "amp": amp,
                    "focus_position": float(row["focus_position"]),
                    "median_fwhm": med_fwhm,
                    "median_e": med_e,
                    "n_stars": n_stars,
                    "filter": str(row["filter"]) if "filter" in time_series.colnames else "",
                    "obs_time_iso": str(row["obs_time_iso"]) if "obs_time_iso" in time_series.colnames else "",
                }
            )

    return Table(rows=rows)


def fit_per_amp(points: Table, amps: List[int], min_points: int) -> Table:
    rows = []
    for amp in amps:
        sub = points[points["amp"] == amp]
        x = np.array(sub["focus_position"], dtype=float)
        y = np.array(sub["median_fwhm"], dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        if len(x) < min_points:
            rows.append(
                {
                    "amp": amp,
                    "n_points": int(len(x)),
                    "best_focus": np.nan,
                    "best_focus_sigma": np.nan,
                    "min_fwhm": np.nan,
                    "A": np.nan,
                    "R2": np.nan,
                    "status": "not_enough_points",
                }
            )
            continue

        order = np.argsort(x)
        x = x[order]
        y = y[order]
        try:
            fit = fit_parabola_vertex_form(x, y)
            rows.append(
                {
                    "amp": amp,
                    "n_points": int(len(x)),
                    "best_focus": float(fit["h"]),
                    "best_focus_sigma": (
                        float(fit["sigma_x_min"])
                        if fit.get("sigma_x_min") is not None
                        else np.nan
                    ),
                    "min_fwhm": float(fit["k"]),
                    "A": float(fit["A"]),
                    "R2": float(fit["R2"]),
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "amp": amp,
                    "n_points": int(len(x)),
                    "best_focus": np.nan,
                    "best_focus_sigma": np.nan,
                    "min_fwhm": np.nan,
                    "A": np.nan,
                    "R2": np.nan,
                    "status": f"fit_failed: {exc}",
                }
            )
    return Table(rows=rows)


def average_focus_and_error(fits: Table) -> Tuple[float, float, int]:
    """Return mean best focus, standard error across amps, and number of good amps."""
    best_focus = np.array(fits["best_focus"], dtype=float)
    good = np.isfinite(best_focus)
    n_good = int(good.sum())
    if n_good == 0:
        return np.nan, np.nan, 0
    avg = float(np.nanmean(best_focus[good]))
    if n_good > 1:
        err = float(np.nanstd(best_focus[good], ddof=1) / np.sqrt(n_good))
    else:
        err = np.nan
    return avg, err, n_good


def plot_per_amp_fits(
    points: Table,
    fits: Table,
    output_path: Path,
    average_best_focus: float,
    average_best_focus_error: float,
    n_good_amps: int,
    show_fit_errors: bool = True,
) -> None:
    amps = [int(a) for a in fits["amp"]]
    n_cols = 4
    n_rows = int(np.ceil(len(amps) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.2 * n_rows), squeeze=False)

    for ax, amp in zip(axes.ravel(), amps):
        sub = points[points["amp"] == amp]
        x = np.array(sub["focus_position"], dtype=float)
        y = np.array(sub["median_fwhm"], dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        ax.scatter(x, y, s=24, color="tab:blue", label="Frame")

        fit_row = fits[fits["amp"] == amp][0]
        if str(fit_row["status"]) == "ok" and len(x) >= 3:
            xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
            A = float(fit_row["A"])
            h = float(fit_row["best_focus"])
            k = float(fit_row["min_fwhm"])
            h_sigma = float(fit_row["best_focus_sigma"])
            ax.plot(xx, A * (xx - h) ** 2 + k, color="tab:green", lw=1.5)
            ax.axvline(h, color="tab:green", ls="--", alpha=0.6)
            sigma_text = (
                f" +/- {h_sigma:.2f}"
                if show_fit_errors and np.isfinite(h_sigma)
                else ""
            )
            ax.set_title(f"Amp {amp}: {h:.2f}{sigma_text}")
        else:
            ax.set_title(f"Amp {amp}: no fit")
        ax.set_xlabel("Focus position")
        ax.set_ylabel("Median FWHM (pix)")
        ax.grid(True, alpha=0.25)

    for ax in axes.ravel()[len(amps):]:
        ax.axis("off")

    if np.isfinite(average_best_focus):
        if show_fit_errors and np.isfinite(average_best_focus_error):
            title = (
                f"Per-amplifier best focus; average = "
                f"{average_best_focus:.2f} +/- {average_best_focus_error:.2f} "
                f"(SEM, {n_good_amps} amps)"
            )
        else:
            title = f"Per-amplifier best focus; average = {average_best_focus:.2f}"
        fig.suptitle(title, fontsize=14, y=1.02)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit best focus independently for each amplifier from realtime monitor outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sources", default="focus_output/focus_sources.fits")
    parser.add_argument("--time-series", default="focus_output/focus_time_series.ecsv")
    parser.add_argument("--outdir", default="focus_output")
    parser.add_argument("--amps", nargs="+", type=int, default=list(range(1, 9)))
    parser.add_argument(
        "--fit-nums",
        nargs="*",
        default=None,
        help="Optional image numbers/ranges to fit, e.g. --fit-nums 150-156",
    )
    parser.add_argument("--min-points", type=int, default=3)
    parser.add_argument("--quality-fwhm-max", type=float, default=15.0)
    parser.add_argument("--quality-flag-max", type=int, default=0)
    parser.add_argument("--quality-flux-ratio-min", type=float, default=1.0)
    parser.add_argument("--points-name", default="focus_per_amp_points.ecsv")
    parser.add_argument("--summary-name", default="focus_per_amp_best_focus.ecsv")
    parser.add_argument("--plot-name", default="focus_per_amp_best_focus.png")
    parser.add_argument(
        "--hide-fit-errors",
        action="store_true",
        help="Hide best-focus uncertainty text in the plot while using the same fit data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sources = Table.read(args.sources)
    time_series = Table.read(args.time_series, format="ascii.ecsv")
    fit_numbers = expand_numbers(args.fit_nums)

    points = build_per_amp_points(
        sources=sources,
        time_series=time_series,
        amps=args.amps,
        fit_numbers=fit_numbers,
        fwhm_max=args.quality_fwhm_max,
        flag_max=args.quality_flag_max,
        flux_ratio_min=args.quality_flux_ratio_min,
    )
    points_path = outdir / args.points_name
    points.write(points_path, format="ascii.ecsv", overwrite=True)

    fits = fit_per_amp(points, args.amps, args.min_points)
    avg_best_focus, avg_best_focus_error, n_good_amps = average_focus_and_error(fits)
    fits.meta["average_best_focus"] = avg_best_focus
    fits.meta["average_best_focus_error_sem"] = avg_best_focus_error
    fits.meta["n_good_amps"] = n_good_amps
    fits.meta["fit_numbers"] = sorted(fit_numbers) if fit_numbers is not None else "all"

    summary_path = outdir / args.summary_name
    fits.write(summary_path, format="ascii.ecsv", overwrite=True)

    plot_path = outdir / args.plot_name
    plot_per_amp_fits(
        points,
        fits,
        plot_path,
        average_best_focus=avg_best_focus,
        average_best_focus_error=avg_best_focus_error,
        n_good_amps=n_good_amps,
        show_fit_errors=not args.hide_fit_errors,
    )

    print(f"Wrote per-amp FWHM points: {points_path}")
    print(f"Wrote per-amp best-focus summary: {summary_path}")
    print(f"Saved plot: {plot_path}")
    print()
    for row in fits:
        if str(row["status"]) == "ok":
            sigma = float(row["best_focus_sigma"])
            sigma_text = f" +/- {sigma:.3f}" if np.isfinite(sigma) else ""
            print(
                f"Amp {int(row['amp'])}: best focus = "
                f"{float(row['best_focus']):.3f}{sigma_text} "
                f"(R^2={float(row['R2']):.3f}, N={int(row['n_points'])})"
            )
        else:
            print(f"Amp {int(row['amp'])}: {row['status']} (N={int(row['n_points'])})")
    avg_sigma_text = (
        f" +/- {avg_best_focus_error:.3f}"
        if np.isfinite(avg_best_focus_error)
        else ""
    )
    print(
        f"\nAverage best focus across {fits.meta['n_good_amps']} amps = "
        f"{avg_best_focus:.3f}{avg_sigma_text}"
    )


if __name__ == "__main__":
    main()
