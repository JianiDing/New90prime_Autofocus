#!/usr/bin/env python3
"""Summarize repeated actuator-tilt measurements.

Use this after running ``focus_pipeline.py --solve-tilt`` on repeated exposures
with the same setup.  The script reads the resulting ``tilt_result_*.json``
files, subtracts piston to isolate tilt, and reports whether the mean actuator
tilt is large compared with trial-to-trial scatter.

It can also recompute the tilt solutions from ``focus_per_amp_points.ecsv`` when
JSON files are not available yet.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


ACTUATORS = ("A", "B", "C")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")


def _expand_image_numbers(entries: Iterable[str]) -> List[int]:
    """Expand CLI tokens like ``150`` or ``150-154`` into image numbers."""

    expanded: List[int] = []
    for entry in entries:
        token = str(entry).strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            step = 1 if end >= start else -1
            expanded.extend(range(start, end + step, step))
        else:
            expanded.append(int(token))

    seen = set()
    out: List[int] = []
    for value in expanded:
        if value not in seen:
            out.append(value)
            seen.add(value)
    if not out:
        raise ValueError("No valid image numbers provided.")
    return out


def _finite_float(value, default=np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _expand_inputs(patterns: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            files.extend(Path(m) for m in matches)
        else:
            path = Path(pattern)
            if path.exists():
                files.append(path)
    seen = set()
    unique: List[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def _tilt_only(corrections: Dict[str, float]) -> Dict[str, float]:
    vals = np.array([_finite_float(corrections.get(name)) for name in ACTUATORS])
    piston = float(np.nanmean(vals))
    return {name: float(_finite_float(corrections.get(name)) - piston) for name in ACTUATORS}


def _trial_from_tilt_json(path: Path) -> Dict:
    with path.open() as fp:
        payload = json.load(fp)

    corrections = {name: _finite_float(payload.get("corrections", {}).get(name)) for name in ACTUATORS}
    current_lvdt = payload.get("current_lvdt", {}) or {}
    lvdt_values = [_finite_float(current_lvdt.get(name)) for name in ACTUATORS]
    lvdt_mean = float(np.nanmean(lvdt_values)) if np.isfinite(lvdt_values).any() else np.nan

    return {
        "source": str(path),
        "frame": payload.get("frame") or path.name,
        "image_number": payload.get("image_number"),
        "filter": payload.get("filter"),
        "focus_position": _finite_float(payload.get("focus_position"), lvdt_mean),
        "current_lvdt": {name: _finite_float(current_lvdt.get(name)) for name in ACTUATORS},
        "raw": corrections,
        "tilt_only": _tilt_only(corrections),
        "tilt_a": _finite_float(payload.get("tilt_a")),
        "tilt_b": _finite_float(payload.get("tilt_b")),
        "tilt_magnitude": _finite_float(payload.get("tilt_magnitude")),
        "R2": _finite_float(payload.get("R2", payload.get("R2_global"))),
        "fit_mode": payload.get("fit_mode", "unknown"),
    }


def _trials_from_points_ecsv(path: Path, min_amps: int) -> List[Dict]:
    from astropy.table import Table

    from focus_pipeline import solve_tilt_focus

    table = Table.read(path, format="ascii.ecsv")
    required = {"image_number", "amp", "median_fwhm"}
    missing = required - set(table.colnames)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    by_image: Dict[int, List] = defaultdict(list)
    for row in table:
        by_image[int(row["image_number"])].append(row)

    trials: List[Dict] = []
    for image_number, rows in sorted(by_image.items()):
        fwhm_per_amp = {
            int(row["amp"]): float(row["median_fwhm"])
            for row in rows
            if np.isfinite(float(row["median_fwhm"]))
        }
        if len(fwhm_per_amp) < min_amps:
            continue
        result = solve_tilt_focus(fwhm_per_amp, {"A": 0.0, "B": 0.0, "C": 0.0})
        first = rows[0]
        corrections = {name: _finite_float(result["corrections"].get(name)) for name in ACTUATORS}
        trials.append({
            "source": str(path),
            "frame": str(first["sci_file"]) if "sci_file" in table.colnames else f"image_{image_number}",
            "image_number": image_number,
            "filter": str(first["filter"]) if "filter" in table.colnames else None,
            "focus_position": (
                _finite_float(first["focus_position"])
                if "focus_position" in table.colnames
                else np.nan
            ),
            "current_lvdt": {"A": np.nan, "B": np.nan, "C": np.nan},
            "raw": corrections,
            "tilt_only": _tilt_only(corrections),
            "tilt_a": _finite_float(result.get("tilt_a")),
            "tilt_b": _finite_float(result.get("tilt_b")),
            "tilt_magnitude": _finite_float(result.get("tilt_magnitude")),
            "R2": _finite_float(result.get("R2")),
            "fit_mode": "recomputed_from_ecsv",
        })
    return trials


def _group_key(trial: Dict, mode: str, tolerance: float) -> str:
    if mode == "all":
        return "all_trials"
    focus = trial.get("focus_position", np.nan)
    focus_label = "unknown_focus"
    if np.isfinite(focus):
        focus_label = f"focus_{round(float(focus) / tolerance) * tolerance:.3f}"
    if mode == "focus":
        return focus_label
    filt = trial.get("filter") or "unknown_filter"
    return f"{filt}_{focus_label}"


def _align_signs(trials: List[Dict]) -> None:
    """Flip degenerate per-frame tilt signs onto the first nonzero reference."""

    ref = None
    for trial in trials:
        vec = np.array([trial["tilt_only"][name] for name in ACTUATORS], dtype=float)
        if np.linalg.norm(vec) > 0:
            ref = vec
            break
    if ref is None:
        return

    for trial in trials:
        vec = np.array([trial["tilt_only"][name] for name in ACTUATORS], dtype=float)
        if np.dot(vec, ref) >= 0:
            trial["sign_flipped"] = False
            continue
        for bucket in ("raw", "tilt_only"):
            for name in ACTUATORS:
                trial[bucket][name] = -trial[bucket][name]
        for key in ("tilt_a", "tilt_b"):
            if np.isfinite(trial[key]):
                trial[key] = -trial[key]
        trial["sign_flipped"] = True


def _series_stats(values: np.ndarray) -> Dict[str, float]:
    values = values[np.isfinite(values)]
    n = int(values.size)
    mean = float(np.mean(values)) if n else np.nan
    std = float(np.std(values, ddof=1)) if n > 1 else np.nan
    sem = float(std / math.sqrt(n)) if n > 1 else np.nan
    snr = float(abs(mean) / std) if n > 1 and std > 0 else np.nan
    ci95 = float(1.96 * sem) if np.isfinite(sem) else np.nan
    signs = np.sign(values[np.abs(values) > 0])
    if signs.size:
        consistent = float(max(np.mean(signs > 0), np.mean(signs < 0)))
    else:
        consistent = np.nan
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "snr_abs_mean_over_std": snr,
        "ci95_half_width": ci95,
        "consistent_sign_fraction": consistent,
    }


def _summarize_group(name: str, trials: List[Dict], signal_sigma: float) -> Tuple[List[Dict], Dict]:
    rows: List[Dict] = []
    group_summary = {"group": name, "n_trials": len(trials), "actuators": {}, "tilt_plane": {}}

    for quantity in ("raw", "tilt_only"):
        for actuator in ACTUATORS:
            values = np.array([trial[quantity][actuator] for trial in trials], dtype=float)
            stats = _series_stats(values)
            is_repeatable = (
                stats["n"] >= 3
                and np.isfinite(stats["snr_abs_mean_over_std"])
                and stats["snr_abs_mean_over_std"] >= signal_sigma
                and stats["consistent_sign_fraction"] >= 0.8
            )
            row = {
                "group": name,
                "quantity": quantity,
                "actuator": actuator,
                **stats,
                "repeatable_signal": bool(is_repeatable),
            }
            rows.append(row)
            group_summary["actuators"][f"{quantity}_{actuator}"] = row

    for quantity in ("tilt_a", "tilt_b", "tilt_magnitude", "R2"):
        values = np.array([trial[quantity] for trial in trials], dtype=float)
        group_summary["tilt_plane"][quantity] = _series_stats(values)

    return rows, group_summary


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_plot(path: Path, grouped: Dict[str, List[Dict]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    n_groups = len(grouped)
    fig, axes = plt.subplots(n_groups, 1, figsize=(8, max(3, 3 * n_groups)), squeeze=False)
    for ax, (group, trials) in zip(axes[:, 0], grouped.items()):
        x = np.arange(len(trials))
        for actuator in ACTUATORS:
            y = [trial["tilt_only"][actuator] for trial in trials]
            ax.plot(x, y, marker="o", label=actuator)
        ax.axhline(0, color="0.4", lw=1)
        ax.set_title(group)
        ax.set_xlabel("trial")
        ax.set_ylabel("tilt-only actuator delta")
        ax.legend(title="Actuator")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _load_points_table(path: Path):
    from astropy.table import Table

    table = Table.read(path, format="ascii.ecsv")
    required = {"image_number", "amp", "median_fwhm"}
    missing = required - set(table.colnames)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return table


def _average_fwhm_by_amp(table, image_numbers: Iterable[int], min_exposures: int) -> Tuple[Dict[int, float], Dict[int, Dict]]:
    image_set = set(int(n) for n in image_numbers)
    selected = table[np.isin(np.array(table["image_number"], dtype=int), list(image_set))]
    if len(selected) == 0:
        raise ValueError(f"No rows found for image numbers {sorted(image_set)}.")

    fwhm_per_amp: Dict[int, float] = {}
    stats: Dict[int, Dict] = {}
    for amp in sorted(set(int(v) for v in selected["amp"])):
        rows = selected[np.array(selected["amp"], dtype=int) == amp]
        vals = np.array(rows["median_fwhm"], dtype=float)
        vals = vals[np.isfinite(vals)]
        n = int(vals.size)
        if n < min_exposures:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else np.nan
        sem = float(std / math.sqrt(n)) if n > 1 else np.nan
        fwhm_per_amp[amp] = mean
        stats[amp] = {
            "amp": amp,
            "n_exposures": n,
            "mean_fwhm_pix": mean,
            "std_fwhm_pix": std,
            "sem_fwhm_pix": sem,
            "image_numbers": sorted(int(v) for v in set(rows["image_number"])),
            "mean_n_stars": (
                float(np.mean(np.array(rows["n_stars"], dtype=float)))
                if "n_stars" in table.colnames
                else np.nan
            ),
        }
    return fwhm_per_amp, stats


def _write_amp_stats_csv(path: Path, amp_stats: Dict[int, Dict]) -> None:
    rows = [amp_stats[amp] for amp in sorted(amp_stats)]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "amp",
        "n_exposures",
        "mean_fwhm_pix",
        "std_fwhm_pix",
        "sem_fwhm_pix",
        "mean_n_stars",
        "image_numbers",
    ]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean = dict(row)
            clean["image_numbers"] = " ".join(str(v) for v in row["image_numbers"])
            writer.writerow(clean)


def _plot_average_tilt_map(
    fwhm_per_amp: Dict[int, float],
    amp_stats: Dict[int, Dict],
    tilt_result: Dict,
    output_path: Path,
    pixscale: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from focus_pipeline import _DEFAULT_AMP_POSITIONS

    amps = sorted(fwhm_per_amp)
    x = np.array([_DEFAULT_AMP_POSITIONS[a][0] for a in amps])
    y = np.array([_DEFAULT_AMP_POSITIONS[a][1] for a in amps])
    fwhm_arcsec = np.array([fwhm_per_amp[a] for a in amps]) * pixscale
    sem_arcsec = np.array([amp_stats[a]["sem_fwhm_pix"] for a in amps]) * pixscale
    defocus = np.array([tilt_result["defocus_per_amp"][a] for a in amps])

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    xpad = max(0.4 * (xmax - xmin), 0.5)
    ypad = max(0.8 * (ymax - ymin), 0.8)

    sc = axes[0].scatter(
        x,
        y,
        c=fwhm_arcsec,
        s=950,
        cmap="RdYlGn_r",
        edgecolor="k",
        vmin=float(np.nanmin(fwhm_arcsec) - 0.05),
        vmax=float(np.nanmax(fwhm_arcsec) + 0.05),
        zorder=5,
    )
    for amp, xv, yv, mean, sem in zip(amps, x, y, fwhm_arcsec, sem_arcsec):
        err = "nan" if not np.isfinite(sem) else f"{sem:.3f}"
        axes[0].text(
            xv,
            yv,
            f"amp {amp}\n{mean:.3f}\" +/- {err}\"",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    axes[0].set_title("Average Measured FWHM", fontsize=14)
    axes[0].set_xlabel("Focal plane X", fontsize=13)
    axes[0].set_ylabel("Focal plane Y", fontsize=13)
    axes[0].set_xlim(xmin - xpad, xmax + xpad)
    axes[0].set_ylim(ymin - ypad, ymax + ypad)
    plt.colorbar(sc, ax=axes[0], label='mean FWHM (")')

    sc2 = axes[1].scatter(x, y, c=defocus, s=950, cmap="coolwarm", edgecolor="k", zorder=5)
    for amp, xv, yv, dz in zip(amps, x, y, defocus):
        axes[1].text(xv, yv, f"amp {amp}\n{dz:+.3f}", ha="center", va="center", fontsize=10, fontweight="bold")
    axes[1].set_title(f"Defocus Plane  (tilt = {tilt_result['tilt_magnitude']:.4f})", fontsize=14)
    axes[1].set_xlabel("Focal plane X", fontsize=13)
    axes[1].set_xlim(xmin - xpad, xmax + xpad)
    axes[1].set_ylim(ymin - ypad, ymax + ypad)
    plt.colorbar(sc2, ax=axes[1], label="defocus")

    names = list(tilt_result["corrections"].keys())
    raw_deltas = np.array([tilt_result["corrections"][name] for name in names], dtype=float)
    tilt_only = raw_deltas - float(np.nanmean(raw_deltas))
    colors = ["#4682B4" if value >= 0 else "#C44E52" for value in tilt_only]
    axes[2].bar(names, tilt_only, color=colors, edgecolor="k", width=0.5)
    for i, (name, value) in enumerate(zip(names, tilt_only)):
        direction = "HOLD" if abs(value) < 0.01 else ("INC LVDT" if value > 0 else "DEC LVDT")
        axes[2].text(
            i,
            value,
            f"{direction}\n{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=11,
            fontweight="bold",
        )
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set_ylabel("Tilt-only correction", fontsize=13)
    axes[2].set_title("Average Actuator Tilt", fontsize=14)

    n_images = len(set(v for stat in amp_stats.values() for v in stat["image_numbers"]))
    fig.suptitle(
        f"Averaged Tilt Map from {n_images} exposure(s)   |   "
        f"R^2 = {tilt_result['R2']:.3f}   |   "
        f"seeing floor = {tilt_result['seeing_floor'] * pixscale:.2f}\"",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _run_average_tilt_set(args) -> None:
    from focus_pipeline import actuator_direction_summary, solve_tilt_focus

    if not args.points_ecsv:
        raise SystemExit("--average-images requires --points-ecsv.")
    image_numbers = _expand_image_numbers(args.average_images)
    table = _load_points_table(Path(args.points_ecsv))
    fwhm_per_amp, amp_stats = _average_fwhm_by_amp(table, image_numbers, args.min_average_exposures)
    if len(fwhm_per_amp) < args.min_amps:
        raise SystemExit(
            f"Only {len(fwhm_per_amp)} amps have enough repeated FWHM measurements; "
            f"need at least {args.min_amps}."
        )

    if args.current_lvdt:
        current_lvdt = {name: float(value) for name, value in zip(ACTUATORS, args.current_lvdt)}
    else:
        selected = table[np.isin(np.array(table["image_number"], dtype=int), image_numbers)]
        focus = np.array(selected["focus_position"], dtype=float) if "focus_position" in table.colnames else np.array([])
        focus_mean = float(np.nanmean(focus)) if focus.size else 0.0
        current_lvdt = {name: focus_mean for name in ACTUATORS}

    tilt_result = solve_tilt_focus(fwhm_per_amp, current_lvdt)
    label = args.average_label or f"images_{image_numbers[0]}_{image_numbers[-1]}"
    outdir = Path(args.average_outdir)
    map_path = outdir / f"average_tilt_map_{label}.png"
    json_path = outdir / f"average_tilt_result_{label}.json"
    csv_path = outdir / f"average_fwhm_per_amp_{label}.csv"

    _plot_average_tilt_map(fwhm_per_amp, amp_stats, tilt_result, map_path, args.pixscale)
    _write_amp_stats_csv(csv_path, amp_stats)

    payload = {
        "label": label,
        "image_numbers": image_numbers,
        "current_lvdt": current_lvdt,
        "fwhm_per_amp_mean_pix": {str(k): float(v) for k, v in fwhm_per_amp.items()},
        "fwhm_per_amp_stats": {str(k): v for k, v in amp_stats.items()},
        "optimal_lvdt": tilt_result["optimal_lvdt"],
        "corrections": tilt_result["corrections"],
        "actuator_directions": actuator_direction_summary(tilt_result["corrections"], current_lvdt=current_lvdt),
        "seeing_floor_pix": float(tilt_result["seeing_floor"]),
        "seeing_floor_arcsec": float(tilt_result["seeing_floor"] * args.pixscale),
        "piston_z0": float(tilt_result["piston_z0"]),
        "tilt_a": float(tilt_result["tilt_a"]),
        "tilt_b": float(tilt_result["tilt_b"]),
        "tilt_magnitude": float(tilt_result["tilt_magnitude"]),
        "R2": float(tilt_result["R2"]),
        "tilt_map_png": map_path.name,
        "fit_mode": "average_repeated_exposures",
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as fp:
        json.dump(payload, fp, indent=2)

    print(f"Averaged {len(image_numbers)} requested image(s): {image_numbers}")
    for amp in sorted(amp_stats):
        stat = amp_stats[amp]
        print(
            f"  amp {amp}: FWHM={stat['mean_fwhm_pix']:.4g} pix, "
            f"std={stat['std_fwhm_pix']:.4g}, sem={stat['sem_fwhm_pix']:.4g}, "
            f"n={stat['n_exposures']}"
        )
    print("\nTilt-only actuator deltas:")
    raw = np.array([tilt_result["corrections"][name] for name in ACTUATORS], dtype=float)
    tilt_only = raw - float(np.nanmean(raw))
    for name, value in zip(ACTUATORS, tilt_only):
        print(f"  {name}: {value:+.4g}")
    print(f"\nWrote {map_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure repeatability of actuator tilt across repeated exposures."
    )
    parser.add_argument(
        "--tilt-json",
        nargs="+",
        help="Tilt result JSON files or glob patterns, e.g. 'focus_output/tilt_result_*.json'.",
    )
    parser.add_argument(
        "--points-ecsv",
        help="Optional focus_per_amp_points.ecsv file; tilt is recomputed per image.",
    )
    parser.add_argument(
        "--average-images",
        nargs="+",
        help=(
            "Image numbers for one repeated setup, e.g. '150-154'. "
            "Computes average FWHM per amp and writes an averaged tilt map."
        ),
    )
    parser.add_argument(
        "--average-label",
        help="Short label for averaged output files, e.g. A1_B1_C1.",
    )
    parser.add_argument(
        "--average-outdir",
        default="focus_output",
        help="Output directory for averaged tilt products.",
    )
    parser.add_argument(
        "--current-lvdt",
        nargs=3,
        type=float,
        metavar=("A", "B", "C"),
        help="Current actuator positions for the averaged setup.",
    )
    parser.add_argument(
        "--min-average-exposures",
        type=int,
        default=2,
        help="Minimum repeated exposures required per amplifier in average mode.",
    )
    parser.add_argument("--pixscale", type=float, default=0.455)
    parser.add_argument(
        "--group-by",
        choices=("all", "focus", "filter_focus"),
        default="all",
        help="How to group repeated trials. Use 'focus' or 'filter_focus' for several setups.",
    )
    parser.add_argument(
        "--focus-tolerance",
        type=float,
        default=0.5,
        help="Focus/LVDT tolerance used when grouping by focus.",
    )
    parser.add_argument(
        "--signal-sigma",
        type=float,
        default=3.0,
        help="Mean/std threshold for flagging a repeatable tilt signal.",
    )
    parser.add_argument(
        "--no-align-signs",
        action="store_true",
        help="Disable sign alignment for the defocus-squared tilt degeneracy.",
    )
    parser.add_argument("--out-csv", default="focus_output/repeated_tilt_summary.csv")
    parser.add_argument("--out-json", default="focus_output/repeated_tilt_summary.json")
    parser.add_argument("--plot", default="focus_output/repeated_tilt_trials.png")
    parser.add_argument("--min-amps", type=int, default=5)
    args = parser.parse_args()

    if args.average_images:
        _run_average_tilt_set(args)
        return 0

    trials: List[Dict] = []
    if args.tilt_json:
        files = _expand_inputs(args.tilt_json)
        if not files:
            raise SystemExit("No tilt JSON files matched --tilt-json.")
        trials.extend(_trial_from_tilt_json(path) for path in files)
    if args.points_ecsv:
        trials.extend(_trials_from_points_ecsv(Path(args.points_ecsv), args.min_amps))
    if not trials:
        raise SystemExit("Provide --tilt-json and/or --points-ecsv.")

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for trial in sorted(trials, key=lambda t: (str(t.get("filter")), t.get("focus_position", np.nan), str(t.get("frame")))):
        grouped[_group_key(trial, args.group_by, args.focus_tolerance)].append(trial)

    if not args.no_align_signs:
        for group_trials in grouped.values():
            _align_signs(group_trials)

    rows: List[Dict] = []
    summary = {
        "n_trials": len(trials),
        "group_by": args.group_by,
        "focus_tolerance": args.focus_tolerance,
        "signal_sigma": args.signal_sigma,
        "sign_alignment": not args.no_align_signs,
        "groups": {},
        "trials": trials,
    }
    for group_name, group_trials in grouped.items():
        group_rows, group_summary = _summarize_group(group_name, group_trials, args.signal_sigma)
        rows.extend(group_rows)
        summary["groups"][group_name] = group_summary

    _write_csv(Path(args.out_csv), rows)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_json).open("w") as fp:
        json.dump(summary, fp, indent=2)
    _make_plot(Path(args.plot), grouped)

    print(f"Analyzed {len(trials)} trial(s) in {len(grouped)} group(s).")
    for group_name, group_trials in grouped.items():
        print(f"\n[{group_name}] n={len(group_trials)}")
        for actuator in ACTUATORS:
            row = summary["groups"][group_name]["actuators"][f"tilt_only_{actuator}"]
            print(
                f"  {actuator}: mean={row['mean']:.4g}, std={row['std']:.4g}, "
                f"|mean|/std={row['snr_abs_mean_over_std']:.3g}, "
                f"repeatable={row['repeatable_signal']}"
            )
    print(f"\nWrote {args.out_csv}")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
