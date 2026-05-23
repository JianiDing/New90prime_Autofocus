#!/usr/bin/env python3
"""Focus calculation pipeline converted from the original notebook."""

from __future__ import annotations

import argparse
import base64
import json
import os
import warnings
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # allow plotting without a display
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.time import Time
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import sep

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ---------------------------------------------------------------------------
# Utility functions mostly copied from the notebook with minimal edits
# ---------------------------------------------------------------------------

def skim_fits_files(
    directory: str = ".",
    object_keyword: str = "OBJECT",
    filter_keywords: Tuple[str, ...] = ("FILTER", "FILTERS"),
    target_bands: Tuple[str, ...] = ("U", "G", "R", "I", "Z"),
    exclusion_list: Optional[List[str]] = None,
) -> Dict:
    """Classify FITS files into calibration and per-band science buckets."""

    if exclusion_list is None:
        exclusion_list = []

    target_bands = [b.upper() for b in target_bands]
    categorized_files = {band: {"flat": [], "other": []} for band in target_bands}
    categorized_files["bias_frames"] = []
    categorized_files["dark_frames"] = []
    categorized_files["unmatched"] = []
    categorized_files["excluded"] = []

    fits_files = list(Path(directory).glob("*.fits")) + list(Path(directory).glob("*.fit"))
    fits_files = sorted(fits_files)
    print(f"Found {len(fits_files)} FITS files to check in '{directory}'.")

    for filename in fits_files:
        basename = filename.name
        if any(ex and (ex in basename) for ex in exclusion_list or []):
            categorized_files["excluded"].append(str(filename))
            continue

        try:
            with fits.open(filename, memmap=False) as hdul:
                hdr0 = hdul[0].header
                obj_val = str(hdr0.get(object_keyword, "")).strip()
                obj_val_lower = obj_val.lower()

                filter_val = ""
                for fk in filter_keywords:
                    fv = hdr0.get(fk)
                    if fv is not None and str(fv).strip():
                        filter_val = str(fv).strip()
                        break
                filter_val_lower = filter_val.lower()

                if "bias" in obj_val_lower or "zero" in obj_val_lower:
                    categorized_files["bias_frames"].append(str(filename))
                    continue
                if "dark" in obj_val_lower:
                    categorized_files["dark_frames"].append(str(filename))
                    continue

                if "flat" in obj_val_lower:
                    found_band = None
                    for b in target_bands:
                        if (filter_val and b.lower() in filter_val_lower) or (b.lower() in obj_val_lower):
                            found_band = b
                            break
                    if found_band:
                        categorized_files[found_band]["flat"].append(str(filename))
                    else:
                        categorized_files["unmatched"].append(
                            f"{filename} (OBJECT: {obj_val}, FILTER: {filter_val})"
                        )
                    continue

                found_band = None
                if filter_val:
                    for b in target_bands:
                        if b.lower() == filter_val_lower or b.lower() in filter_val_lower:
                            found_band = b
                            break
                if not found_band and not filter_val:
                    # Only fall back to OBJECT name when no FILTER keyword exists
                    for b in target_bands:
                        if b.lower() in obj_val_lower:
                            found_band = b
                            break

                if found_band:
                    categorized_files[found_band]["other"].append(str(filename))
                else:
                    extra_info = f"OBJECT: {obj_val}"
                    if filter_val:
                        extra_info += f", FILTER: {filter_val}"
                    categorized_files["unmatched"].append(f"{filename} ({extra_info})")
        except Exception as exc:  # pragma: no cover - diagnostic print preserved
            print(f"Error processing file {filename}: {exc}")
            categorized_files["unmatched"].append(f"{filename} (error: {exc})")

    print("\n--- Skimming Complete ---")
    print(f"Total FITS files found: {len(fits_files)}")
    print(f"Total files processed: {len(fits_files) - len(categorized_files['excluded'])}")
    return categorized_files


def select_files_by_numbers(file_list: Iterable[str], numbers: Iterable[int],
                            strict: bool = False) -> List[str]:
    """Pick files whose stem contains any of the requested image numbers.

    Parameters
    ----------
    strict : bool
        If True, raise ValueError when a number has no match.
        If False (default), print a warning and skip missing numbers.
    """

    selected: List[str] = []
    missing: List[int] = []
    for num in numbers:
        tokens = {str(num), f"{num:04d}", f"{num:05d}"}
        matches = [f for f in file_list if any(tok in Path(f).stem for tok in tokens)]
        if not matches:
            if strict:
                raise ValueError(f"No file found for image number {num} within provided list.")
            missing.append(num)
            continue
        selected.extend(sorted(matches))
    if missing:
        print(f"  [info] Skipped {len(missing)} numbers not in filtered file list "
              f"(first few: {missing[:10]})")
    return selected


def expand_image_numbers(entries: Iterable[str]) -> List[int]:
    """Expand CLI tokens like '5' or '1-5' into a sorted list of integers."""

    expanded: List[int] = []
    for entry in entries:
        if entry is None:
            continue
        if isinstance(entry, int):
            expanded.append(entry)
            continue
        token = str(entry).strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            step = 1 if end >= start else -1
            expanded.extend(list(range(start, end + step, step)))
        else:
            expanded.append(int(token))

    if not expanded:
        raise ValueError("No valid image numbers provided.")

    seen = set()
    deduped: List[int] = []
    for value in expanded:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def get_filter_from_header(
    filepath: str,
    filter_keywords: Tuple[str, ...] = ("FILTER", "FILTERS"),
) -> str:
    """Read the filter/band from a FITS primary header.

    Returns the uppercase single-letter band (e.g. ``'R'``) or an empty
    string when no filter keyword is found.
    """
    with fits.open(filepath, memmap=False) as hdul:
        hdr = hdul[0].header
        for key in filter_keywords:
            val = hdr.get(key)
            if val is not None and str(val).strip():
                return str(val).strip().upper()
    return ""


def diff_amp(
    amp_num: int,
    bias_files: List[str],
    dark_files: List[str],
    flat_files: List[str],
    sciences: List[str],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Load amplifier-specific data arrays for each calibration category."""

    biases = [0] * len(bias_files)
    darks = [0] * len(dark_files)
    flats = [0] * len(flat_files)
    sciences_data = [0] * len(sciences)

    for ii, path in enumerate(bias_files):
        with fits.open(path) as hdul:
            biases[ii] = hdul[amp_num].data - np.median(hdul[amp_num].data[:, 2040:2060])

    for jj, path in enumerate(dark_files):
        with fits.open(path) as hdul:
            darks[jj] = hdul[amp_num].data - np.median(hdul[amp_num].data[:, 2040:2060])

    for kk, path in enumerate(flat_files):
        with fits.open(path) as hdul:
            flats[kk] = hdul[amp_num].data - np.median(hdul[amp_num].data[:, 2040:2060])

    for mm, path in enumerate(sciences):
        with fits.open(path) as hdul:
            sciences_data[mm] = hdul[amp_num].data - np.median(hdul[amp_num].data[:, 2040:2060])

    return biases, darks, flats, sciences_data


def flat_reduction_b(
    biases: List[np.ndarray],
    darks: List[np.ndarray],
    flats: List[np.ndarray],
    plot_master_bias: bool = False,
    plot_unbias_dark: bool = False,
    plot_master_flat: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build master calibration frames using notebook logic."""

    master_bias = np.median(biases, axis=0)
    if plot_master_bias:
        plt.figure()
        plt.title("Master Bias")
        plt.imshow(master_bias, vmax=1200, origin="lower")

    if len(darks) > 0:
        master_dark = np.median(darks, axis=0)
        unbiased_dark = master_dark - master_bias
    else:
        # No dark frames provided: skip dark subtraction (zeros).
        master_dark = np.zeros_like(master_bias)
        unbiased_dark = np.zeros_like(master_bias)
    if plot_unbias_dark and len(darks) > 0:
        plt.figure()
        plt.title("Unbiased Dark")
        plt.imshow(unbiased_dark, vmax=1200, origin="lower")

    master_flat = np.median(flats, axis=0)
    if plot_master_flat:
        plt.figure()
        plt.title("Master Flat")
        plt.imshow(master_flat, vmax=2.5e4, origin="lower")

    return master_bias, master_dark, unbiased_dark, master_flat


def _find_image_hdu(hdul, ext: int) -> Optional[int]:
    if 0 <= ext < len(hdul) and getattr(hdul[ext], "data", None) is not None:
        return ext
    for idx, hdu in enumerate(hdul):
        if getattr(hdu, "data", None) is not None:
            return idx
    return None


def data_reduction(
    ext: int,
    master_bias: np.ndarray,
    master_flat: np.ndarray,
    sci_paths: List[str],
    plot_normalize_flat: bool = False,
    plot_final_sci: bool = False,
    write_output: bool = False,
    outdir: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Bias-subtract and flat-correct the supplied science files."""

    master_bias = np.asarray(master_bias, dtype=float)
    master_flat = np.asarray(master_flat, dtype=float)

    flat_minus_bias = master_flat - master_bias
    median_flat = np.median(flat_minus_bias)
    if median_flat == 0:
        raise ValueError("Median of (Master_flat - Master_bias) is zero; cannot normalize flat")
    normalized_flat = flat_minus_bias / median_flat

    if plot_normalize_flat:
        plt.figure()
        plt.title("Normalized flat")
        plt.imshow(normalized_flat, origin="lower")
        plt.colorbar()
        plt.close()

    last_reduced = None
    for sci_path in sci_paths:
        with fits.open(sci_path, memmap=False) as hdul:
            use_ext = _find_image_hdu(hdul, ext)
            if use_ext is None:
                raise ValueError(f"No image data found in {sci_path}")
            sci_data = hdul[use_ext].data.astype(float)
            hdr = hdul[use_ext].header.copy()

        if sci_data.shape != master_bias.shape or sci_data.shape != normalized_flat.shape:
            raise ValueError("Shape mismatch between science and master frames")

        final = (sci_data - master_bias) / normalized_flat
        last_reduced = final

        if plot_final_sci:
            plt.figure()
            plt.title(f"Reduced: {Path(sci_path).name} (ext={use_ext})")
            plt.imshow(final, origin="lower", vmax=np.mean(final) * 2)
            plt.colorbar()
            plt.close()

        if write_output:
            base = Path(sci_path).with_suffix("").name
            outname = f"{base}_amp{ext}_reduced.fits"
            outdir_path = Path(outdir) if outdir else Path(sci_path).parent
            outdir_path.mkdir(parents=True, exist_ok=True)
            outpath = outdir_path / outname
            fits.writeto(outpath, final, hdr, overwrite=True)

    return normalized_flat, last_reduced


def find_bad_columns(
    ccd_data: np.ndarray,
    saturation_threshold: float,
    black_column_threshold: float,
    sf: float,
    bf: float,
) -> List[int]:
    """Return column indices that are likely bad (too saturated or too dark)."""

    rows, cols = ccd_data.shape
    bad_columns: List[int] = []
    for col_idx in range(cols):
        column_data = ccd_data[:, col_idx]
        saturated_pixel_count = np.sum(column_data >= saturation_threshold)
        black_pixel_count = np.sum(column_data <= black_column_threshold)
        is_partially_saturated = saturated_pixel_count > rows * sf
        is_partially_black = black_pixel_count > rows * bf
        if is_partially_saturated or is_partially_black:
            bad_columns.append(col_idx)
    return bad_columns


def create_bad_pixel_map(shape: Tuple[int, int], bad_columns: List[int]) -> np.ndarray:
    """Build 2D binary mask (1 = bad) from a list of column indices."""

    bad_map = np.zeros(shape, dtype=int)
    for col_index in bad_columns:
        if 0 <= col_index < shape[1]:
            bad_map[:, col_index] = 1
    return bad_map


def apply_mask_from_map(ccd_data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply binary mask (value == 1) to CCD image, setting masked pixels to NaN."""

    if mask.shape != ccd_data.shape:
        raise ValueError("Mask shape does not match CCD data shape")
    masked = np.copy(ccd_data).astype(float)
    masked[mask == 1] = np.nan
    return masked


def mask_bad_columns(ccd_data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for `apply_mask_from_map`."""

    return apply_mask_from_map(ccd_data, mask)


def radial_profile(data: np.ndarray, center: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(int)
    r_max = int(r.max()) + 1
    radial_mean = np.array([data[r == i].mean() for i in range(r_max)])
    return np.arange(r_max), radial_mean


def gaussian(r, a, mu, sigma, c):
    return a * np.exp(-((r - mu) ** 2) / (2 * sigma ** 2)) + c


def sep_run_2(
    mask_data: np.ndarray,
    threshold: float,
    cutout_size: int = 15,
    minarea: int = 20,
    verbose: bool = False,
):
    """Run SEP extraction and compute radial FWHM per detection."""

    data = mask_data.astype(np.float32)
    mask = data > (np.median(data) + 5 * np.std(data))
    bkg = sep.Background(data, bw=64, bh=64, fw=3, fh=3, mask=mask)
    data_sub = data - bkg
    objects, segmap = sep.extract(
        data_sub,
        thresh=threshold,
        err=bkg.globalrms,
        segmentation_map=True,
        minarea=minarea,
    )

    if objects is None:
        if verbose:
            print("No objects returned by sep.extract")
        empty = np.array([])
        return data_sub, None, empty, empty, empty, empty, empty, []

    if verbose:
        print("Number of objects detected:", len(objects))

    a_arr = np.array(objects["a"], dtype=float)
    b_arr = np.array(objects["b"], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        fwhm_sep = 2.355 * np.sqrt(a_arr * b_arr)

    nobj = len(objects)
    fwhm_radial = np.full(nobj, np.nan, dtype=float)
    flux_in_fwhm = np.full(nobj, np.nan, dtype=float)
    flux_peak_ratio = np.full(nobj, np.nan, dtype=float)
    # Store peak-normalized cutouts for objects with valid FWHM fits
    cutouts: List[Optional[np.ndarray]] = [None] * nobj

    for idx in range(nobj):
        x_val = objects["x"][idx]
        y_val = objects["y"][idx]
        if not (np.isfinite(x_val) and np.isfinite(y_val)):
            continue
        x = int(np.round(x_val))
        y = int(np.round(y_val))
        ny, nx = data_sub.shape
        if (
            (y - cutout_size < 0)
            or (y + cutout_size >= ny)
            or (x - cutout_size < 0)
            or (x + cutout_size >= nx)
        ):
            continue

        cutout = data_sub[y - cutout_size : y + cutout_size, x - cutout_size : x + cutout_size]
        if not np.isfinite(cutout).all():
            cutout = np.nan_to_num(cutout, nan=0.0)
        rp_r, rp_flux = radial_profile(cutout, (cutout_size, cutout_size))
        mask_valid = np.isfinite(rp_flux) & np.isfinite(rp_r)
        if mask_valid.sum() < 4:
            continue
        rp_r_fit = rp_r[mask_valid]
        rp_flux_fit = rp_flux[mask_valid]

        try:
            baseline = np.median(rp_flux_fit[-5:]) if rp_flux_fit.size >= 5 else np.median(rp_flux_fit)
            amp0 = max(rp_flux_fit.max() - baseline, 0.0)
            popt, _ = curve_fit(
                gaussian,
                rp_r_fit,
                rp_flux_fit,
                p0=[amp0, 0.0, 2.0, baseline],
                maxfev=5000,
            )
            sigma = abs(popt[2])
            fwhm = 2.355 * sigma
            fwhm_radial[idx] = fwhm
        except Exception as exc:
            if verbose:
                print(f"Gaussian fit failed for obj {idx}: {exc}")
            continue

        yy, xx = np.indices(cutout.shape)
        rr = np.sqrt((xx - cutout_size) ** 2 + (yy - cutout_size) ** 2)
        if np.isfinite(fwhm_radial[idx]):
            mask_fwhm = rr <= fwhm_radial[idx]
            flux_in_fwhm[idx] = float(cutout[mask_fwhm].sum())
            peak_pix = float(np.nanmax(cutout)) if np.isfinite(cutout).any() else 0.0
            flux_peak_ratio[idx] = flux_in_fwhm[idx] / peak_pix if peak_pix != 0 else np.nan
            # Store peak-normalized cutout for stacking
            if peak_pix > 0:
                cutouts[idx] = cutout / peak_pix

    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.where(
            np.isfinite(a_arr) & (a_arr != 0),
            1.0 - (b_arr / a_arr),
            np.nan,
        ).astype(float)

    return data_sub, objects, fwhm_sep, fwhm_radial, flux_in_fwhm, flux_peak_ratio, e, cutouts


def _to_str_array(col) -> np.ndarray:
    arr = np.asarray(col)
    return np.array([x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in arr])


def _parse_file_idx_array(sid_all: np.ndarray) -> np.ndarray:
    """Extract the leading integer file index from subset_id strings.

    subset_id format is ``"{file_idx}_amp{amp}"`` (e.g. ``"3_amp2"``).
    Returns an int32 array of length len(sid_all) computed in one pass,
    replacing repeated np.char.startswith scans over the full catalog.
    """
    # Fast vectorized extraction: find the '_' position and parse everything before it
    find_us = np.char.find(sid_all, "_")
    result = np.empty(len(sid_all), dtype=np.int32)
    for i, (s, p) in enumerate(zip(sid_all, find_us)):
        result[i] = int(s[:p]) if p > 0 else 0
    return result


def compute_gmm_labels(tbl_cutt: Table) -> np.ndarray:
    fwhm_vals = np.array(tbl_cutt["FWHM"])
    flux_ratio_vals = np.array(tbl_cutt["flux_ratio"])
    flux_vals = np.array(tbl_cutt["flux"])
    mag_vals = -2.5 * np.log10(np.clip(flux_vals, 1e-12, None))
    e_vals = np.array(tbl_cutt["e"])

    Xt = np.column_stack([fwhm_vals, flux_ratio_vals, mag_vals, e_vals])
    # Drop rows with any NaN values
    valid = np.all(np.isfinite(Xt), axis=1)
    Xt_valid = Xt[valid]
    fwhm_valid = fwhm_vals[valid]
    if Xt_valid.shape[0] < 2:
        # Not enough valid data for GMM, return all ones (non-star)
        return np.ones_like(fwhm_vals, dtype=int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xt_valid)

    gmm = GaussianMixture(n_components=2, random_state=0)
    labelst_valid = gmm.fit_predict(X_scaled)

    unique_labels = np.unique(labelst_valid)
    medians = []
    for lbl in unique_labels:
        med = np.nanmedian(fwhm_valid[labelst_valid == lbl]) if np.any(labelst_valid == lbl) else np.inf
        medians.append((lbl, med))
    star_label = min(medians, key=lambda x: x[1])[0]
    # Map back to original indices: assign 0 for star, 1 for non-star, NaN rows as 1
    out = np.ones_like(fwhm_vals, dtype=int)
    out_idx = np.where(valid)[0]
    out[out_idx[labelst_valid == star_label]] = 0
    return out


def fit_parabola_vertex_form(x, y, p0=None):
    """Fit y = A (x - h)^2 + k and return vertex parameters."""

    def vertex_parabola(x_arr, A, h, k):
        return A * (x_arr - h) ** 2 + k

    if p0 is None:
        coeffs = np.polyfit(x, y, 2)
        a0, b0, c0 = coeffs
        A0 = a0
        h0 = -b0 / (2 * a0) if abs(a0) > 1e-12 else x[np.argmin(y)]
        k0 = a0 * h0 * h0 + b0 * h0 + c0
        p0 = [A0, h0, k0]

    popt, pcov = curve_fit(vertex_parabola, x, y, p0=p0)
    A, h, k = popt
    y_fit = vertex_parabola(x, *popt)
    resid = y - y_fit
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    x_min = float(h)
    y_min = float(k)
    sigma_x = None
    try:
        sigma = np.sqrt(np.diag(pcov))
        sigma_x = float(sigma[1])
    except Exception:
        sigma_x = None

    return {
        "A": float(A),
        "h": float(h),
        "k": float(k),
        "x_min": x_min,
        "y_min": y_min,
        "sigma_x_min": sigma_x,
        "R2": float(r2),
        "popt": popt,
        "pcov": pcov,
        "y_fit": y_fit,
        "residuals": resid,
    }


def plot_focus_curve(
    x: np.ndarray,
    y: np.ndarray,
    fit_dict: Dict,
    output_path: Path,
):
    # matplotlib >= 3.6 renamed seaborn styles to "seaborn-v0_8-*"
    for _style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        if _style in plt.style.available:
            plt.style.use(_style)
            break
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(7, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    ax1.scatter(x, y, color="tab:blue", label="Median FWHM")
    ax1.plot(x, fit_dict["y_fit"], color="tab:green", label="Parabola fit")
    if fit_dict.get("x_min") is not None:
        ax1.plot(fit_dict["x_min"], fit_dict["y_min"], "go", label="Focus minimum")
        ax1.axvline(fit_dict["x_min"], color="green", alpha=0.3, linestyle="--")
        if fit_dict.get("sigma_x_min"):
            ax1.errorbar(
                fit_dict["x_min"],
                fit_dict["y_min"],
                xerr=fit_dict["sigma_x_min"],
                fmt="none",
                ecolor="green",
                capsize=4,
                alpha=0.6,
            )
    ax1.set_ylabel("Median FWHM (pix)")
    ax1.set_title("Focus curve fit")
    ax1.legend()

    ax2.axhline(0, color="k", lw=0.8, alpha=0.6)
    ax2.plot(x, fit_dict["residuals"], "g.")
    ax2.set_xlabel("Focus position")
    ax2.set_ylabel("Residual")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tilt + Focus solver  (3-actuator independent adjustment)
# ---------------------------------------------------------------------------

# Default 90Prime actuator geometry: 120° apart at radius R from optical axis.
# Override with actual engineering values if available.
_DEFAULT_ACTUATOR_ANGLES = {"A": 0.0, "B": 2 * np.pi / 3, "C": 4 * np.pi / 3}
_DEFAULT_ACTUATOR_RADIUS = 1.0  # normalised; cancel out in relative corrections

# Approximate amplifier positions on the focal plane (normalised).
# Amps 1-4 on one CCD half, 5-8 on the other; arranged as 4×2 grid.
# Replace with precise values from the 90Prime optical model for production use.
_DEFAULT_AMP_POSITIONS = {
    1: (-0.75, -0.50),
    2: (-0.75,  0.50),
    3: (-0.25, -0.50),
    4: (-0.25,  0.50),
    5: ( 0.25, -0.50),
    6: ( 0.25,  0.50),
    7: ( 0.75, -0.50),
    8: ( 0.75,  0.50),
}


def _actuator_xy(
    angles: Dict[str, float] = None,
    radius: float = None,
) -> Dict[str, Tuple[float, float]]:
    """Return {name: (x, y)} for each actuator."""
    if angles is None:
        angles = _DEFAULT_ACTUATOR_ANGLES
    if radius is None:
        radius = _DEFAULT_ACTUATOR_RADIUS
    return {
        name: (radius * np.cos(theta), radius * np.sin(theta))
        for name, theta in angles.items()
    }


def solve_tilt_focus_global(
    frames: List[Dict],
    amp_positions: Dict[int, Tuple[float, float]] = None,
    actuator_angles: Dict[str, float] = None,
    actuator_radius: float = None,
) -> Dict:
    """
    Joint tilt + focus fit across MANY frames.

    Model
    -----
    Across all N frames the atmosphere/optics share

        FWHM²(x,y; f) = FWHM_0² + α · (z₀_f + a_f·x + b_f·y)²

    where (FWHM_0, α) are GLOBAL constants and (z₀_f, a_f, b_f)
    vary per frame.  This breaks the seeing ↔ piston degeneracy that
    affects the per-frame solver because seeing must be consistent
    across all frames while the LVDT-driven defocus changes between
    them.

    Parameters
    ----------
    frames : list of dicts, each with keys
        "fwhm_per_amp" : {amp: median_fwhm_pix}
        "current_lvdt" : {"A":..., "B":..., "C":...}   (used to
                         compute per-frame corrections)
        "image_number" : int (optional, for output bookkeeping)
        "frame"        : str filename (optional)

    Returns
    -------
    dict with global parameters and a list of per-frame results:
        seeing_floor   : float, pixels
        alpha          : float
        per_frame      : list of dicts, same shape as solve_tilt_focus
                         output, one entry per input frame.
        R2_global      : float (over all amps × frames)
    """
    if amp_positions is None:
        amp_positions = _DEFAULT_AMP_POSITIONS
    act_xy = _actuator_xy(actuator_angles, actuator_radius)

    # Build flat arrays of (amp_x, amp_y, fwhm², frame_idx)
    xs, ys, f2, fidx = [], [], [], []
    for fi, fr in enumerate(frames):
        for amp, fwhm in fr["fwhm_per_amp"].items():
            x, y = amp_positions[amp]
            xs.append(x); ys.append(y)
            f2.append(float(fwhm) ** 2)
            fidx.append(fi)
    xs = np.array(xs); ys = np.array(ys)
    f2 = np.array(f2); fidx = np.array(fidx, dtype=int)
    Nf = len(frames)

    # Parameter vector: [FWHM0², α, z0_0, a_0, b_0, z0_1, a_1, b_1, ...]
    def _unpack(p):
        fwhm0_sq = p[0]; alpha = p[1]
        z0 = p[2:2 + 3 * Nf:3]
        a  = p[3:2 + 3 * Nf:3]
        b  = p[4:2 + 3 * Nf:3]
        return fwhm0_sq, alpha, z0, a, b

    def _residuals(p):
        fwhm0_sq, alpha, z0, a, b = _unpack(p)
        dz = z0[fidx] + a[fidx] * xs + b[fidx] * ys
        return (fwhm0_sq + alpha * dz * dz) - f2

    # --- Initial guess ---
    fwhm0_sq_init = float(np.min(f2))
    alpha_init = 1.0
    p0 = [fwhm0_sq_init, alpha_init]
    for fi in range(Nf):
        m = fidx == fi
        excess = np.maximum(f2[m] - fwhm0_sq_init, 0.0)
        sign = np.sign(np.sqrt(excess) + 1e-6)
        seed = sign * np.sqrt(excess)
        G = np.column_stack([np.ones(m.sum()), xs[m], ys[m]])
        try:
            z0_i, a_i, b_i = np.linalg.lstsq(G, seed, rcond=None)[0]
        except Exception:
            z0_i, a_i, b_i = 0.1, 0.1, 0.1
        if abs(z0_i) + abs(a_i) + abs(b_i) < 1e-6:
            z0_i = 0.1
        p0 += [float(z0_i), float(a_i), float(b_i)]

    from scipy.optimize import least_squares as _lsq
    bounds_lo = [0.0, 0.0] + [-np.inf] * (3 * Nf)
    bounds_hi = [np.inf, np.inf] + [np.inf] * (3 * Nf)
    result = _lsq(_residuals, p0, bounds=(bounds_lo, bounds_hi))
    fwhm0_sq, alpha, z0_arr, a_arr, b_arr = _unpack(result.x)

    ss_res = float(np.sum(result.fun ** 2))
    ss_tot = float(np.sum((f2 - np.mean(f2)) ** 2))
    r2_global = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    seeing_floor = float(np.sqrt(max(fwhm0_sq, 0)))

    # Per-frame breakdown
    per_frame = []
    for fi, fr in enumerate(frames):
        z0 = float(z0_arr[fi]); a = float(a_arr[fi]); b = float(b_arr[fi])
        amps = sorted(fr["fwhm_per_amp"].keys())
        defocus_per_amp = {}; fwhm_model = {}
        for amp in amps:
            x, y = amp_positions[amp]
            dz = z0 + a * x + b * y
            defocus_per_amp[amp] = dz
            fwhm_model[amp] = float(np.sqrt(max(fwhm0_sq + alpha * dz * dz, 0)))
        cur = fr.get("current_lvdt", {"A": 0, "B": 0, "C": 0})
        corrections = {}; optimal_lvdt = {}
        for name, (ax, ay) in act_xy.items():
            d_at = z0 + a * ax + b * ay
            corrections[name] = float(-d_at)
            optimal_lvdt[name] = float(cur[name] - d_at)
        per_frame.append({
            "frame": fr.get("frame"),
            "image_number": fr.get("image_number"),
            "current_lvdt": cur,
            "optimal_lvdt": optimal_lvdt,
            "corrections": corrections,
            "piston_z0": z0,
            "tilt_a": a,
            "tilt_b": b,
            "tilt_magnitude": float(np.sqrt(a * a + b * b)),
            "defocus_per_amp": defocus_per_amp,
            "fwhm_model": fwhm_model,
            "seeing_floor": seeing_floor,
            "alpha": float(alpha),
        })
    return {
        "seeing_floor": seeing_floor,
        "alpha": float(alpha),
        "R2_global": float(r2_global),
        "per_frame": per_frame,
    }


def solve_tilt_focus(
    fwhm_per_amp: Dict[int, float],
    current_lvdt: Dict[str, float],
    amp_positions: Dict[int, Tuple[float, float]] = None,
    actuator_angles: Dict[str, float] = None,
    actuator_radius: float = None,
) -> Dict:
    """
    Solve for optimal independent actuator adjustments to remove both
    defocus (piston) and focal-plane tilt (tip-tilt).

    Model
    -----
    At position (x, y) on the focal plane the local defocus is

        δz(x, y) = z₀ + a·x + b·y          (plane)

    and FWHM responds quadratically:

        FWHM²(x, y) = FWHM₀² + α · δz(x, y)²

    We fit (FWHM₀², α, z₀, a, b) from per-amplifier median FWHM,
    then compute the correction at each actuator location so that
    δz = 0 everywhere (flat, in-focus plane).

    Parameters
    ----------
    fwhm_per_amp : dict  {amp_number: median_fwhm_pixels}
        At least 5 amps needed for 5 free parameters; 8 is ideal.
    current_lvdt : dict  {'A': float, 'B': float, 'C': float}
        Current actuator encoder readings.
    amp_positions : dict {amp: (x, y)} or None for default 90Prime layout.
    actuator_angles : dict {'A': θ_A, ...} in radians, or None for 120° default.
    actuator_radius : float or None.

    Returns
    -------
    dict with keys:
        seeing_floor  : float  – best-case FWHM (atmosphere only), pixels
        alpha         : float  – defocus sensitivity coefficient
        piston_z0     : float  – overall focus offset
        tilt_a        : float  – tip  (∂z/∂x)
        tilt_b        : float  – tilt (∂z/∂y)
        tilt_arcsec   : float  – total tilt magnitude in defocus units
        corrections   : dict {'A': Δ, 'B': Δ, 'C': Δ} – move each actuator by Δ
        optimal_lvdt  : dict {'A': val, ...} – target encoder readings
        defocus_per_amp : dict {amp: δz} – residual defocus at each amp (before correction)
        fwhm_model    : dict {amp: fwhm_predicted}
        R2            : float
    """
    if amp_positions is None:
        amp_positions = _DEFAULT_AMP_POSITIONS
    act_xy = _actuator_xy(actuator_angles, actuator_radius)

    amps = sorted(fwhm_per_amp.keys())
    if len(amps) < 5:
        raise ValueError(f"Need ≥ 5 amplifiers for the tilt+focus fit; got {len(amps)}")

    x_arr = np.array([amp_positions[a][0] for a in amps])
    y_arr = np.array([amp_positions[a][1] for a in amps])
    fwhm_arr = np.array([fwhm_per_amp[a] for a in amps])
    fwhm_sq = fwhm_arr ** 2

    # ---------- Least-squares fit ----------
    def _residuals(params):
        fwhm0_sq, alpha, z0, a, b = params
        defocus = z0 + a * x_arr + b * y_arr
        model_sq = fwhm0_sq + alpha * defocus ** 2
        return model_sq - fwhm_sq

    # Initial guess.  Avoid the degenerate point (z0=a=b=0): at that location
    # the Jacobian of the residual w.r.t. (z0, a, b) vanishes (∝ α·δz) and
    # Levenberg-Marquardt cannot escape, leaving tilt = 0 and R² = 0.
    # Seed z0/a/b from a linear plane fit to FWHM² so the optimiser starts in
    # a well-conditioned region.
    fwhm0_sq_init = float(np.min(fwhm_sq))
    excess = np.maximum(fwhm_sq - fwhm0_sq_init, 0.0)
    sign = np.sign(np.sqrt(excess) + 1e-6)  # arbitrary sign — symmetric in δz
    delta_seed = sign * np.sqrt(excess)
    G = np.column_stack([np.ones_like(x_arr), x_arr, y_arr])
    try:
        z0_init, a_init, b_init = np.linalg.lstsq(G, delta_seed, rcond=None)[0]
    except Exception:
        z0_init, a_init, b_init = 0.1, 0.1, 0.1
    if abs(z0_init) + abs(a_init) + abs(b_init) < 1e-6:
        z0_init = 0.1
    p0 = [fwhm0_sq_init, 1.0, float(z0_init), float(a_init), float(b_init)]

    from scipy.optimize import least_squares as _lsq
    result = _lsq(_residuals, p0, method="lm")
    fwhm0_sq, alpha, z0, a, b = result.x

    # Sign of (z0, a, b) is degenerate (only δz² appears).  Try the negated
    # initial guess too and keep whichever fit has lower residual.
    try:
        result_neg = _lsq(_residuals, [p0[0], p0[1], -p0[2], -p0[3], -p0[4]], method="lm")
        if np.sum(result_neg.fun ** 2) < np.sum(result.fun ** 2):
            result = result_neg
            fwhm0_sq, alpha, z0, a, b = result.x
    except Exception:
        pass

    # ---------- Corrections ----------
    corrections = {}
    optimal_lvdt = {}
    for name, (ax, ay) in act_xy.items():
        defocus_at_act = z0 + a * ax + b * ay
        corrections[name] = -defocus_at_act
        optimal_lvdt[name] = current_lvdt[name] - defocus_at_act

    # Per-amp diagnostics
    defocus_per_amp = {}
    fwhm_model = {}
    for amp, xv, yv in zip(amps, x_arr, y_arr):
        dz = z0 + a * xv + b * yv
        defocus_per_amp[amp] = float(dz)
        fwhm_model[amp] = float(np.sqrt(max(fwhm0_sq + alpha * dz ** 2, 0)))

    # R²
    ss_res = np.sum(result.fun ** 2)
    ss_tot = np.sum((fwhm_sq - np.mean(fwhm_sq)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    tilt_mag = np.sqrt(a ** 2 + b ** 2)

    return {
        "seeing_floor": float(np.sqrt(max(fwhm0_sq, 0))),
        "alpha": float(alpha),
        "piston_z0": float(z0),
        "tilt_a": float(a),
        "tilt_b": float(b),
        "tilt_magnitude": float(tilt_mag),
        "corrections": corrections,
        "optimal_lvdt": optimal_lvdt,
        "defocus_per_amp": defocus_per_amp,
        "fwhm_model": fwhm_model,
        "R2": float(r2),
    }


def actuator_direction_summary(
    corrections: Dict[str, float],
    current_lvdt: Optional[Dict[str, float]] = None,
    deadband: float = 0.01,
) -> Dict[str, Dict[str, float]]:
    """
    Convert actuator correction values into human-readable move directions.

    ``corrections`` are absolute model corrections and can include piston.
    ``tilt_delta`` subtracts their mean, which is the safer quantity when the
    overall focus piston is handled by the focus-curve best focus.
    """
    names = list(corrections.keys())
    raw = np.array([float(corrections[n]) for n in names], dtype=float)
    piston = float(np.nanmean(raw)) if raw.size else 0.0
    out: Dict[str, Dict[str, float]] = {}
    for name, delta in zip(names, raw):
        tilt_delta = float(delta - piston)
        if abs(tilt_delta) < deadband:
            direction = "hold"
        elif tilt_delta > 0:
            direction = "increase LVDT"
        else:
            direction = "decrease LVDT"
        current = None
        target = None
        if current_lvdt is not None and name in current_lvdt:
            current = float(current_lvdt[name])
            target = current + tilt_delta
        out[name] = {
            "raw_delta": float(delta),
            "tilt_delta": tilt_delta,
            "direction": direction,
            "current_lvdt": current,
            "tilt_only_target_lvdt": target,
        }
    return out


def plot_tilt_map(
    fwhm_per_amp: Dict[int, float],
    tilt_result: Dict,
    output_path: Path,
    amp_positions: Dict[int, Tuple[float, float]] = None,
    pixscale: float = 0.455,
):
    """
    Visualise the focal-plane tilt: 2D map of per-amp FWHM with the
    fitted tilt plane overlaid, plus a bar chart of actuator corrections.
    """
    if amp_positions is None:
        amp_positions = _DEFAULT_AMP_POSITIONS

    amps = sorted(fwhm_per_amp.keys())
    x = np.array([amp_positions[a][0] for a in amps])
    y = np.array([amp_positions[a][1] for a in amps])
    fwhm = np.array([fwhm_per_amp[a] for a in amps]) * pixscale  # arcsec
    fwhm_mod = np.array([tilt_result["fwhm_model"][a] for a in amps]) * pixscale

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    # --- Panel 1: Measured FWHM map ---
    sc = axes[0].scatter(x, y, c=fwhm, s=900, cmap="RdYlGn_r",
                         edgecolor="k", vmin=fwhm.min() - 0.05,
                         vmax=fwhm.max() + 0.05, zorder=5)
    axes[0].set_title("Measured FWHM (arcsec)", fontsize=14)
    axes[0].set_xlabel("Focal plane X", fontsize=13)
    axes[0].set_ylabel("Focal plane Y", fontsize=13)
    axes[0].tick_params(axis="both", labelsize=12)
    _xmin, _xmax = float(x.min()), float(x.max())
    _ymin, _ymax = float(y.min()), float(y.max())
    _xpad = max(0.4 * (_xmax - _xmin), 0.5)
    _ypad = max(0.8 * (_ymax - _ymin), 0.8)
    axes[0].set_xlim(_xmin - _xpad, _xmax + _xpad)
    axes[0].set_ylim(_ymin - _ypad, _ymax + _ypad)
    plt.colorbar(sc, ax=axes[0], label='FWHM (")')

    # --- Panel 2: Defocus map (tilt plane) ---
    defocus = np.array([tilt_result["defocus_per_amp"][a] for a in amps])
    sc2 = axes[1].scatter(x, y, c=defocus, s=900, cmap="coolwarm",
                          edgecolor="k", zorder=5)
    axes[1].set_title(f"Defocus δz  (tilt = {tilt_result['tilt_magnitude']:.4f})", fontsize=14)
    axes[1].set_xlabel("Focal plane X", fontsize=13)
    axes[1].tick_params(axis="both", labelsize=12)
    axes[1].set_xlim(_xmin - _xpad, _xmax + _xpad)
    axes[1].set_ylim(_ymin - _ypad, _ymax + _ypad)
    plt.colorbar(sc2, ax=axes[1], label="δz (defocus)")

    # --- Panel 3: Actuator corrections ---
    names = list(tilt_result["corrections"].keys())
    raw_deltas = np.array([tilt_result["corrections"][n] for n in names], dtype=float)
    piston_delta = float(np.nanmean(raw_deltas)) if raw_deltas.size else 0.0
    deltas = raw_deltas - piston_delta
    colors = ["#4682B4" if d >= 0 else "#C44E52" for d in deltas]
    axes[2].bar(names, deltas, color=colors, edgecolor="k", width=0.5)
    for i, (n, d) in enumerate(zip(names, deltas)):
        if abs(d) < 0.01:
            direction = "HOLD"
        elif d > 0:
            direction = "INC LVDT"
        else:
            direction = "DEC LVDT"
        axes[2].text(
            i, d, f"{direction}\n{d:+.2f}", ha="center",
            va="bottom" if d >= 0 else "top", fontsize=11, fontweight="bold",
        )
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set_ylabel("Tilt-only correction (LVDT units)", fontsize=13)
    axes[2].set_title("Actuator Tilt Directions\n(mean piston removed)", fontsize=14)
    axes[2].tick_params(axis="both", labelsize=12)
    for n in names:
        opt = tilt_result["optimal_lvdt"][n]
        raw = tilt_result["corrections"][n]
        axes[2].text(list(names).index(n), 0, f"\nraw {raw:+.2f}\n-> {opt:.1f}",
                     ha="center", va="top", fontsize=9, color="gray")
    axes[2].text(
        0.5, 0.03,
        "Use INC/DEC as relative tilt guidance.\nUse the focus curve for common piston.",
        transform=axes[2].transAxes,
        ha="center", va="bottom", fontsize=9, color="dimgray",
    )

    fig.suptitle(
        f"Tilt + Focus Solution   |   Seeing floor = {tilt_result['seeing_floor'] * pixscale:.2f}\"   "
        f"|   R² = {tilt_result['R2']:.3f}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Higher-level orchestration helpers
# ---------------------------------------------------------------------------
def load_bad_mask(mask_dir: Path, filter_band: str, amp: int, shape: Tuple[int, int]) -> np.ndarray:
    mask_path = mask_dir / f"bad_pixel_mask_amp_{filter_band.upper()}{amp}.npy"
    if not mask_path.exists():
        warnings.warn(f"Missing mask {mask_path}; proceeding without masking.")
        return np.zeros(shape, dtype=int)
    mask = np.load(mask_path)
    if mask.shape != shape:
        if mask.ndim == 1:
            bad_cols = [int(v) for v in mask]
            mask = create_bad_pixel_map(shape, bad_cols)
        else:
            raise ValueError(f"Mask {mask_path} has shape {mask.shape}, expected {shape}")
    return mask


# ---------------------------------------------------------------------------
# Incremental-mode caching helpers
# ---------------------------------------------------------------------------

def _get_cache_dir(outdir: Path) -> Path:
    """Return (and create) the cache directory for incremental processing."""
    d = outdir / ".cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _validate_cache_params(
    cache_dir: Path, threshold: float, amps: List[int],
) -> None:
    """Wipe stale per-file caches when detection parameters change."""
    meta_path = cache_dir / "_params.json"
    current = json.dumps({"threshold": threshold, "amps": sorted(amps)}, sort_keys=True)
    if meta_path.exists():
        if meta_path.read_text().strip() == current:
            return
        print("[incremental] Detection parameters changed \u2014 clearing cached detections")
        for p in cache_dir.glob("*_cat.fits"):
            p.unlink()
        for p in cache_dir.glob("*_cutouts.npz"):
            p.unlink()
    meta_path.write_text(current)


def _save_master_calibrations(
    cache_dir: Path, amp: int, cal_tuple: Tuple,
    band: str = "",
) -> None:
    """Persist master bias and master flat for one amplifier.

    The bias file is band-independent; the flat file includes the band
    name so that multi-band runs store separate flats.
    """
    master_bias, _, _, master_flat = cal_tuple
    np.save(cache_dir / f"master_bias_amp{amp}.npy", master_bias)
    flat_tag = f"_{band}" if band else ""
    np.save(cache_dir / f"master_flat_amp{amp}{flat_tag}.npy", master_flat)


def _load_master_calibrations(
    cache_dir: Path, amp: int,
    band: str = "",
) -> Optional[Tuple]:
    """Load cached master bias/flat.  Returns *None* on cache miss."""
    bp = cache_dir / f"master_bias_amp{amp}.npy"
    flat_tag = f"_{band}" if band else ""
    fp = cache_dir / f"master_flat_amp{amp}{flat_tag}.npy"
    if not (bp.exists() and fp.exists()):
        return None
    mb = np.load(bp)
    mf = np.load(fp)
    # data_reduction() only uses master_bias ([0]) and master_flat ([3]).
    return (mb, np.zeros_like(mb), np.zeros_like(mb), mf)


def _file_cache_stem(sci_path: str) -> str:
    """Deterministic cache key derived from the science filename."""
    return Path(sci_path).stem


def _save_file_detections(
    cache_dir: Path,
    sci_path: str,
    tab: Table,
    cutouts: List[Optional[np.ndarray]],
) -> None:
    """Write per-file detection table and cutouts to the cache directory."""
    stem = _file_cache_stem(sci_path)
    tab.write(cache_dir / f"{stem}_cat.fits", overwrite=True)
    valid = np.array([c is not None for c in cutouts], dtype=bool)
    if valid.any():
        ref = next(c for c in cutouts if c is not None)
        arr = np.full((len(cutouts), *ref.shape), np.nan, dtype=np.float64)
        for i, c in enumerate(cutouts):
            if c is not None:
                arr[i] = c
        np.savez_compressed(
            cache_dir / f"{stem}_cutouts.npz", data=arr, valid=valid,
        )
    else:
        np.savez_compressed(
            cache_dir / f"{stem}_cutouts.npz",
            data=np.array([]), valid=valid,
        )


def _load_file_detections(
    cache_dir: Path, sci_path: str,
) -> Optional[Tuple[Table, List[Optional[np.ndarray]]]]:
    """Load cached detections.  Returns *None* on cache miss."""
    stem = _file_cache_stem(sci_path)
    cat_p = cache_dir / f"{stem}_cat.fits"
    cut_p = cache_dir / f"{stem}_cutouts.npz"
    if not (cat_p.exists() and cut_p.exists()):
        return None
    tab = Table.read(cat_p)
    if "_cache_amp" not in tab.colnames:
        return None  # incompatible old cache; reprocess
    npz = np.load(cut_p, allow_pickle=False)
    valid = npz["valid"]
    data = npz["data"]
    cutouts: List[Optional[np.ndarray]] = []
    for i in range(len(valid)):
        if valid[i] and data.size > 0:
            cutouts.append(data[i])
        else:
            cutouts.append(None)
    return tab, cutouts


def _detect_single_file(
    sci_path: str,
    file_idx: int,
    amps: List[int],
    master_cache: Dict[int, Tuple],
    filter_band: str,
    mask_dir: Path,
    threshold: float,
    outdir: Path,
    write_reduced: bool,
) -> Tuple[Optional[Table], List[Optional[np.ndarray]]]:
    """Run SEP detection on one science file across all requested amps."""
    all_tables: List[Table] = []
    all_fwhm: List[np.ndarray] = []
    all_e: List[np.ndarray] = []
    all_flux_ratio: List[np.ndarray] = []
    all_cutouts: List[List[Optional[np.ndarray]]] = []
    subset_labels: List[np.ndarray] = []
    amp_nums: List[np.ndarray] = []

    for amp in amps:
        try:
            master_bias, _, _, master_flat = master_cache[amp]
            _, reduced = data_reduction(
                amp, master_bias, master_flat, [sci_path],
                write_output=write_reduced, outdir=str(outdir),
            )
            if reduced is None:
                raise RuntimeError("Reduction returned no data")

            mask = load_bad_mask(mask_dir, filter_band, amp, reduced.shape)
            masked_d = mask_bad_columns(reduced, mask)

            data_sub, objects, _, fwhm_radial, _, flux_peak_ratio, e, cutouts = sep_run_2(
                masked_d, threshold,
            )
            if objects is None or len(objects) == 0:
                continue

            tab = Table(objects)
            all_tables.append(tab)
            all_fwhm.append(np.array(fwhm_radial))
            all_e.append(np.array(e))
            all_flux_ratio.append(np.array(flux_peak_ratio))
            all_cutouts.append(cutouts)
            subset_labels.append(np.repeat(f"{file_idx}_amp{amp}", len(tab)))
            amp_nums.append(np.repeat(amp, len(tab)))
        except Exception as exc:
            print(f"  Skipping {Path(sci_path).name} amp{amp}: {exc}")
            continue

    if not all_tables:
        return None, []

    stacked = vstack(all_tables, join_type="exact")
    stacked["subset_id"] = np.concatenate(subset_labels)
    stacked["FWHM"] = np.concatenate(all_fwhm)
    stacked["e"] = np.concatenate(all_e)
    stacked["flux_ratio"] = np.concatenate(all_flux_ratio)
    stacked["_cache_amp"] = np.concatenate(amp_nums)

    cutouts_flat: List[Optional[np.ndarray]] = []
    for cl in all_cutouts:
        cutouts_flat.extend(cl)

    return stacked, cutouts_flat


def run_detection_loop_incremental(
    scifiles: List[str],
    amps: List[int],
    master_cache: Dict[int, Tuple],
    filter_band: str,
    mask_dir: Path,
    threshold: float,
    outdir: Path,
    write_reduced: bool,
    cache_dir: Path,
) -> Tuple[Table, List[str], List[Optional[np.ndarray]]]:
    """Incremental version of *run_detection_loop*.

    Cached files are loaded from disk; only new files are reduced and
    extracted.  Newly processed results are saved to the cache.
    """
    all_tables: List[Table] = []
    all_cutouts: List[Optional[np.ndarray]] = []
    n_cached = 0
    n_processed = 0

    for file_idx, sci_path in enumerate(scifiles):
        cached = _load_file_detections(cache_dir, sci_path)
        if cached is not None:
            tab, cutouts = cached
            # Rewrite subset_id to match current file position
            amp_arr = np.array(tab["_cache_amp"], dtype=int)
            tab["subset_id"] = np.array(
                [f"{file_idx}_amp{a}" for a in amp_arr]
            )
            all_tables.append(tab)
            all_cutouts.extend(cutouts)
            n_cached += 1
            continue

        # Cache miss — process this file
        print(f"  Processing {Path(sci_path).name} "
              f"({file_idx + 1}/{len(scifiles)}) ...")
        tab, cutouts = _detect_single_file(
            sci_path, file_idx, amps, master_cache,
            filter_band, mask_dir, threshold, outdir, write_reduced,
        )
        if tab is not None:
            _save_file_detections(cache_dir, sci_path, tab, cutouts)
            all_tables.append(tab)
            all_cutouts.extend(cutouts)
            n_processed += 1
        else:
            print(f"  No detections in {Path(sci_path).name}")

    print(f"[incremental] {n_cached} cached + {n_processed} newly processed")

    if not all_tables:
        raise RuntimeError("No detections found; cannot proceed with focus fit")

    stacked = vstack(all_tables, join_type="outer")
    # Drop internal bookkeeping column
    if "_cache_amp" in stacked.colnames:
        stacked.remove_column("_cache_amp")

    return stacked, [Path(p).name for p in scifiles], all_cutouts


def run_detection_loop(
    scifiles: List[str],
    amps: Iterable[int],
    master_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    filter_band: str,
    mask_dir: Path,
    threshold: float,
    outdir: Path,
    write_reduced: bool,
) -> Tuple[Table, List[str]]:
    all_tables: List[Table] = []
    all_fwhm: List[np.ndarray] = []
    all_e: List[np.ndarray] = []
    all_flux_ratio: List[np.ndarray] = []
    all_cutouts: List[List[Optional[np.ndarray]]] = []
    subset_labels: List[np.ndarray] = []

    for file_idx, sci_path in enumerate(scifiles):
        base_noext = Path(sci_path).with_suffix("").name
        for amp in amps:
            try:
                master_bias, _, _, master_flat = master_cache[amp]
                _, reduced = data_reduction(
                    amp,
                    master_bias,
                    master_flat,
                    [sci_path],
                    write_output=write_reduced,
                    outdir=str(outdir),
                )
                if reduced is None:
                    raise RuntimeError("Reduction returned no data; cannot continue")
                image_data = reduced

                mask = load_bad_mask(mask_dir, filter_band, amp, image_data.shape)
                masked_d = mask_bad_columns(image_data, mask)

                data_sub, objects, _, fwhm_radial, _, flux_peak_ratio, e, cutouts = sep_run_2(
                    masked_d,
                    threshold,
                )

                if objects is None or len(objects) == 0:
                    continue

                tab = Table(objects)
                all_tables.append(tab)
                all_fwhm.append(np.array(fwhm_radial))
                all_e.append(np.array(e))
                all_flux_ratio.append(np.array(flux_peak_ratio))
                all_cutouts.append(cutouts)
                subset_labels.append(np.repeat(f"{file_idx}_amp{amp}", len(tab)))
            except Exception as exc:
                print(f"Skipping {sci_path} amp{amp} due to error: {exc}")
                continue

    if not all_tables:
        raise RuntimeError("No detections found; cannot proceed with focus fit")

    stacked = vstack(all_tables, join_type="exact")
    labels_flat = np.concatenate(subset_labels)
    fwhm_flat = np.concatenate(all_fwhm)
    e_flat = np.concatenate(all_e)
    flux_ratio_flat = np.concatenate(all_flux_ratio)
    cutouts_flat: List[Optional[np.ndarray]] = []
    for cl in all_cutouts:
        cutouts_flat.extend(cl)

    stacked["subset_id"] = labels_flat
    stacked["FWHM"] = fwhm_flat
    stacked["e"] = e_flat
    stacked["flux_ratio"] = flux_ratio_flat

    return stacked, [Path(path).name for path in scifiles], cutouts_flat


def aggregate_per_file_metrics(
    stacked: Table,
    n_files: int,
    n_amps: int,
    metric_names: Tuple[str, ...],
    quality_cuts: Optional[Dict[str, Tuple[str, float]]] = None,
) -> Dict[str, List[float]]:
    if not metric_names:
        raise ValueError("metric_names cannot be empty")
    if quality_cuts is None:
        quality_cuts = {"FWHM": ("<", 15), "flag": ("==", 0), "flux_ratio": (">", 1)}

    sid_all = _to_str_array(stacked["subset_id"])
    file_idx_arr = _parse_file_idx_array(sid_all)
    results: Dict[str, List[float]] = {name: [] for name in metric_names}

    for subset_idx in range(n_files):
        mask_subset = file_idx_arr == subset_idx
        if not np.any(mask_subset):
            for name in metric_names:
                results[name].append(np.nan)
            continue

        tbl_subset = stacked[mask_subset]
        mask = np.ones(len(tbl_subset), dtype=bool)
        for col, (op, val) in quality_cuts.items():
            col_arr = np.array(tbl_subset[col])
            if op == "<":
                mask &= col_arr < val
            elif op == "<=":
                mask &= col_arr <= val
            elif op == ">":
                mask &= col_arr > val
            elif op == ">=":
                mask &= col_arr >= val
            elif op == "==":
                mask &= col_arr == val
            elif op == "!=":
                mask &= col_arr != val
            else:
                raise ValueError(f"Unsupported op {op}")

        tbl_cutt = tbl_subset[mask]
        if len(tbl_cutt) == 0:
            for name in metric_names:
                results[name].append(np.nan)
            continue

        labelst = compute_gmm_labels(tbl_cutt)
        sid_cutt = _to_str_array(tbl_cutt["subset_id"])
        metric_arrays = {name: np.array(tbl_cutt[name], dtype=float) for name in metric_names}
        per_metric_values: Dict[str, List[float]] = {name: [] for name in metric_names}

        for amp in range(1, n_amps + 1):
            amp_prefix = f"{subset_idx}_amp{amp}"
            mask_amp = np.char.startswith(sid_cutt, amp_prefix)
            mask_sel = (labelst == 0) & mask_amp
            if not np.any(mask_sel):
                continue
            for name in metric_names:
                vals = metric_arrays[name][mask_sel]
                if vals.size:
                    per_metric_values[name].append(float(np.nanmedian(vals)))

        for name in metric_names:
            if per_metric_values[name]:
                results[name].append(float(np.nanmedian(per_metric_values[name])))
            else:
                results[name].append(np.nan)

    return results


def generate_star_stack_images(
    stacked: Table,
    cutouts_flat: List[Optional[np.ndarray]],
    n_files: int,
    n_amps: int,
    quality_cuts: Optional[Dict[str, Tuple[str, float]]] = None,
    pixscale: Optional[float] = None,
    skip_indices: Optional[set] = None,
) -> List[Optional[str]]:
    """Create average-stacked star thumbnails per exposure.

    Uses the same GMM star selection as the FWHM metrics.  For each
    exposure, collects the peak-normalised cutouts of selected stars,
    median-stacks them, and renders the result as a base64-encoded PNG
    string suitable for embedding in HTML hover tooltips.

    Returns a list (one entry per science file) of base64 PNG strings,
    or ``None`` for exposures where no valid cutouts are available.
    """
    if quality_cuts is None:
        quality_cuts = {"FWHM": ("<", 15), "flag": ("==", 0), "flux_ratio": (">", 1)}

    sid_all = _to_str_array(stacked["subset_id"])
    file_idx_arr = _parse_file_idx_array(sid_all)
    # Map each row in stacked to its index so we can look up cutouts
    global_indices = np.arange(len(stacked))
    result: List[Optional[str]] = []

    for subset_idx in range(n_files):
        if skip_indices and subset_idx in skip_indices:
            result.append(None)  # will be restored from cache in main()
            continue
        mask_subset = file_idx_arr == subset_idx
        if not np.any(mask_subset):
            result.append(None)
            continue

        subset_global_idx = global_indices[mask_subset]
        tbl_subset = stacked[mask_subset]
        mask = np.ones(len(tbl_subset), dtype=bool)
        for col, (op, val) in quality_cuts.items():
            col_arr = np.array(tbl_subset[col])
            if op == "<":
                mask &= col_arr < val
            elif op == "<=":
                mask &= col_arr <= val
            elif op == ">":
                mask &= col_arr > val
            elif op == ">=":
                mask &= col_arr >= val
            elif op == "==":
                mask &= col_arr == val
            elif op == "!=":
                mask &= col_arr != val

        tbl_cutt = tbl_subset[mask]
        cutt_global_idx = subset_global_idx[mask]
        if len(tbl_cutt) == 0:
            result.append(None)
            continue

        labelst = compute_gmm_labels(tbl_cutt)
        sid_cutt = _to_str_array(tbl_cutt["subset_id"])

        # Collect cutouts for GMM-selected stars across all amps
        selected_cutouts: List[np.ndarray] = []
        for amp in range(1, n_amps + 1):
            amp_prefix = f"{subset_idx}_amp{amp}"
            mask_amp = np.char.startswith(sid_cutt, amp_prefix)
            mask_sel = (labelst == 0) & mask_amp
            for local_i in np.where(mask_sel)[0]:
                gi = cutt_global_idx[local_i]
                if gi < len(cutouts_flat) and cutouts_flat[gi] is not None:
                    selected_cutouts.append(cutouts_flat[gi])

        if not selected_cutouts:
            result.append(None)
            continue

        # Median-stack: use the most common cutout shape
        ref_shape = selected_cutouts[0].shape
        same_shape = [c for c in selected_cutouts if c.shape == ref_shape]
        if not same_shape:
            result.append(None)
            continue

        stack = np.nanmedian(same_shape, axis=0)

        # Render as contour plot thumbnail
        b64 = _render_psf_contour_thumbnail(stack, n_stars=len(same_shape), pixscale=pixscale)
        result.append(b64)

    return result


def _render_psf_contour_thumbnail(
    psf_image: np.ndarray,
    n_stars: Optional[int] = None,
    pixscale: Optional[float] = None,
) -> str:
    """Render a 2-D PSF array as a contour-over-image thumbnail.

    Parameters
    ----------
    psf_image : 2-D array
    n_stars : optional count shown in the title
    pixscale : arcsec / pixel (if given, a scale bar is drawn)

    Returns a base64-encoded PNG string.
    """
    ny, nx = psf_image.shape
    cx, cy = nx / 2.0, ny / 2.0
    # Pixel-offset coordinates centred on (0, 0)
    extent = [-cx, nx - cx, -cy, ny - cy]

    fig_t, ax_t = plt.subplots(figsize=(2.4, 2.4))
    vmin = np.nanpercentile(psf_image, 1)
    vmax = np.nanpercentile(psf_image, 99.5)
    ax_t.imshow(
        psf_image,
        origin="lower",
        cmap="gray_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="bicubic",
        aspect="equal",
        extent=extent,
    )
    # Overlay contours at percentage-of-peak levels
    peak = np.nanmax(psf_image)
    if peak > 0:
        levels = peak * np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        y_coords = np.linspace(extent[2], extent[3], ny)
        x_coords = np.linspace(extent[0], extent[1], nx)
        ax_t.contour(
            x_coords,
            y_coords,
            psf_image,
            levels=levels,
            colors=["#2196F3", "#03A9F4", "#00BCD4", "#FF9800", "#F44336"],
            linewidths=1.0,
        )

    ax_t.set_xlabel("pixels", fontsize=7)
    ax_t.set_ylabel("pixels", fontsize=7)
    ax_t.tick_params(labelsize=6)

    # Draw an arcsec scale bar if a plate scale is provided
    if pixscale is not None and pixscale > 0:
        bar_arcsec = 1.0  # 1" bar; bump to 2" if that would be too small
        bar_px = bar_arcsec / pixscale
        if bar_px < 1.5:
            bar_arcsec = 2.0
            bar_px = bar_arcsec / pixscale
        # Place in lower-right corner
        x0 = extent[1] - bar_px - 1
        y0 = extent[2] + 1
        ax_t.plot([x0, x0 + bar_px], [y0, y0], color="white", linewidth=2.5)
        ax_t.plot([x0, x0 + bar_px], [y0, y0], color="red", linewidth=1.5)
        ax_t.text(
            x0 + bar_px / 2,
            y0 + 0.8,
            f'{bar_arcsec:.0f}"',
            color="red",
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="bottom",
        )

    title = "Stacked PSF contour"
    if n_stars is not None:
        title += f" (N={n_stars})"
    ax_t.set_title(title, fontsize=7, pad=2)
    fig_t.tight_layout(pad=0.4)

    buf = BytesIO()
    fig_t.savefig(buf, format="png", dpi=72, bbox_inches="tight")
    plt.close(fig_t)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _synthetic_gaussian_psf(
    a: float, b: float, theta: float, size: int = 31
) -> np.ndarray:
    """Create a synthetic 2-D Gaussian PSF from SEP shape parameters."""
    cy, cx = size // 2, size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    dx = xx - cx
    dy = yy - cy
    x_rot = cos_t * dx + sin_t * dy
    y_rot = -sin_t * dx + cos_t * dy
    sigma_a = max(a, 0.5)
    sigma_b = max(b, 0.5)
    psf = np.exp(-0.5 * ((x_rot / sigma_a) ** 2 + (y_rot / sigma_b) ** 2))
    return psf


def generate_psf_contours_from_catalog(
    stacked: Table,
    n_files: int,
    n_amps: int,
    quality_cuts: Optional[Dict[str, Tuple[str, float]]] = None,
    pixscale: Optional[float] = None,
    skip_indices: Optional[set] = None,
) -> List[Optional[str]]:
    """Generate PSF contour thumbnails from catalog shape parameters.

    This is a fallback for when real pixel cutouts are not available.
    It synthesises a 2-D Gaussian from the median ``a``, ``b``, ``theta``
    of the GMM-selected stars in each exposure and renders a contour plot.
    """
    if quality_cuts is None:
        quality_cuts = {"FWHM": ("<", 15), "flag": ("==", 0), "flux_ratio": (">", 1)}

    sid_all = _to_str_array(stacked["subset_id"])
    file_idx_arr = _parse_file_idx_array(sid_all)
    result: List[Optional[str]] = []

    for subset_idx in range(n_files):
        if skip_indices and subset_idx in skip_indices:
            result.append(None)  # will be restored from cache in main()
            continue
        mask_subset = file_idx_arr == subset_idx
        if not np.any(mask_subset):
            result.append(None)
            continue

        tbl_subset = stacked[mask_subset]
        mask = np.ones(len(tbl_subset), dtype=bool)
        for col, (op, val) in quality_cuts.items():
            col_arr = np.array(tbl_subset[col])
            if op == "<":
                mask &= col_arr < val
            elif op == "<=":
                mask &= col_arr <= val
            elif op == ">":
                mask &= col_arr > val
            elif op == ">=":
                mask &= col_arr >= val
            elif op == "==":
                mask &= col_arr == val
            elif op == "!=":
                mask &= col_arr != val

        tbl_cutt = tbl_subset[mask]
        if len(tbl_cutt) == 0:
            result.append(None)
            continue

        labelst = compute_gmm_labels(tbl_cutt)
        sid_cutt = _to_str_array(tbl_cutt["subset_id"])

        # Collect shape params for GMM-selected stars
        sel_a: List[float] = []
        sel_b: List[float] = []
        sel_theta: List[float] = []
        for amp in range(1, n_amps + 1):
            amp_prefix = f"{subset_idx}_amp{amp}"
            mask_amp = np.char.startswith(sid_cutt, amp_prefix)
            mask_sel = (labelst == 0) & mask_amp
            if np.any(mask_sel):
                sel_a.extend(np.array(tbl_cutt["a"][mask_sel], dtype=float))
                sel_b.extend(np.array(tbl_cutt["b"][mask_sel], dtype=float))
                sel_theta.extend(np.array(tbl_cutt["theta"][mask_sel], dtype=float))

        if not sel_a:
            result.append(None)
            continue

        med_a = float(np.nanmedian(sel_a))
        med_b = float(np.nanmedian(sel_b))
        med_theta = float(np.nanmedian(sel_theta))
        n_stars = len(sel_a)

        psf = _synthetic_gaussian_psf(med_a, med_b, med_theta, size=31)
        b64 = _render_psf_contour_thumbnail(psf, n_stars=n_stars, pixscale=pixscale)
        result.append(b64)

    return result


def per_file_median_fwhm(
    stacked: Table,
    n_files: int,
    n_amps: int,
    quality_cuts: Optional[Dict[str, Tuple[str, float]]] = None,
) -> List[float]:
    metrics = aggregate_per_file_metrics(
        stacked,
        n_files,
        n_amps,
        ("FWHM",),
        quality_cuts=quality_cuts,
    )
    return metrics["FWHM"]


def _normalize_obs_time(value, date_str: Optional[str] = None) -> Tuple[Optional[str], float]:
    """Convert a header time value to (ISO string, MJD float).

    If *value* is a time-only string (e.g. ``'01:09:52.538'``) and
    *date_str* is supplied (e.g. ``'2025-11-12'``), the two are combined
    into a full ISO timestamp before parsing.
    """
    if value is None:
        return None, np.nan
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if np.isfinite(value):
            try:
                t = Time(value, format="mjd", scale="utc")
                return t.isot, float(t.mjd)
            except Exception:
                return None, float(value)
        return None, np.nan

    value_str = str(value).strip()
    if not value_str:
        return None, np.nan

    # If value looks like a bare time (no date component) and we have a date,
    # combine them into an ISO timestamp.
    if date_str and "T" not in value_str and "-" not in value_str.split(":")[0]:
        combined = f"{date_str.strip()}T{value_str}"
        try:
            t = Time(combined, format="isot", scale="utc")
            return t.isot, float(t.mjd)
        except Exception:
            pass  # fall through to other formats

    for fmt in ("isot", "fits", "iso"):
        try:
            t = Time(value_str, format=fmt, scale="utc")
            return t.isot, float(t.mjd)
        except Exception:
            continue

    try:
        t = Time(value_str)
        return t.isot, float(t.mjd)
    except Exception:
        return value_str, np.nan


def build_time_series_table(
    stacked: Table,
    sci_files: List[str],
    obs_times: List,
    n_amps: int,
    obs_dates: Optional[List] = None,
    airmasses: Optional[List[float]] = None,
    filters: Optional[List[str]] = None,
    focus_positions: Optional[List[float]] = None,
    quality_cuts: Optional[Dict[str, Tuple[str, float]]] = None,
) -> Table:
    metrics = aggregate_per_file_metrics(
        stacked,
        len(sci_files),
        n_amps,
        ("FWHM", "e"),
        quality_cuts=quality_cuts,
    )

    if obs_dates is None:
        obs_dates = [None] * len(obs_times)

    iso_times: List[Optional[str]] = []
    mjd_times: List[float] = []
    for raw_time, raw_date in zip(obs_times, obs_dates):
        date_str = str(raw_date).strip() if raw_date else None
        iso, mjd = _normalize_obs_time(raw_time, date_str=date_str)
        iso_times.append(iso if iso is not None else "")
        mjd_times.append(mjd)

    tbl = Table()
    tbl["sci_file"] = [Path(p).name for p in sci_files]
    tbl["obs_time_iso"] = iso_times
    tbl["obs_time_mjd"] = mjd_times
    tbl["avg_fwhm"] = metrics["FWHM"]
    tbl["avg_e"] = metrics["e"]
    if airmasses is not None:
        tbl["airmass"] = np.array(airmasses, dtype=float)
    if filters is not None:
        tbl["filter"] = filters
    if focus_positions is not None:
        tbl["focus_position"] = np.array(focus_positions, dtype=float)
    return tbl


def plot_time_series_metrics(
    time_table: Table, output_path: Path, pixscale: Optional[float] = None,
) -> None:
    if "obs_time_mjd" not in time_table.colnames:
        raise ValueError("time_table must contain 'obs_time_mjd'")
    if "avg_fwhm" not in time_table.colnames or "avg_e" not in time_table.colnames:
        raise ValueError("time_table must contain 'avg_fwhm' and 'avg_e'")

    mjd = np.array(time_table["obs_time_mjd"], dtype=float)
    fwhm_pix = np.array(time_table["avg_fwhm"], dtype=float)
    ellipticity = np.array(time_table["avg_e"], dtype=float)

    # Convert FWHM to arcsec if plate scale is available
    if pixscale and pixscale > 0:
        fwhm = fwhm_pix * pixscale
        fwhm_label = 'FWHM (arcsec)'
    else:
        fwhm = fwhm_pix
        fwhm_label = 'FWHM (pix)'

    airmass = None
    if "airmass" in time_table.colnames:
        airmass = np.array(time_table["airmass"], dtype=float)
    has_airmass = airmass is not None and np.any(np.isfinite(airmass))

    # Multi-band colour map (darkorange + steelblue, seaborn deep)
    band_colors = {"U": "#8172B3", "G": "#4682B4", "R": "#FF8C00",
                   "I": "#C44E52", "Z": "#937860"}
    filters_col = None
    if "filter" in time_table.colnames:
        filters_col = [str(v).upper() for v in time_table["filter"]]
        unique_bands = sorted(set(filters_col))
    else:
        unique_bands = [""]

    finite_time = np.isfinite(mjd)
    if finite_time.sum() == 0:
        warnings.warn("No finite observation times available for time-series plot")
        return

    n_panels = 3 if has_airmass else 2
    height_ratios = [3, 2, 2] if has_airmass else [3, 2]
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(8, 3 + 2.5 * n_panels),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    ax1 = axes[0]

    def _plot_band(ax, vals, marker, default_color, default_label):
        if filters_col and len(unique_bands) > 1:
            for band in unique_bands:
                mask_b = np.array([f == band for f in filters_col])
                mask = finite_time & np.isfinite(vals) & mask_b
                if mask.sum() == 0:
                    continue
                color = band_colors.get(band, default_color)
                ax.plot(
                    Time(mjd[mask], format="mjd", scale="utc").to_datetime(),
                    vals[mask],
                    marker=marker, linestyle="-", color=color,
                    label=band.lower() if band else default_label, alpha=0.85,
                )
        else:
            mask = finite_time & np.isfinite(vals)
            if mask.sum() > 0:
                ax.plot(
                    Time(mjd[mask], format="mjd", scale="utc").to_datetime(),
                    vals[mask],
                    marker=marker, linestyle="-", color=default_color,
                    label=default_label,
                )

    _plot_band(ax1, fwhm, "o", "tab:blue", "Average FWHM")
    ax1.set_ylabel(fwhm_label)
    ax1.set_title("Per-exposure averages across all amps")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    # Airmass panel
    if has_airmass:
        ax_am = axes[1]
        _plot_band(ax_am, airmass, "d", "tab:green", "Airmass")
        ax_am.set_ylabel("Airmass")
        ax_am.grid(True, alpha=0.3)
        ax_am.legend(loc="best")

    # Ellipticity panel (last panel)
    ax_e = axes[-1]
    _plot_band(ax_e, ellipticity, "s", "tab:orange", "Average ellipticity")
    ax_e.set_ylabel("Ellipticity")
    ax_e.set_xlabel("UTC time")
    ax_e.grid(True, alpha=0.3)
    ax_e.legend(loc="best")

    fig.autofmt_xdate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_time_series_metrics_interactive(
    time_table: Table,
    output_path: Path,
    star_stack_b64: Optional[List[Optional[str]]] = None,
    pixscale: Optional[float] = None,
) -> None:
    """Create an interactive HTML time-series plot with hover tooltips.

    Each data point shows the source filename, observation time, FWHM, and
    ellipticity when the mouse hovers over it.  When *star_stack_b64* is
    provided the FWHM panel also shows a median-stacked star thumbnail so
    you can visually judge the PSF shape and spot de-focus at a glance.

    Requires the ``plotly`` package.
    """
    if not HAS_PLOTLY:
        warnings.warn(
            "plotly is not installed – skipping interactive time-series plot. "
            "Install it with: pip install plotly"
        )
        return

    if "obs_time_mjd" not in time_table.colnames:
        raise ValueError("time_table must contain 'obs_time_mjd'")
    if "avg_fwhm" not in time_table.colnames or "avg_e" not in time_table.colnames:
        raise ValueError("time_table must contain 'avg_fwhm' and 'avg_e'")

    mjd = np.array(time_table["obs_time_mjd"], dtype=float)
    fwhm_pix = np.array(time_table["avg_fwhm"], dtype=float)
    ellipticity = np.array(time_table["avg_e"], dtype=float)
    airmass = (
        np.array(time_table["airmass"], dtype=float)
        if "airmass" in time_table.colnames
        else None
    )

    # Multi-band support
    # darkorange + steelblue, seaborn deep
    band_plotly_colors = {
        "U": "#8172B3", "G": "#4682B4", "R": "#FF8C00",
        "I": "#C44E52", "Z": "#937860",
    }
    filters_col = None
    if "filter" in time_table.colnames:
        filters_col = [str(v).upper() for v in time_table["filter"]]
        unique_bands = sorted(set(filters_col))
    else:
        unique_bands = [""]
    multiband = filters_col is not None and len(unique_bands) > 1

    # Convert FWHM to arcsec if plate scale is available
    if pixscale and pixscale > 0:
        fwhm = fwhm_pix * pixscale
        fwhm_label = 'FWHM (arcsec)'
        fwhm_unit = 'arcsec'
    else:
        fwhm = fwhm_pix
        fwhm_label = 'FWHM (pix)'
        fwhm_unit = 'pix'
    filenames = list(time_table["sci_file"]) if "sci_file" in time_table.colnames else [""] * len(mjd)
    iso_times = (
        list(time_table["obs_time_iso"])
        if "obs_time_iso" in time_table.colnames
        else [""] * len(mjd)
    )

    finite_time = np.isfinite(mjd)
    if finite_time.sum() == 0:
        warnings.warn("No finite observation times available for interactive plot")
        return

    has_airmass = airmass is not None and np.any(np.isfinite(airmass))

    # ---- Build hover text (text only – no <img>; images via JS overlay) ----
    def _hover_text(idx):
        parts = []
        parts.append(f"<b>{filenames[idx]}</b>")
        parts.append(f"Time: {iso_times[idx]}")
        if filters_col:
            parts.append(f"Filter: {filters_col[idx].lower()}")
        parts.append(f"FWHM: {fwhm[idx]:.3f} {fwhm_unit}")
        parts.append(f"Ellipticity: {ellipticity[idx]:.4f}")
        if has_airmass and np.isfinite(airmass[idx]):
            parts.append(f"Airmass: {airmass[idx]:.3f}")
        return "<br>".join(parts)

    # Build subplot layout: 3 rows if airmass available, else 2
    if has_airmass:
        n_rows = 3
        row_heights = [0.45, 0.25, 0.30]
        subplot_titles = ("Average FWHM", "Airmass", "Average Ellipticity")
        ellip_row = 3
    else:
        n_rows = 2
        row_heights = [0.6, 0.4]
        subplot_titles = ("Average FWHM", "Average Ellipticity")
        ellip_row = 2

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # Helper to add traces per band (or single trace if single band)
    def _add_band_traces(
        panel_row, y_arr, marker_sym, default_color, name_prefix,
        include_customdata=False,
    ):
        bands_iter = unique_bands if multiband else [""]
        for band in bands_iter:
            if multiband:
                mask_b = np.array([f == band for f in filters_col])
            else:
                mask_b = np.ones(len(mjd), dtype=bool)
            mask = finite_time & np.isfinite(y_arr) & mask_b
            if mask.sum() == 0:
                continue
            idx_sel = np.where(mask)[0]
            utc_sel = Time(mjd[mask], format="mjd", scale="utc").to_datetime()
            hover_sel = [_hover_text(i) for i in idx_sel]
            color = band_plotly_colors.get(band, default_color) if multiband else default_color
            trace_name = f"{name_prefix} ({band.lower()})" if multiband else name_prefix

            kwargs = dict(
                x=list(utc_sel),
                y=y_arr[mask].tolist(),
                mode="lines+markers",
                marker=dict(size=8 if panel_row == 1 else 6, symbol=marker_sym, color=color),
                line=dict(color=color),
                name=trace_name,
                hovertext=hover_sel,
                hoverinfo="text",
                legendgroup=band if multiband else None,
                showlegend=(panel_row == 1) if multiband else True,
            )
            if include_customdata:
                import re as _re
                _num_re = _re.compile(r"(\d+)")
                cd = []
                for i in idx_sel:
                    fname = filenames[i] if i < len(filenames) else ""
                    stem = Path(fname).stem if fname else ""
                    nums = _num_re.findall(stem)
                    img_num = int(nums[-1]) if nums else i
                    cd.append([img_num])
                kwargs["customdata"] = cd
            fig.add_trace(go.Scatter(**kwargs), row=panel_row, col=1)

    # FWHM panel (row 1) — with PSF customdata
    _add_band_traces(1, fwhm, "circle", "#1f77b4", "FWHM", include_customdata=True)

    # Airmass panel (row 2)
    if has_airmass:
        _add_band_traces(2, airmass, "diamond", "#2ca02c", "Airmass")

    # Ellipticity panel
    _add_band_traces(ellip_row, ellipticity, "square", "#ff7f0e", "Ellipticity")

    fig.update_yaxes(title_text=fwhm_label, row=1, col=1)
    if has_airmass:
        fig.update_yaxes(title_text="Airmass", row=2, col=1)
    fig.update_yaxes(title_text="Ellipticity", row=ellip_row, col=1)
    fig.update_xaxes(title_text="UTC time", row=n_rows, col=1)
    fig.update_layout(
        title="Per-exposure averages across all amps",
        height=250 * n_rows,
        hovermode="closest",
        template="plotly_white",
    )

    # --- Write HTML with custom JS for PSF image overlay on hover + auto-refresh ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_html = fig.to_html(include_plotlyjs="cdn", full_html=True)

    # Inject auto-refresh meta tag (every 30 seconds) into <head>
    auto_refresh_meta = '<meta http-equiv="refresh" content="30">\n'
    raw_html = raw_html.replace("<head>", "<head>\n" + auto_refresh_meta)

    # Inject a floating overlay div + JS for PSF hover + a live-update status bar
    from datetime import datetime as _dt
    generated_time = _dt.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    psf_overlay_js = f"""
<style>
#psf-overlay {{
    display: none;
    position: fixed;
    pointer-events: none;
    z-index: 10000;
    background: white;
    border: 2px solid #333;
    border-radius: 8px;
    padding: 6px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.35);
    bottom: 12px;
    right: 12px;
}}
#psf-overlay img {{
    display: block;
    width: 280px;
    height: 280px;
}}
#psf-overlay .psf-label {{
    text-align: center;
    font-size: 11px;
    color: #555;
    margin-top: 2px;
}}
#refresh-bar {{
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: #222;
    color: #aaa;
    font-size: 12px;
    padding: 4px 12px;
    display: flex;
    justify-content: space-between;
    z-index: 9999;
}}
#refresh-bar .updated {{ color: #4CAF50; }}
#refresh-bar .countdown {{ color: #FFB74D; }}
#fit-controls {{
    position: fixed;
    top: 12px;
    left: 12px;
    background: rgba(34,34,34,0.93);
    color: #eee;
    border-radius: 8px;
    padding: 10px 14px;
    z-index: 9998;
    font-size: 13px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-width: 270px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.45);
    cursor: move;
    user-select: none;
}}
#fit-selected-btn {{
    background: #4CAF50;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 7px 14px;
    cursor: pointer;
    font-size: 13px;
    font-weight: bold;
}}
#fit-selected-btn:hover {{ background: #45a049; }}
#fit-selected-btn:disabled {{ background: #555; cursor: not-allowed; }}
#fit-selected-count {{ color: #FFB74D; font-size: 12px; }}
#fit-status {{ color: #90CAF9; font-size: 12px; word-break: break-word; }}
#amp-fwhm-panel {{
    position: fixed;
    bottom: 36px;
    right: 12px;
    background: rgba(34,34,34,0.94);
    color: #eee;
    border-radius: 8px;
    padding: 10px 12px;
    z-index: 9998;
    font-size: 12px;
    min-width: 270px;
    max-width: 360px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.45);
    display: none;
}}
#amp-fwhm-panel b {{ color: #FFB74D; }}
#amp-fwhm-panel button {{
    float: right;
    background: #555;
    color: white;
    border: none;
    border-radius: 3px;
    cursor: pointer;
    font-size: 11px;
}}
#amp-fwhm-panel table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
    font-family: monospace;
}}
#amp-fwhm-panel th,
#amp-fwhm-panel td {{
    padding: 2px 4px;
    text-align: right;
    border-bottom: 1px solid rgba(255,255,255,0.12);
}}
#amp-fwhm-panel th:first-child,
#amp-fwhm-panel td:first-child {{ text-align: left; }}
#amp-fwhm-panel .muted {{ color: #aaa; font-size: 11px; }}
#tilt-panel {{
    position: fixed;
    bottom: 36px;
    right: 12px;
    background: rgba(34,34,34,0.93);
    color: #eee;
    border-radius: 8px;
    padding: 10px 14px;
    z-index: 9998;
    font-size: 12px;
    max-width: 260px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.45);
    display: none;
}}
#tilt-panel b {{ color: #FFB74D; }}
#tilt-panel .lvdt-row {{ font-family: monospace; margin: 2px 0; }}
#tilt-panel .delta-pos {{ color: #66bb6a; }}
#tilt-panel .delta-neg {{ color: #ef5350; }}
#tilt-panel a {{ color: #58A6FF; cursor: pointer; text-decoration: underline; }}
#tilt-map-overlay {{
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(255,255,255,0.94);
    z-index: 10001;
    display: none;
    justify-content: center;
    align-items: center;
    cursor: zoom-out;
}}
#tilt-map-overlay .tilt-map-content {{
    display: flex;
    gap: 16px;
    align-items: center;
    justify-content: center;
    max-width: 96vw;
    max-height: 92vh;
}}
#tilt-map-overlay .tilt-map-pane {{
    color: #222;
    font-family: sans-serif;
    font-size: 12px;
    text-align: center;
}}
#tilt-map-overlay .tilt-map-pane.tilt-figure img {{
    display: block;
    max-width: 74vw;
    max-height: 88vh;
    background: white;
    box-shadow: 0 6px 28px rgba(0,0,0,0.22);
}}
#tilt-map-overlay .amp-detail-pane {{
    width: min(28vw, 380px);
    max-height: 88vh;
    overflow: auto;
    background: #fff;
    color: #222;
    border: 1px solid #d7dce2;
    border-radius: 8px;
    padding: 12px;
    box-shadow: 0 6px 28px rgba(0,0,0,0.20);
    text-align: left;
}}
#tilt-map-overlay .amp-detail-pane b {{ color: #a75a00; }}
#tilt-map-overlay .amp-detail-pane .muted {{ color: #555; font-size: 11px; margin-top: 3px; }}
#tilt-map-overlay .amp-detail-pane table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
    font-family: monospace;
    font-size: 12px;
}}
#tilt-map-overlay .amp-detail-pane th,
#tilt-map-overlay .amp-detail-pane td {{
    padding: 4px 5px;
    text-align: right;
    border-bottom: 1px solid #e2e6ea;
}}
#tilt-map-overlay .amp-detail-pane th:first-child,
#tilt-map-overlay .amp-detail-pane td:first-child {{ text-align: left; }}
#tilt-map-overlay .amp-detail-pane .error {{ color: #b42318; margin-top: 6px; }}
#tilt-map-overlay .amp-detail-pane .loading {{ color: #1f5f99; margin-top: 6px; }}
#tilt-map-overlay .amp-detail-title {{
    margin-bottom: 8px;
    font-weight: bold;
    color: #222;
}}
#tilt-map-overlay .tilt-map-label {{ margin-top: 6px; }}
#tilt-map-overlay .close-hint {{
    position: absolute; top: 14px; right: 18px;
    color: #555; font-size: 13px; font-family: sans-serif;
}}
@media (max-width: 900px) {{
    #tilt-map-overlay .tilt-map-content {{ flex-direction: column; }}
    #tilt-map-overlay .tilt-map-pane.tilt-figure img {{ max-width: 94vw; max-height: 58vh; }}
    #tilt-map-overlay .amp-detail-pane {{ width: min(92vw, 420px); max-height: 30vh; }}
}}
</style>
<div id="psf-overlay"><img id="psf-img" src=""><div class="psf-label">PSF contour</div></div>
<div id="tilt-map-overlay">
    <span class="close-hint">click to close</span>
    <div class="tilt-map-content">
        <div class="tilt-map-pane tilt-figure">
            <img id="tilt-map-img" src="">
            <div class="tilt-map-label" id="tilt-map-label">Tilt map</div>
        </div>
        <div class="tilt-map-pane amp-detail-pane" id="amp-detail-pane">
            <div id="amp-detail-content"></div>
        </div>
    </div>
</div>
<div id="refresh-bar">
    <span>Last updated: <span class="updated">{generated_time}</span></span>
    <span>Auto-refresh in <span class="countdown" id="countdown">60</span>s</span>
</div>
<div id="fit-controls">
    <div><b>Focus Fit</b></div>
    <div style="font-size:11px;color:#aaa">Use the box/lasso tool to select points, then click:</div>
    <button id="fit-selected-btn">Fit Selected</button>
    <button id="reset-view-btn" style="background:#555;color:white;border:none;border-radius:4px;padding:5px 14px;cursor:pointer;font-size:12px;">Reset View</button>
    <div id="fit-selected-count"></div>
    <div id="fit-status"></div>
</div>
<div id="amp-fwhm-panel">
    <button id="amp-fwhm-close">x</button>
    <div id="amp-fwhm-content"></div>
</div>
<script>
(function() {{
    var overlay = document.getElementById('psf-overlay');
    var psfImg  = document.getElementById('psf-img');
    var selectedFrameNums = [];
    var psfThumbs = null;
    var ampPanel = document.getElementById('amp-fwhm-panel');
    var ampContent = document.getElementById('amp-fwhm-content');
    var ampClose = document.getElementById('amp-fwhm-close');
    if (ampClose) ampClose.onclick = function() {{ ampPanel.style.display = 'none'; }};
    fetch('psf_thumbnails.json')
        .then(function(r) {{ return r.json(); }})
        .then(function(d) {{ psfThumbs = d; }})
        .catch(function() {{ psfThumbs = {{}}; }});

    // --- Tilt map overlay click-to-close ---
    (function() {{
        var mapOverlay = document.getElementById('tilt-map-overlay');
        if (mapOverlay) mapOverlay.onclick = function() {{ mapOverlay.style.display = 'none'; }};
    }})();

    function renderAmpFwhmTable(d) {{
        var html = '<b>Frame ' + d.frame + '</b>';
        html += '<div class="muted">' + d.sci_file + '</div>';
        html += '<div class="muted">Focus: ' + d.focus_position + '</div>';
        html += '<div class="muted">Median all amps: ' + d.avg_fwhm_arcsec + ' arcsec (' + d.avg_fwhm_pix + ' pix)</div>';
        html += '<table><thead><tr><th>Amp</th><th>FWHM"</th><th>FWHM px</th><th>N</th></tr></thead><tbody>';
        d.amps.forEach(function(row) {{
            html += '<tr><td>' + row.amp + '</td><td>' +
                    (row.fwhm_arcsec == null ? '--' : row.fwhm_arcsec.toFixed(3)) +
                    '</td><td>' +
                    (row.fwhm_pix == null ? '--' : row.fwhm_pix.toFixed(3)) +
                    '</td><td>' + row.n_stars + '</td></tr>';
        }});
        html += '</tbody></table>';
        return html;
    }}

    var ampFwhmRowsCache = null;
    function loadAmpFwhmFromEcsv(frame) {{
        function parseRows(text) {{
            var lines = text.split(/\\r?\\n/);
            var header = null;
            var rows = [];
            lines.forEach(function(line) {{
                line = line.trim();
                if (!line || line.charAt(0) === '#') return;
                if (!header) {{
                    header = line.split(/\\s+/);
                    return;
                }}
                var values = line.split(/\\s+/);
                if (values.length !== header.length) return;
                var row = {{}};
                header.forEach(function(key, idx) {{ row[key] = values[idx]; }});
                rows.push(row);
            }});
            return rows;
        }}
        function buildPayload(rows) {{
            var frameRows = rows.filter(function(row) {{
                return Number(row.image_number) === Number(frame);
            }});
            if (!frameRows.length) throw new Error('No per-amplifier FWHM data found for frame ' + frame);
            var pixscale = {pixscale if pixscale and pixscale > 0 else 0.455};
            var amps = frameRows.map(function(row) {{
                var fwhmPix = Number(row.median_fwhm);
                return {{
                    amp: Number(row.amp),
                    fwhm_pix: Number.isFinite(fwhmPix) ? fwhmPix : null,
                    fwhm_arcsec: Number.isFinite(fwhmPix) ? fwhmPix * pixscale : null,
                    median_e: Number(row.median_e),
                    n_stars: Number(row.n_stars)
                }};
            }});
            var valid = amps.map(function(row) {{ return row.fwhm_pix; }})
                            .filter(function(v) {{ return v != null && Number.isFinite(v); }})
                            .sort(function(a, b) {{ return a - b; }});
            var mid = Math.floor(valid.length / 2);
            var avgPix = valid.length % 2 ? valid[mid] : 0.5 * (valid[mid - 1] + valid[mid]);
            return {{
                frame: Number(frame),
                sci_file: frameRows[0].sci_file,
                focus_position: Number(frameRows[0].focus_position).toFixed(4),
                pixscale: pixscale,
                avg_fwhm_pix: avgPix.toFixed(4),
                avg_fwhm_arcsec: (avgPix * pixscale).toFixed(4),
                amps: amps
            }};
        }}
        var rowsPromise = ampFwhmRowsCache
            ? Promise.resolve(ampFwhmRowsCache)
            : fetch('focus_per_amp_points.ecsv')
                .then(function(r) {{
                    if (!r.ok) throw new Error('focus_per_amp_points.ecsv not found');
                    return r.text();
                }})
                .then(function(text) {{
                    ampFwhmRowsCache = parseRows(text);
                    return ampFwhmRowsCache;
                }});
        return rowsPromise.then(buildPayload);
    }}

    function loadAmpFwhmFromServer(frame) {{
        return fetch('/amp_fwhm', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{frame: frame}})
        }})
        .then(function(r) {{
            var contentType = r.headers.get('content-type') || '';
            if (contentType.indexOf('application/json') === -1) {{
                throw new Error('Live amp-FWHM service is not available');
            }}
            return r.json().then(function(d) {{ return {{ok: r.ok, data: d}}; }});
        }})
        .then(function(result) {{
            if (!result.ok) throw new Error(result.data.error || 'Amp FWHM request failed');
            return result.data;
        }});
    }}

    function showTiltAndAmpTable(frame) {{
        var mapOverlay = document.getElementById('tilt-map-overlay');
        var mapImg = document.getElementById('tilt-map-img');
        var mapLabel = document.getElementById('tilt-map-label');
        var ampDetailContent = document.getElementById('amp-detail-content');
        if (!mapOverlay || !mapImg) return;
        var frameKey = String(frame);
        mapImg.onerror = function() {{
            mapImg.onerror = null;
            mapImg.src = 'tilt_map.png';
            if (mapLabel) mapLabel.textContent = 'Tilt map (default)';
        }};
        mapImg.src = 'tilt_map_' + frameKey + '.png';
        if (mapLabel) mapLabel.textContent = 'Tilt map: frame ' + frameKey;
        if (ampDetailContent) {{
            ampDetailContent.innerHTML = '<b>Frame ' + frameKey + '</b><div class="loading">Loading amp FWHM...</div>';
        }}
        mapOverlay.style.display = 'flex';
    }}

    var gd = document.querySelector('.plotly-graph-div') ||
             document.querySelector('[class*="plotly"]');
    if (gd) {{
        // --- PSF hover overlay (thumbnails loaded from psf_thumbnails.json) ---
        gd.on('plotly_hover', function(data) {{
            var pt = data.points[0];
            if (pt && pt.customdata != null && psfThumbs) {{
                var b64 = psfThumbs[String(pt.customdata[0])];
                if (b64) {{
                    psfImg.src = 'data:image/png;base64,' + b64;
                    overlay.style.display = 'block';
                    return;
                }}
            }}
            overlay.style.display = 'none';
        }});
        gd.on('plotly_unhover', function() {{
            overlay.style.display = 'none';
        }});

        // --- Click any point -> show per-amplifier FWHM without writing PNGs ---
        gd.on('plotly_click', function(data) {{
            var pt = data.points && data.points[0];
            if (!pt || !pt.customdata || pt.customdata[0] == null) return;
            var frame = pt.customdata[0];
            showTiltAndAmpTable(frame);
            if (ampPanel) ampPanel.style.display = 'none';
            loadAmpFwhmFromServer(frame)
            .then(function(d) {{
                document.getElementById('amp-detail-content').innerHTML = renderAmpFwhmTable(d);
            }})
            .catch(function(err) {{
                if (window.console) console.warn(err.message);
                loadAmpFwhmFromEcsv(frame)
                    .then(function(d) {{
                        document.getElementById('amp-detail-content').innerHTML = renderAmpFwhmTable(d);
                    }})
                    .catch(function(fallbackErr) {{
                        document.getElementById('amp-detail-content').innerHTML =
                            '<b>Frame ' + frame + '</b><div class="error">' +
                            fallbackErr.message + '</div>';
                    }});
            }});
        }});

        // --- Frame selection for focus fit ---
        function loadTiltForFrame(frame, openMap) {{ /* deprecated: kept as no-op */ }}

        gd.on('plotly_selected', function(eventData) {{
            var countEl = document.getElementById('fit-selected-count');
            if (!eventData || !eventData.points) {{
                selectedFrameNums = [];
                countEl.textContent = '';
                return;
            }}
            var nums = eventData.points
                .filter(function(pt) {{ return pt.customdata && pt.customdata[0] != null; }})
                .map(function(pt) {{ return pt.customdata[0]; }});
            // deduplicate
            selectedFrameNums = nums.filter(function(v, i, a) {{ return a.indexOf(v) === i; }});
            countEl.textContent = selectedFrameNums.length + ' frame(s) selected';
        }});
        gd.on('plotly_deselect', function() {{
            selectedFrameNums = [];
            document.getElementById('fit-selected-count').textContent = '';
            document.getElementById('fit-status').textContent = '';
        }});
    }}

    // --- Draggable fit-controls panel ---
    (function() {{
        var panel = document.getElementById('fit-controls');
        var drag = {{ active: false, startX: 0, startY: 0, origLeft: 0, origTop: 0 }};
        panel.addEventListener('mousedown', function(e) {{
            if (e.target.tagName === 'BUTTON') return;
            drag.active = true;
            drag.startX = e.clientX;
            drag.startY = e.clientY;
            drag.origLeft = panel.offsetLeft;
            drag.origTop  = panel.offsetTop;
            e.preventDefault();
        }});
        document.addEventListener('mousemove', function(e) {{
            if (!drag.active) return;
            panel.style.left = (drag.origLeft + e.clientX - drag.startX) + 'px';
            panel.style.top  = (drag.origTop  + e.clientY - drag.startY) + 'px';
        }});
        document.addEventListener('mouseup', function() {{ drag.active = false; }});
    }})();

    // --- Reset View button ---
    document.getElementById('reset-view-btn').onclick = function() {{
        if (gd) {{
            Plotly.relayout(gd, {{
                'xaxis.autorange': true, 'yaxis.autorange': true,
                'xaxis2.autorange': true, 'yaxis2.autorange': true,
                'xaxis3.autorange': true, 'yaxis3.autorange': true,
            }});
        }}
    }};

    // --- Fit Selected button ---
    document.getElementById('fit-selected-btn').onclick = function() {{
        if (selectedFrameNums.length < 3) {{
            alert('Please select at least 3 data points for a valid parabola fit.');
            return;
        }}
        var btn      = document.getElementById('fit-selected-btn');
        var statusEl = document.getElementById('fit-status');
        btn.disabled = true;
        statusEl.textContent = 'Running fit\u2026';
        fetch('/fit_selected', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{frames: selectedFrameNums}})
        }})
        .then(function(r) {{ return r.json(); }})
        .then(function(data) {{
            btn.disabled = false;
            if (data.error) {{
                statusEl.textContent = 'Error: ' + data.error;
            }} else if (data.best_focus !== undefined) {{
                // Fast path: result available immediately, no page reload needed
                statusEl.innerHTML = '\u2713 Best focus: <strong>' + data.best_focus +
                    '</strong> &nbsp;R\u00b2=' + data.r2 +
                    ' &nbsp;<a href="/focus_fit.png" target="_blank" style="color:#4af">view plot</a>';
            }} else {{
                statusEl.textContent = 'Fit done! Refreshing\u2026';
                setTimeout(function() {{ location.reload(); }}, 3000);
            }}
        }})
        .catch(function(e) {{
            btn.disabled = false;
            statusEl.textContent = 'Request failed: ' + e.message;
        }});
    }};

    // --- Countdown timer ---
    var remaining = 60;
    var el = document.getElementById('countdown');
    setInterval(function() {{
        remaining--;
        if (remaining < 0) remaining = 0;
        el.textContent = remaining;
    }}, 1000);
}})();
</script>
"""
    # Insert the overlay just before </body>
    raw_html = raw_html.replace("</body>", psf_overlay_js + "\n</body>")
    output_path.write_text(raw_html, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focus finder script derived from notebook")
    parser.add_argument("--data-dir", type=str, default=".", help="Directory with FITS files")
    parser.add_argument(
        "--filter",
        required=True,
        help="Photometric filter: a single band (e.g. R), a comma-separated "
             "list (e.g. r,g,i), or 'all' to auto-detect every band present "
             "in the science frames.",
    )
    parser.add_argument("--bias-nums", nargs="+", type=str, required=True, help="Bias image numbers or ranges")
    parser.add_argument("--dark-nums", nargs="*", type=str, default=[], help="Dark image numbers or ranges (optional; omit if no darks)")
    parser.add_argument(
        "--flat-nums",
        nargs="+",
        type=str,
        default=None,
        help="Flat image numbers or ranges.  Required for single-band mode.  "
             "In multi-band / 'all' mode, if omitted the pipeline auto-discovers "
             "flat frames per band from the FITS headers.",
    )
    parser.add_argument("--sci-nums", nargs="+", type=str, required=True, help="Science image numbers or ranges")
    parser.add_argument(
        "--focus-key",
        default="LVDTC",
        help="Header keyword that carries the focus position",
    )
    parser.add_argument(
        "--time-key",
        default="TIME-OBS",
        help="Header keyword that carries the observation timestamp",
    )
    parser.add_argument(
        "--date-key",
        default="DATE-OBS",
        help="Header keyword that carries the observation date (combined with --time-key when the time value has no date)",
    )
    parser.add_argument(
        "--airmass-key",
        default="AIRMASS",
        help="Header keyword that carries the airmass value (default: AIRMASS)",
    )
    parser.add_argument(
        "--pixscale",
        type=float,
        default=0.455,
        help="Plate scale in arcsec/pixel for scale bar on PSF thumbnails (default: 0.455 for Bok 90Prime)",
    )
    parser.add_argument(
        "--amps",
        nargs="+",
        type=int,
        default=list(range(1, 9)),
        help="Amplifier numbers to include (1-indexed)",
    )
    parser.add_argument("--threshold", type=float, default=25.0, help="SEP detection threshold")
    parser.add_argument(
        "--mask-dir",
        type=str,
        default="bad_pixel_masks",
        help="Directory with bad pixel masks (default: bad_pixel_masks)",
    )
    parser.add_argument(
        "--auto-generate-masks",
        action="store_true",
        help="Derive bad pixel masks from master flats and save them to --mask-dir",
    )
    parser.add_argument(
        "--mask-sat-mult",
        type=float,
        default=1.0,
        help="Multiplier for std dev added to the median when computing saturation threshold",
    )
    parser.add_argument(
        "--mask-black-mult",
        type=float,
        default=4.0,
        help="Multiplier for std dev subtracted from the median when computing black-column threshold",
    )
    parser.add_argument(
        "--mask-sat-frac",
        type=float,
        default=0.25,
        help="Fraction of rows above the saturation threshold required to flag a column",
    )
    parser.add_argument(
        "--mask-black-frac",
        type=float,
        default=0.2,
        help="Fraction of rows below the black threshold required to flag a column",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="focus_output",
        help="Output directory for products",
    )
    write_group = parser.add_mutually_exclusive_group()
    write_group.add_argument(
        "--write-reduced",
        dest="write_reduced",
        action="store_true",
        help="Write reduced amplifier FITS files to disk",
    )
    write_group.add_argument(
        "--skip-reduced",
        dest="write_reduced",
        action="store_false",
        help="Skip writing reduced FITS products (overrides --write-reduced)",
    )
    parser.set_defaults(write_reduced=False)
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental mode: cache master calibrations and per-file detections "
             "so only new exposures are reduced on subsequent runs.  Cached "
             "results are stored in <outdir>/.cache/.  Delete that directory "
             "to force a full reprocessing.",
    )
    parser.add_argument(
        "--solve-tilt",
        action="store_true",
        help="After computing per-amplifier FWHM, solve for focal-plane tilt "
             "and recommend independent actuator adjustments (A, B, C).  "
             "Requires LVDTA/LVDTB/LVDTC header keywords.",
    )
    parser.add_argument(
        "--solve-tilt-frame",
        type=int,
        default=None,
        help="Image number to use for --solve-tilt (default: middle of sequence).",
    )
    parser.add_argument(
        "--global-tilt-fit",
        action="store_true",
        help="With --solve-tilt: fit a SINGLE atmosphere (FWHM_0, alpha) "
             "jointly across all frames, with per-frame (z0, a, b). "
             "Breaks the seeing/piston degeneracy so the piston part "
             "of the actuator correction becomes meaningful.",
    )
    parser.add_argument(
        "--no-fit",
        action="store_true",
        default=False,
        help="Skip the automatic parabola focus fit and plot. Use this when "
             "you want to manually select frames for fitting via the dashboard.",
    )
    parser.add_argument(
        "--fit-nums",
        nargs="+",
        type=str,
        default=None,
        help="Restrict the parabola fit to these frame numbers only. "
             "All --sci-nums frames are still reduced and shown in the dashboard.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mask_dir = Path(args.mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Determine which bands to process
    # ------------------------------------------------------------------
    filter_arg = args.filter.strip()
    if filter_arg.lower() == "all":
        requested_bands = ("U", "G", "R", "I", "Z")
    else:
        requested_bands = tuple(b.strip().upper() for b in filter_arg.split(","))

    categorized = skim_fits_files(directory=str(data_dir), target_bands=requested_bands)

    bias_numbers = expand_image_numbers(args.bias_nums)
    dark_numbers = expand_image_numbers(args.dark_nums) if args.dark_nums else []
    sci_numbers = expand_image_numbers(args.sci_nums)

    # Determine which bands actually have science frames + flats
    active_bands: List[str] = []
    for band in requested_bands:
        sci_pool = categorized[band]["other"]
        flat_pool = categorized[band]["flat"]
        # Check if any of the requested sci numbers exist in this band
        sci_found = select_files_by_numbers(sci_pool, sci_numbers)
        if not sci_found:
            continue
        if not flat_pool:
            # If explicit --flat-nums was given, check if those exist; else warn
            if args.flat_nums:
                flat_found = select_files_by_numbers(flat_pool, expand_image_numbers(args.flat_nums))
                if not flat_found:
                    print(f"  [warn] No flats found for band {band}; skipping this band")
                    continue
            else:
                print(f"  [warn] No flats found for band {band}; skipping this band")
                continue
        active_bands.append(band)

    if not active_bands:
        raise RuntimeError("No bands have both science frames and flats. Nothing to do.")

    print(f"\nBands to process: {', '.join(b.lower() for b in active_bands)}")

    # ------------------------------------------------------------------
    # Collect all science files across bands, sorted by frame number,
    # and record each file's band.
    # ------------------------------------------------------------------
    sci_files: List[str] = []
    sci_bands: List[str] = []  # parallel to sci_files
    for band in active_bands:
        band_sci = select_files_by_numbers(categorized[band]["other"], sci_numbers)
        for f in band_sci:
            sci_files.append(f)
            sci_bands.append(band)
    # Sort by filename to get chronological order
    sort_order = sorted(range(len(sci_files)), key=lambda i: Path(sci_files[i]).name)
    sci_files = [sci_files[i] for i in sort_order]
    sci_bands = [sci_bands[i] for i in sort_order]

    print(f"Total science frames selected: {len(sci_files)}")
    for band in active_bands:
        n = sum(1 for b in sci_bands if b == band)
        print(f"  {band.lower()}-band: {n} frames")

    cache_dir = _get_cache_dir(outdir)

    # ------------------------------------------------------------------
    # Master calibrations: one bias (shared), one flat per band per amp
    # ------------------------------------------------------------------
    # master_cache_per_band[band][amp] = (master_bias, _, _, master_flat)
    master_cache_per_band: Dict[str, Dict[int, Tuple]] = {}
    masters_from_cache = False

    if args.incremental:
        _validate_cache_params(cache_dir, args.threshold, list(args.amps))
        loaded_all = True
        for band in active_bands:
            master_cache_per_band[band] = {}
            for amp in args.amps:
                cached = _load_master_calibrations(cache_dir, amp, band=band)
                if cached is not None:
                    master_cache_per_band[band][amp] = cached
                else:
                    loaded_all = False
                    break
            if not loaded_all:
                break
        if loaded_all:
            masters_from_cache = True
            print(f"Loaded cached master calibrations for {len(active_bands)} band(s) "
                  f"× {len(args.amps)} amps")
        else:
            master_cache_per_band = {}  # reset partial load

    if not masters_from_cache:
        bias_files = select_files_by_numbers(categorized["bias_frames"], bias_numbers)
        dark_files = select_files_by_numbers(categorized["dark_frames"], dark_numbers)

        for band in active_bands:
            # Select flats for this band
            flat_pool = categorized[band]["flat"]
            if args.flat_nums:
                flat_files = select_files_by_numbers(flat_pool, expand_image_numbers(args.flat_nums))
            else:
                flat_files = sorted(flat_pool)  # auto: use all available flats
            print(f"\nBuilding master calibrations for {band.lower()}-band "
                  f"({len(flat_files)} flats) ...")

            master_cache_per_band[band] = {}
            for amp in args.amps:
                biases, darks, flats, _ = diff_amp(
                    amp, bias_files, dark_files, flat_files, sci_files[:1],
                )
                cal = flat_reduction_b(biases, darks, flats)
                master_cache_per_band[band][amp] = cal
                _save_master_calibrations(cache_dir, amp, cal, band=band)

                mask_path = mask_dir / f"bad_pixel_mask_amp_{band}{amp}.npy"
                if args.auto_generate_masks or not mask_path.exists():
                    master_flat = cal[3]
                    median_flat = np.median(master_flat)
                    std_flat = np.std(master_flat)
                    sat_threshold = median_flat + args.mask_sat_mult * std_flat
                    black_threshold = median_flat - args.mask_black_mult * std_flat
                    bad_cols = find_bad_columns(
                        master_flat,
                        sat_threshold,
                        black_threshold,
                        args.mask_sat_frac,
                        args.mask_black_frac,
                    )
                    mask_map = create_bad_pixel_map(master_flat.shape, bad_cols)
                    np.save(mask_path, mask_map)

    # ------------------------------------------------------------------
    # Source detection loop — per-file, using matching band's flat
    # ------------------------------------------------------------------
    all_tables: List[Table] = []
    all_cutouts: List[Optional[np.ndarray]] = []
    n_cached = 0
    n_processed = 0

    for file_idx, (sci_path, band) in enumerate(zip(sci_files, sci_bands)):
        master_cache = master_cache_per_band[band]

        if args.incremental:
            cached = _load_file_detections(cache_dir, sci_path)
            if cached is not None:
                tab, cutouts = cached
                # Rewrite subset_id to match current file position
                amp_arr = np.array(tab["_cache_amp"], dtype=int)
                tab["subset_id"] = np.array(
                    [f"{file_idx}_amp{a}" for a in amp_arr]
                )
                all_tables.append(tab)
                all_cutouts.extend(cutouts)
                n_cached += 1
                continue

        # Process this file
        print(f"  Processing {Path(sci_path).name} [{band.lower()}] "
              f"({file_idx + 1}/{len(sci_files)}) ...")
        tab, cutouts = _detect_single_file(
            sci_path, file_idx, list(args.amps), master_cache,
            band, mask_dir, args.threshold, outdir, args.write_reduced,
        )
        if tab is not None:
            if args.incremental:
                _save_file_detections(cache_dir, sci_path, tab, cutouts)
            all_tables.append(tab)
            all_cutouts.extend(cutouts)
            n_processed += 1
        else:
            print(f"  No detections in {Path(sci_path).name}")

    print(f"[detection] {n_cached} cached + {n_processed} newly processed")

    if not all_tables:
        raise RuntimeError("No detections found; cannot proceed with focus fit")

    stacked = vstack(all_tables, join_type="outer")
    if "_cache_amp" in stacked.colnames:
        stacked.remove_column("_cache_amp")
    sci_names = [Path(p).name for p in sci_files]

    catalog_path = outdir / "focus_sources.fits"
    stacked.write(catalog_path, overwrite=True)
    print(f"Wrote catalog with {len(stacked)} sources to {catalog_path}")

    # Load existing thumbnail cache (skip re-rendering already-known frames)
    thumb_cache_path = outdir / "psf_thumbnails.json"
    existing_thumb_cache: dict = {}
    if thumb_cache_path.exists():
        try:
            with open(thumb_cache_path) as _tf:
                existing_thumb_cache = json.load(_tf)
        except Exception:
            existing_thumb_cache = {}
    import re as _re_t
    _rnum_t = _re_t.compile(r"(\d+)")
    sci_img_nums = []
    for _sp in sci_files:
        _snums = _rnum_t.findall(Path(_sp).stem)
        sci_img_nums.append(int(_snums[-1]) if _snums else None)
    skip_indices = {
        i for i, n in enumerate(sci_img_nums)
        if n is not None and str(n) in existing_thumb_cache
    }
    star_stack_b64 = generate_star_stack_images(
        stacked, all_cutouts, len(sci_files), len(args.amps),
        pixscale=args.pixscale, skip_indices=skip_indices,
    )
    n_cutout_thumbs = sum(1 for s in star_stack_b64 if s)
    if n_cutout_thumbs < len(sci_files):
        catalog_b64 = generate_psf_contours_from_catalog(
            stacked, len(sci_files), len(args.amps),
            pixscale=args.pixscale, skip_indices=skip_indices,
        )
        for i in range(len(star_stack_b64)):
            if star_stack_b64[i] is None and i < len(catalog_b64):
                star_stack_b64[i] = catalog_b64[i]
    # Merge new thumbnails into persistent cache; restore cached entries
    for i, b64 in enumerate(star_stack_b64):
        n = sci_img_nums[i] if i < len(sci_img_nums) else None
        if b64 and n is not None:
            existing_thumb_cache[str(n)] = b64
        elif n is not None and str(n) in existing_thumb_cache:
            star_stack_b64[i] = existing_thumb_cache[str(n)]
    with open(thumb_cache_path, "w") as _tf:
        json.dump(existing_thumb_cache, _tf)
    n_rendered = sum(1 for s in star_stack_b64 if s)
    print(f"PSF thumbnails: {n_rendered} ready ({len(existing_thumb_cache)} total in cache).")

    med_fwhm = per_file_median_fwhm(stacked, len(sci_files), len(args.amps))
    focus_positions: List[float] = []
    obs_times: List = []
    obs_dates: List = []
    airmasses: List[float] = []
    for path in sci_files:
        with fits.open(path) as hdul:
            hdr0 = hdul[0].header
            # Focus position: if --focus-key is the default (LVDTC) and all
            # three LVDT keys are present, use their mean; otherwise honour
            # whatever single key the user requested.
            if args.focus_key == "LVDTC" and all(
                k in hdr0 for k in ("LVDTA", "LVDTB", "LVDTC")
            ):
                _vals = [
                    float(hdr0["LVDTA"]),
                    float(hdr0["LVDTB"]),
                    float(hdr0["LVDTC"]),
                ]
                focus_positions.append(float(np.mean(_vals)))
            else:
                focus_positions.append(hdr0.get(args.focus_key, np.nan))
            obs_times.append(hdr0.get(args.time_key))
            obs_dates.append(hdr0.get(args.date_key))
            airmasses.append(float(hdr0.get(args.airmass_key, np.nan)))
    focus_positions = np.array(focus_positions, dtype=float)
    med_fwhm = np.array(med_fwhm, dtype=float)

    mask_valid = np.isfinite(focus_positions) & np.isfinite(med_fwhm)
    if getattr(args, "fit_nums", None):
        fit_numbers = set(expand_image_numbers(args.fit_nums))
        import re as _re_fn
        _rnum_fn = _re_fn.compile(r"(\d+)")
        fit_mask = np.zeros(len(sci_files), dtype=bool)
        for _i, _sp in enumerate(sci_files):
            _nums = _rnum_fn.findall(Path(_sp).stem)
            if _nums and int(_nums[-1]) in fit_numbers:
                fit_mask[_i] = True
        mask_valid = mask_valid & fit_mask
    if getattr(args, "no_fit", False):
        print("[pipeline] --no-fit set: skipping automatic focus fit. "
              "Select frames manually from the dashboard to run the fit.")
    elif mask_valid.sum() >= 3:
        x = focus_positions[mask_valid]
        y = med_fwhm[mask_valid]
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        fit_result = fit_parabola_vertex_form(x, y)

        plot_path = outdir / "focus_fit.png"
        plot_focus_curve(x, y, fit_result, plot_path)

        print("Focus fit (vertex form):")
        print(f"  A = {fit_result['A']:.6g}")
        print(f"  h (focus position) = {fit_result['h']:.6g}")
        print(f"  k (minimum FWHM) = {fit_result['k']:.6g}")
        print(f"  R^2 = {fit_result['R2']:.4f}")
        if fit_result["x_min"] is not None:
            sigma_txt = (
                f" ± {fit_result['sigma_x_min']:.3f}"
                if fit_result["sigma_x_min"] is not None
                else ""
            )
            print(f"  Best focus at {fit_result['x_min']:.3f}{sigma_txt}")
        else:
            print("  Parabola fit is degenerate; cannot compute focus minimum.")
        print(f"Saved focus plot to {plot_path}")
    else:
        print("Not enough valid focus points for parabola fitting — skipping focus curve.")

    # ------------------------------------------------------------------
    # Tilt + focus solver  (--solve-tilt)
    # ------------------------------------------------------------------
    if getattr(args, "solve_tilt", False):
        print("\n--- Tilt + Focus analysis ---")
        import json as _json
        import re as _re_st
        _rnum_st = _re_st.compile(r"(\d+)")

        # Build (index, image_number) list once
        sci_idx_nums: List[tuple] = []
        for _i, _sp in enumerate(sci_files):
            _nums = _rnum_st.findall(Path(_sp).stem)
            sci_idx_nums.append((_i, int(_nums[-1]) if _nums else None))

        # Decide which frames to solve: explicit single frame, or all frames
        requested = getattr(args, "solve_tilt_frame", None)
        if requested is not None:
            req = int(requested)
            targets = [(i, n) for i, n in sci_idx_nums if n == req]
            if not targets:
                print(f"  --solve-tilt-frame {req} not in sci_files; using middle frame.")
                mid = len(sci_files) // 2
                targets = [(mid, sci_idx_nums[mid][1])]
        else:
            targets = [(i, n) for i, n in sci_idx_nums if n is not None]
            print(f"  Pre-computing tilt for {len(targets)} frames "
                  "(use --solve-tilt-frame N to limit to one).")

        # Pre-compute shared catalog arrays once
        sid_col = stacked["subset_id"] if "subset_id" in stacked.colnames else None
        if sid_col is None:
            print("  No subset_id column in catalog — cannot compute per-amp FWHM.")
        else:
            sid_arr = np.array([
                s.decode() if isinstance(s, (bytes, bytearray)) else str(s)
                for s in sid_col
            ])
            fwhm_col = np.array(stacked["FWHM"], dtype=float)
            labelst = compute_gmm_labels(stacked)

            default_payload = None  # written to tilt_result.json (most recent successful)
            n_ok = 0

            # ----- Gather per-frame inputs (used by both per-frame and global modes) -----
            frame_inputs = []   # list of dicts for solve_tilt_focus_global()
            for _idx, _num in targets:
                _path = sci_files[_idx]
                with fits.open(_path) as hdul:
                    hdr0 = hdul[0].header
                    lvdt_a = float(hdr0.get("LVDTA", np.nan))
                    lvdt_b = float(hdr0.get("LVDTB", np.nan))
                    lvdt_c = float(hdr0.get("LVDTC", np.nan))
                if not (np.isfinite(lvdt_a) and np.isfinite(lvdt_b) and np.isfinite(lvdt_c)):
                    continue
                current_lvdt = {"A": lvdt_a, "B": lvdt_b, "C": lvdt_c}

                fwhm_per_amp = {}
                for amp in args.amps:
                    amp_tag = f"{_idx}_amp{amp}"
                    mask = (sid_arr == amp_tag) & (labelst == 0) & np.isfinite(fwhm_col)
                    vals = fwhm_col[mask]
                    if len(vals) >= 3:
                        fwhm_per_amp[amp] = float(np.nanmedian(vals))
                if len(fwhm_per_amp) < 5:
                    continue
                frame_inputs.append({
                    "idx": _idx,
                    "image_number": _num,
                    "frame": Path(_path).name,
                    "fwhm_per_amp": fwhm_per_amp,
                    "current_lvdt": current_lvdt,
                })

            # ----- Global joint fit, if requested -----
            global_result = None
            if getattr(args, "global_tilt_fit", False) and len(frame_inputs) >= 2:
                print(f"  Running GLOBAL joint tilt+focus fit over "
                      f"{len(frame_inputs)} frames ...")
                try:
                    global_result = solve_tilt_focus_global(frame_inputs)
                    print(f"  Global fit:  seeing_floor = "
                          f"{global_result['seeing_floor'] * args.pixscale:.3f}\"   "
                          f"alpha = {global_result['alpha']:.4g}   "
                          f"R²_global = {global_result['R2_global']:.3f}")
                    # Persist global summary
                    with open(outdir / "tilt_global.json", "w") as _fp:
                        _json.dump({
                            "n_frames": len(frame_inputs),
                            "seeing_floor_pix": global_result["seeing_floor"],
                            "seeing_floor_arcsec": global_result["seeing_floor"] * args.pixscale,
                            "alpha": global_result["alpha"],
                            "R2_global": global_result["R2_global"],
                            "frames": [fr["image_number"] for fr in frame_inputs],
                        }, _fp, indent=2)
                except Exception as exc:
                    print(f"  Global fit failed ({exc}); falling back to per-frame.")
                    global_result = None

            # ----- Write per-frame artifacts -----
            for fi, fr in enumerate(frame_inputs):
                _idx = fr["idx"]; _num = fr["image_number"]
                fwhm_per_amp = fr["fwhm_per_amp"]
                current_lvdt = fr["current_lvdt"]

                if global_result is not None:
                    # Reuse joint-fit per-frame breakdown
                    tilt_result = global_result["per_frame"][fi].copy()
                    tilt_result["R2"] = global_result["R2_global"]
                else:
                    try:
                        tilt_result = solve_tilt_focus(fwhm_per_amp, current_lvdt)
                    except Exception as exc:
                        print(f"  Frame {_num}: tilt solve failed ({exc})")
                        continue

                map_name = f"tilt_map_{_num}.png" if _num is not None else "tilt_map.png"
                json_name = f"tilt_result_{_num}.json" if _num is not None else "tilt_result.json"
                tilt_plot = outdir / map_name
                plot_tilt_map(fwhm_per_amp, tilt_result, tilt_plot, pixscale=args.pixscale)

                payload = {
                    "frame": fr["frame"],
                    "image_number": _num,
                    "current_lvdt": current_lvdt,
                    "optimal_lvdt": tilt_result["optimal_lvdt"],
                    "corrections": tilt_result["corrections"],
                    "actuator_directions": actuator_direction_summary(
                        tilt_result["corrections"],
                        current_lvdt=current_lvdt,
                    ),
                    "seeing_floor_pix": float(tilt_result["seeing_floor"]),
                    "seeing_floor_arcsec": float(tilt_result["seeing_floor"] * args.pixscale),
                    "piston_z0": float(tilt_result["piston_z0"]),
                    "tilt_a": float(tilt_result["tilt_a"]),
                    "tilt_b": float(tilt_result["tilt_b"]),
                    "tilt_magnitude": float(tilt_result["tilt_magnitude"]),
                    "R2": float(tilt_result["R2"]),
                    "tilt_map_png": map_name,
                    "fit_mode": "global" if global_result is not None else "per_frame",
                }
                with open(outdir / json_name, "w") as _fp:
                    _json.dump(payload, _fp, indent=2)
                default_payload = payload
                n_ok += 1

            # Write a "default" tilt_result.json + tilt_map.png pointing to the
            # middle (or only) successful frame for back-compat with the dashboard
            if default_payload is not None:
                if requested is None and n_ok > 1:
                    mid = len(sci_files) // 2
                    mid_num = sci_idx_nums[mid][1]
                    candidate = outdir / f"tilt_result_{mid_num}.json"
                    if candidate.exists():
                        default_payload = _json.loads(candidate.read_text())
                with open(outdir / "tilt_result.json", "w") as _fp:
                    _json.dump(default_payload, _fp, indent=2)
                # Copy default tilt_map.png as well
                src_map = outdir / default_payload["tilt_map_png"]
                if src_map.exists():
                    import shutil as _sh
                    _sh.copyfile(src_map, outdir / "tilt_map.png")
                print(f"  Tilt solved for {n_ok}/{len(targets)} frames; "
                      f"default = frame {default_payload.get('image_number')}.")
            else:
                print("  No frames produced a valid tilt solution.")

    time_table = build_time_series_table(
        stacked,
        sci_files,
        obs_times,
        len(args.amps),
        obs_dates=obs_dates,
        airmasses=airmasses,
        filters=[b.lower() for b in sci_bands],
        focus_positions=focus_positions,
    )
    time_table_path = outdir / "focus_time_series.ecsv"
    time_table.write(time_table_path, format="ascii.ecsv", overwrite=True)
    print(f"Wrote FWHM/ellipticity time series to {time_table_path}")

    time_series_plot = outdir / "focus_time_series.png"
    plot_time_series_metrics(time_table, time_series_plot, pixscale=args.pixscale)
    print(f"Saved time-series QA plot to {time_series_plot}")

    interactive_plot = outdir / "focus_time_series.html"
    plot_time_series_metrics_interactive(
        time_table, interactive_plot,
        star_stack_b64=star_stack_b64, pixscale=args.pixscale,
    )
    if interactive_plot.exists():
        print(f"Saved interactive time-series plot to {interactive_plot}")


if __name__ == "__main__":
    main()
