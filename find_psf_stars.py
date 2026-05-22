#!/usr/bin/env python3
"""
find_psf_stars.py
-----------------
Standalone PSF-star finder.

For each input FITS file, runs SEP source extraction on every science
extension (one per amplifier / HDU), collects all detections across all
files into a single pool, then runs the same 2-component GMM as
focus_pipeline.py to isolate the stellar locus.

Outputs
-------
* stars_catalog.fits  – star-only catalog with pixel coords, FWHM, flux,
                        ellipticity, and – when WCS is present – RA / Dec
* mag_vs_fwhm.png     – magnitude vs FWHM scatter plot coloured by image
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import sep


# ---------------------------------------------------------------------------
# Core helpers (copied verbatim from focus_pipeline.py)
# ---------------------------------------------------------------------------

def radial_profile(
    data: np.ndarray, center: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2).astype(int)
    r_max = int(r.max()) + 1
    radial_mean = np.array([data[r == i].mean() for i in range(r_max)])
    return np.arange(r_max), radial_mean


def gaussian(r, a, mu, sigma, c):
    return a * np.exp(-((r - mu) ** 2) / (2 * sigma**2)) + c


def sep_extract_one(
    data: np.ndarray,
    threshold: float = 3.0,
    cutout_size: int = 15,
    minarea: int = 20,
) -> Optional[Table]:
    """Run SEP on a single 2-D array; return per-object Table or None."""
    data = data.astype(np.float32, copy=False)
    sat_mask = data > (np.median(data) + 5 * np.std(data))
    bkg = sep.Background(data, bw=64, bh=64, fw=3, fh=3, mask=sat_mask)
    data_sub = data - bkg

    try:
        objects, _ = sep.extract(
            data_sub,
            thresh=threshold,
            err=bkg.globalrms,
            segmentation_map=True,
            minarea=minarea,
        )
    except Exception:
        return None

    if objects is None or len(objects) == 0:
        return None

    a_arr = np.array(objects["a"], dtype=float)
    b_arr = np.array(objects["b"], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        fwhm_sep = 2.355 * np.sqrt(a_arr * b_arr)
        e = np.where(np.isfinite(a_arr) & (a_arr != 0), 1.0 - b_arr / a_arr, np.nan)

    nobj = len(objects)
    fwhm_radial = np.full(nobj, np.nan)
    flux_peak_ratio = np.full(nobj, np.nan)

    for idx in range(nobj):
        xc = objects["x"][idx]
        yc = objects["y"][idx]
        if not (np.isfinite(xc) and np.isfinite(yc)):
            continue
        xi, yi = int(round(xc)), int(round(yc))
        ny, nx = data_sub.shape
        if yi - cutout_size < 0 or yi + cutout_size >= ny:
            continue
        if xi - cutout_size < 0 or xi + cutout_size >= nx:
            continue
        cutout = data_sub[yi - cutout_size : yi + cutout_size,
                          xi - cutout_size : xi + cutout_size]
        if not np.isfinite(cutout).all():
            cutout = np.nan_to_num(cutout, nan=0.0)
        rp_r, rp_flux = radial_profile(cutout, (cutout_size, cutout_size))
        ok = np.isfinite(rp_flux) & np.isfinite(rp_r)
        if ok.sum() < 4:
            continue
        try:
            baseline = np.median(rp_flux[ok][-5:]) if ok.sum() >= 5 else np.median(rp_flux[ok])
            amp0 = max(rp_flux[ok].max() - baseline, 0.0)
            popt, _ = curve_fit(
                gaussian, rp_r[ok], rp_flux[ok],
                p0=[amp0, 0.0, 2.0, baseline], maxfev=5000,
            )
            fwhm_radial[idx] = 2.355 * abs(popt[2])
        except Exception:
            continue
        peak = float(np.nanmax(cutout)) if np.isfinite(cutout).any() else 0.0
        if peak > 0 and np.isfinite(fwhm_radial[idx]):
            yy, xx = np.indices(cutout.shape)
            rr = np.sqrt((xx - cutout_size) ** 2 + (yy - cutout_size) ** 2)
            flux_peak_ratio[idx] = float(cutout[rr <= fwhm_radial[idx]].sum()) / peak

    tbl = Table(objects)
    tbl["FWHM"] = np.where(np.isfinite(fwhm_radial), fwhm_radial, fwhm_sep)
    tbl["e"] = e
    tbl["flux_ratio"] = flux_peak_ratio
    tbl["flux"] = np.array(objects["flux"], dtype=float)
    return tbl


def compute_gmm_labels(tbl: Table) -> np.ndarray:
    """
    2-component GMM on (FWHM, flux_ratio, instrumental_mag, e).
    Returns 0 for the stellar component, 1 for everything else.
    Identical logic to focus_pipeline.compute_gmm_labels.
    """
    fwhm = np.array(tbl["FWHM"], dtype=float)
    fr   = np.array(tbl["flux_ratio"], dtype=float)
    flux = np.array(tbl["flux"], dtype=float)
    e    = np.array(tbl["e"], dtype=float)
    mag  = -2.5 * np.log10(np.clip(flux, 1e-12, None))

    X = np.column_stack([fwhm, fr, mag, e])
    valid = np.all(np.isfinite(X), axis=1)
    if valid.sum() < 2:
        return np.ones(len(tbl), dtype=int)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X[valid])
    gmm = GaussianMixture(n_components=2, random_state=0)
    labels_valid = gmm.fit_predict(X_s)

    # Star cluster = component with smaller median FWHM
    star_label = min(
        np.unique(labels_valid),
        key=lambda lbl: np.nanmedian(fwhm[valid][labels_valid == lbl]),
    )
    out = np.ones(len(tbl), dtype=int)
    idx_valid = np.where(valid)[0]
    out[idx_valid[labels_valid == star_label]] = 0
    return out


# ---------------------------------------------------------------------------
# WCS helper
# ---------------------------------------------------------------------------

def pixel_to_radec(
    wcs: Optional[WCS], x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert pixel coords to RA/Dec using WCS; return NaN arrays if no WCS."""
    if wcs is None:
        nan = np.full(len(x), np.nan)
        return nan, nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sky = wcs.all_pix2world(np.column_stack([x, y]), 0)
    return sky[:, 0], sky[:, 1]


# ---------------------------------------------------------------------------
# Per-file extraction
# ---------------------------------------------------------------------------

def process_fits(
    path: Path,
    threshold: float,
    sci_extnames: Tuple[str, ...],
    sci_extname_prefix: str,
) -> Optional[Table]:
    """
    Extract detections from all matching HDUs in one FITS file.
    Returns a stacked Table with an added 'file' and 'hdu' column, or None.
    """
    tables: List[Table] = []
    with fits.open(str(path)) as hdul:
        for hdu in hdul:
            if hdu.data is None or hdu.data.ndim != 2:
                continue
            name = (hdu.name or "").upper().strip()
            # Accept HDUs whose name matches explicit list or prefix, or
            # any IMAGE extension when no filter is configured.
            if sci_extnames:
                if name not in sci_extnames and not name.startswith(sci_extname_prefix.upper()):
                    continue
            # Get WCS (best-effort)
            try:
                wcs = WCS(hdu.header, naxis=2)
                if not wcs.has_celestial:
                    wcs = None
            except Exception:
                wcs = None

            tbl = sep_extract_one(hdu.data, threshold=threshold)
            if tbl is None or len(tbl) == 0:
                continue

            ra, dec = pixel_to_radec(wcs, np.array(tbl["x"]), np.array(tbl["y"]))
            tbl["ra"]  = ra
            tbl["dec"] = dec
            tbl["file"] = str(path.name)
            tbl["hdu"]  = name if name else str(hdu.ver)
            tables.append(tbl)

    if not tables:
        return None
    return vstack(tables)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_mag_vs_fwhm(
    all_tbl: Table,
    star_mask: np.ndarray,
    output_path: Path,
    pixscale: float = 1.0,
) -> None:
    unit = "arcsec" if pixscale != 1.0 else "pixels"
    fwhm = np.array(all_tbl["FWHM"]) * pixscale
    flux = np.array(all_tbl["flux"])
    mag  = -2.5 * np.log10(np.clip(flux, 1e-12, None))
    files = np.array(all_tbl["file"])
    unique_files = list(dict.fromkeys(files))          # preserve order, unique
    cmap = plt.get_cmap("tab20")
    colors = {f: cmap(i % 20) for i, f in enumerate(unique_files)}

    fig, ax = plt.subplots(figsize=(8, 5))
    # Non-stars first (background)
    non = ~star_mask
    ax.scatter(mag[non], fwhm[non], s=6, alpha=0.3, color="silver", label="Non-star", zorder=1)
    # Stars coloured by file
    for fn in unique_files:
        sel = star_mask & (files == fn)
        if not np.any(sel):
            continue
        ax.scatter(
            mag[sel], fwhm[sel], s=14, alpha=0.7,
            color=colors[fn], label=fn, zorder=2,
        )

    ax.set_xlabel("Instrumental magnitude  (−2.5 log flux)")
    ax.set_ylabel(f"FWHM ({unit})")
    ax.set_title("Magnitude vs PSF FWHM  [GMM-selected stars]")
    # Compact legend (files only, not non-star if many)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) <= 20:
        ax.legend(fontsize=6, markerscale=1.5, loc="upper left",
                  bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find PSF stars in FITS images using GMM classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("fits_files", nargs="+", metavar="FITS",
                   help="Input FITS files (glob expansion handled by shell)")
    p.add_argument("--outdir", default=".", metavar="DIR",
                   help="Output directory for catalog and plot")
    p.add_argument("--threshold", type=float, default=3.0,
                   help="SEP extraction threshold in units of background RMS")
    p.add_argument("--minarea", type=int, default=20,
                   help="Minimum area (pixels) for SEP detections")
    p.add_argument("--pixscale", type=float, default=1.0,
                   help="Plate scale in arcsec/pixel (used only for FWHM axis label)")
    p.add_argument("--ext", nargs="*", default=[], metavar="NAME",
                   help="HDU extension names to process (e.g. SCI AMP1 AMP2). "
                        "By default all 2-D image HDUs are used.")
    p.add_argument("--ext-prefix", default="", metavar="PREFIX",
                   help="Accept HDUs whose name starts with this prefix (e.g. 'AMP')")
    p.add_argument("--catalog-name", default="stars_catalog.fits",
                   help="Output catalog filename")
    p.add_argument("--plot-name", default="mag_vs_fwhm.png",
                   help="Output plot filename")
    p.add_argument("--quality-fwhm-max", type=float, default=15.0,
                   help="Reject detections with FWHM > this value before GMM")
    p.add_argument("--quality-flag-max", type=int, default=0,
                   help="Reject detections with SEP flag > this value before GMM")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sci_extnames = tuple(n.upper() for n in args.ext) if args.ext else ()

    # ------------------------------------------------------------------
    # 1. Extract detections from every input file / HDU
    # ------------------------------------------------------------------
    all_tables: List[Table] = []
    for raw in args.fits_files:
        path = Path(raw)
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        print(f"Processing {path.name} …", end=" ", flush=True)
        tbl = process_fits(path, args.threshold, sci_extnames, args.ext_prefix)
        if tbl is None:
            print("no detections")
        else:
            print(f"{len(tbl)} detections")
            all_tables.append(tbl)

    if not all_tables:
        print("No detections in any input file. Exiting.")
        return

    combined = vstack(all_tables)
    print(f"\nTotal detections across all files: {len(combined)}")

    # ------------------------------------------------------------------
    # 2. Quality pre-cuts (same defaults as focus_pipeline)
    # ------------------------------------------------------------------
    fwhm_arr  = np.array(combined["FWHM"], dtype=float)
    flag_arr  = np.array(combined["flag"], dtype=int) if "flag" in combined.colnames else np.zeros(len(combined), dtype=int)
    pre_mask  = (fwhm_arr < args.quality_fwhm_max) & (flag_arr <= args.quality_flag_max)
    combined_cut = combined[pre_mask]
    print(f"After quality cuts: {len(combined_cut)} detections")

    if len(combined_cut) < 2:
        print("Too few detections after cuts. Exiting.")
        return

    # ------------------------------------------------------------------
    # 3. GMM across all targets combined
    # ------------------------------------------------------------------
    print("Running GMM …", end=" ", flush=True)
    labels = compute_gmm_labels(combined_cut)
    star_mask = labels == 0
    n_stars = int(star_mask.sum())
    print(f"{n_stars} stars / {len(combined_cut) - n_stars} non-stars")

    stars = combined_cut[star_mask]

    # ------------------------------------------------------------------
    # 4. Save catalog
    # ------------------------------------------------------------------
    # Keep a clean, useful column subset; add instrumental magnitude
    flux_s = np.array(stars["flux"], dtype=float)
    stars["mag_inst"] = -2.5 * np.log10(np.clip(flux_s, 1e-12, None))

    keep_cols = ["file", "hdu", "x", "y"]
    if "ra" in stars.colnames:
        keep_cols += ["ra", "dec"]
    keep_cols += ["FWHM", "e", "flux_ratio", "flux", "mag_inst"]
    # Add optional SEP shape columns if present
    for col in ("a", "b", "theta", "flag"):
        if col in stars.colnames:
            keep_cols.append(col)

    catalog = stars[keep_cols]
    catalog_path = outdir / args.catalog_name
    catalog.write(str(catalog_path), overwrite=True)
    print(f"Saved: {catalog_path}  ({len(catalog)} stars)")

    # ------------------------------------------------------------------
    # 5. Mag vs FWHM plot
    # ------------------------------------------------------------------
    plot_path = outdir / args.plot_name
    # Rebuild full-combined star_mask aligned to combined_cut indices
    plot_mag_vs_fwhm(combined_cut, star_mask, plot_path, pixscale=args.pixscale)


if __name__ == "__main__":
    main()
