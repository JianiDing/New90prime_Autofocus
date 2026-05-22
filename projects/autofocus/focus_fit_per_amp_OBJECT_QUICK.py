import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from pathlib import Path
from focus_pipeline import (
    skim_fits_files, select_files_by_numbers, diff_amp, flat_reduction_b,
    sep_run_2, compute_gmm_labels, fit_parabola_vertex_form
)

DATA_DIR = "/Users/Jenny/projects/observation/bok/20251111"
IMAGE_NUMS = range(150, 157)
BAND = "R"
N_AMPS = 8
ECSV_PATH = os.path.join("focus_output", "focus_time_series.ecsv")

# Read focus positions from ECSV
from astropy.table import Table as AstroTable
ecsv = AstroTable.read(ECSV_PATH, format="ascii.ecsv")
ecsv_map = {row["sci_file"]: row["focus_position"] for row in ecsv if row["filter"].lower() == BAND.lower()}

cat = skim_fits_files(DATA_DIR, target_bands=(BAND,))
bias_files = cat["bias_frames"]
dark_files = cat["dark_frames"]

flat_files = cat[BAND]["flat"]
object_files = cat[BAND]["other"]
selected_files = select_files_by_numbers(object_files, IMAGE_NUMS)

# Use explicit calibration file numbers as in the monitor
BIAS_NUMS = range(1, 11)
DARK_NUMS = range(21, 23)
FLAT_NUMS = range(91, 101)
from focus_pipeline import expand_image_numbers, select_files_by_numbers
bias_files = select_files_by_numbers(cat["bias_frames"], BIAS_NUMS)
dark_files = select_files_by_numbers(cat["dark_frames"], DARK_NUMS)
flat_files = select_files_by_numbers(cat[BAND]["flat"], FLAT_NUMS)

# Build master calibration frames for each amp
master_bias = {}
master_dark = {}
master_flat = {}

for amp in range(1, N_AMPS+1):
    biases, darks, flats, _ = diff_amp(amp, bias_files, dark_files, flat_files, [])
    mb, md, _, mf = flat_reduction_b(biases, darks, flats)
    master_bias[amp] = mb
    master_dark[amp] = md
    master_flat[amp] = mf


focus_positions = []
fwhm_per_amp = {amp: [] for amp in range(1, N_AMPS+1)}
used_files = []

for path in selected_files:
    fname = os.path.basename(path)
    focus = ecsv_map.get(fname)
    if focus is None:
        print(f"No focus position in {fname}")
        continue
    focus_positions.append(float(focus))
    with fits.open(path) as hdul:
        for amp in range(1, N_AMPS+1):
            # Find extension for this amp
            for ext, h in enumerate(hdul):
                extname = h.header.get("EXTNAME", "")
                if extname.endswith(str(amp)) and h.data is not None:
                    sci = h.data.astype(float)
                    # Quick reduction: (sci - bias) / (flat - bias), avoid division by zero
                    mb = master_bias[amp]
                    mf = master_flat[amp]
                    denom = mf - mb
                    denom_safe = np.where(np.abs(denom) < 1e-6, np.nan, denom)
                    reduced = (sci - mb) / denom_safe
                    # Print statistics of reduced image and denominator
                    print(f"Frame {fname} amp {amp}: denom stats min={np.nanmin(denom):.2f} max={np.nanmax(denom):.2f} mean={np.nanmean(denom):.2f} std={np.nanstd(denom):.2f}")
                    print(f"Frame {fname} amp {amp}: reduced stats min={np.nanmin(reduced):.2f} max={np.nanmax(reduced):.2f} mean={np.nanmean(reduced):.2f} std={np.nanstd(reduced):.2f}")
                    # Run SEP
                    data_sub, objects, fwhm_sep, fwhm_radial, flux_in_fwhm, flux_peak_ratio, e, _ = sep_run_2(
                        reduced, threshold=3.0, cutout_size=15, minarea=20, verbose=False)
                    if objects is None or len(objects) == 0:
                        print(f"Frame {fname} amp {amp}: No objects found by SEP.")
                        fwhm_per_amp[amp].append(np.nan)
                        break
                    tbl = Table()
                    tbl["FWHM"] = fwhm_radial
                    tbl["flux_ratio"] = flux_in_fwhm / np.nanmax(reduced) if np.nanmax(reduced) > 0 else np.nan
                    tbl["flux"] = flux_in_fwhm
                    tbl["e"] = e
                    gmm_labels = compute_gmm_labels(tbl)
                    star_mask = (gmm_labels == 0) & np.isfinite(tbl["FWHM"])
                    print(f"Frame {fname} amp {amp}: GMM stars = {np.sum(star_mask)} (total objects: {len(tbl['FWHM'])})")
                    fwhm_per_amp[amp].append(median_fwhm)
                    break
            else:
                fwhm_per_amp[amp].append(np.nan)

focus_positions = np.array(focus_positions)


for amp in range(1, N_AMPS+1):
    fwhms = np.array(fwhm_per_amp[amp])
    mask = np.isfinite(fwhms)
    if np.count_nonzero(mask) < 3:
        print(f"Amp {amp}: not enough data for fit")
        continue
    x = focus_positions[mask]
    y = fwhms[mask]
    fit = fit_parabola_vertex_form(x, y)
    print(f"Amp {amp}: best focus = {fit['h']:.2f}, min FWHM = {fit['k']:.3f}")
    plt.figure()
    plt.scatter(x, y, label="Median FWHM (GMM stars)")
    xx = np.linspace(x.min(), x.max(), 100)
    yy = fit['A'] * (xx - fit['h']) ** 2 + fit['k']
    plt.plot(xx, yy, label="Parabola fit")
    plt.axvline(fit['h'], color="green", linestyle="--", label="Best focus")
    plt.xlabel("Focus position")
    plt.ylabel("Median FWHM (pix)")
    plt.title(f"Amp {amp} focus fit (OBJECT frames {IMAGE_NUMS.start}-{IMAGE_NUMS.stop-1})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"focus_fit_OBJECT_QUICK_amp{amp}.png")
    plt.close()
