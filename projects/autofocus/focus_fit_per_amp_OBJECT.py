#!/usr/bin/env python3
"""
Fit focus curve for each amplifier separately for a set of OBJECT frames (e.g., bs.OBJECT.0277.fits) in r-band.
- Uses image numbers 150-156 (or any range you specify)
- Follows the pipeline logic: skims files, selects by number, extracts focus, fits per-amp
- Plots and prints best focus for each amp
"""
import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path
from focus_pipeline import skim_fits_files, select_files_by_numbers, fit_parabola_vertex_form

# --- Parameters ---
DATA_DIR = "/Users/Jenny/projects/observation/bok/20251111"
IMAGE_NUMS = range(150, 157)  # inclusive
BAND = "R"
N_AMPS = 8

# --- Skim and select files ---
cat = skim_fits_files(DATA_DIR, target_bands=(BAND,))
object_files = cat[BAND]["other"]
selected_files = select_files_by_numbers(object_files, IMAGE_NUMS)

focus_positions = []
fwhm_per_amp = {amp: [] for amp in range(1, N_AMPS+1)}
used_files = []

for path in selected_files:
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        focus = hdr.get("FOCUSPOS") or hdr.get("FOCUS")
        if focus is None:
            print(f"No focus position in {os.path.basename(path)}")
            continue
        focus_positions.append(float(focus))
        used_files.append(path)
        for amp in range(1, N_AMPS+1):
            for h in hdul:
                extname = h.header.get("EXTNAME", "")
                if extname.endswith(str(amp)) and h.data is not None:
                    data = h.data.astype(float)
                    # Estimate FWHM using stddev as placeholder
                    fwhm = np.std(data)
                    fwhm_per_amp[amp].append(fwhm)
                    break
            else:
                fwhm_per_amp[amp].append(np.nan)

focus_positions = np.array(focus_positions)

# --- Fit and plot for each amp ---
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
    plt.scatter(x, y, label="Median FWHM")
    xx = np.linspace(x.min(), x.max(), 100)
    yy = fit['A'] * (xx - fit['h']) ** 2 + fit['k']
    plt.plot(xx, yy, label="Parabola fit")
    plt.axvline(fit['h'], color="green", linestyle="--", label="Best focus")
    plt.xlabel("Focus position")
    plt.ylabel("FWHM (pix, rough)")
    plt.title(f"Amp {amp} focus fit (OBJECT frames {IMAGE_NUMS.start}-{IMAGE_NUMS.stop-1})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"focus_fit_OBJECT_amp{amp}.png")
    plt.close()
