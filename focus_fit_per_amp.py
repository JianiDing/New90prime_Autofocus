#!/usr/bin/env python3
"""
Fit focus curve for each amplifier separately for a set of r-band images.

- Uses images 150-156 in the specified directory
- Fits a parabola to median FWHM vs focus position for each amp
- Plots and prints best focus for each amp
"""
import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Parameters ---
DATA_DIR = "/Users/Jenny/projects/observation/bok/20251111"
IMAGE_NUMS = range(150, 157)  # inclusive
BAND = "R"
N_AMPS = 8

# --- Helper: fit parabola vertex form ---
def fit_parabola_vertex(x, y):
    def vertex_parabola(x, A, h, k):
        return A * (x - h) ** 2 + k
    coeffs = np.polyfit(x, y, 2)
    a0, b0, c0 = coeffs
    A0 = a0
    h0 = -b0 / (2 * a0) if abs(a0) > 1e-12 else x[np.argmin(y)]
    k0 = a0 * h0 * h0 + b0 * h0 + c0
    p0 = [A0, h0, k0]
    popt, pcov = curve_fit(vertex_parabola, x, y, p0=p0)
    A, h, k = popt
    return dict(A=A, h=h, k=k, popt=popt, pcov=pcov)

# --- Main ---
focus_positions = []
fwhm_per_amp = {amp: [] for amp in range(1, N_AMPS+1)}

for num in IMAGE_NUMS:
    # Find file
    for ext in (".fits", ".fit"):
        fname = f"r{num:04d}{ext}"
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            break
    else:
        print(f"Missing file for image {num}")
        continue
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        # Try to get focus position from header
        focus = hdr.get("FOCUSPOS") or hdr.get("FOCUS")
        if focus is None:
            print(f"No focus position in {fname}")
            continue
        focus_positions.append(float(focus))
        for amp in range(1, N_AMPS+1):
            # Try to find extension with EXTNAME ending in amp number
            for h in hdul:
                extname = h.header.get("EXTNAME", "")
                if extname.endswith(str(amp)) and h.data is not None:
                    data = h.data.astype(float)
                    # Estimate FWHM using SEP or a simple method
                    # Here: use stddev as a placeholder (replace with SEP for real stars)
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
    fit = fit_parabola_vertex(x, y)
    print(f"Amp {amp}: best focus = {fit['h']:.2f}, min FWHM = {fit['k']:.3f}")
    plt.figure()
    plt.scatter(x, y, label="Median FWHM")
    xx = np.linspace(x.min(), x.max(), 100)
    yy = fit['A'] * (xx - fit['h']) ** 2 + fit['k']
    plt.plot(xx, yy, label="Parabola fit")
    plt.axvline(fit['h'], color="green", linestyle="--", label="Best focus")
    plt.xlabel("Focus position")
    plt.ylabel("FWHM (pix, rough)")
    plt.title(f"Amp {amp} focus fit (r-band images {IMAGE_NUMS.start}-{IMAGE_NUMS.stop-1})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"focus_fit_amp{amp}.png")
    plt.close()
