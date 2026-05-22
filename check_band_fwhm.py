#!/usr/bin/env python3
"""Quick check: g vs r FWHM comparison."""
from astropy.table import Table
import numpy as np

t = Table.read('focus_output/focus_time_series.ecsv')

print("=== Per-band FWHM summary (pixels) ===")
for band in ['u', 'g', 'r', 'i', 'z']:
    mask = np.array([f == band for f in t['filter']])
    if mask.sum() == 0:
        continue
    fwhm = np.array(t['avg_fwhm'][mask])
    am = np.array(t['airmass'][mask])
    print(f"  {band}-band: N={mask.sum():3d}  "
          f"median FWHM={np.nanmedian(fwhm):.3f}  "
          f"mean FWHM={np.nanmean(fwhm):.3f}  "
          f"mean airmass={np.nanmean(am):.3f}")

print("\n=== Consecutive r↔g transitions (dt < 5 min) ===")
mjd = np.array(t['obs_time_mjd'])
filt = [str(f) for f in t['filter']]
fwhm = np.array(t['avg_fwhm'])

ratios = []
for i in range(len(t) - 1):
    dt_min = (mjd[i + 1] - mjd[i]) * 1440
    if dt_min < 5 and set([filt[i], filt[i + 1]]) == {'r', 'g'}:
        ri = i if filt[i] == 'r' else i + 1
        gi = i if filt[i] == 'g' else i + 1
        ratio = fwhm[gi] / fwhm[ri]
        ratios.append(ratio)
        print(f"  r: {t['sci_file'][ri]}  FWHM={fwhm[ri]:.3f}  |  "
              f"g: {t['sci_file'][gi]}  FWHM={fwhm[gi]:.3f}  "
              f"g/r={ratio:.3f}  dt={dt_min:.1f}min")

if ratios:
    print(f"\n  Mean g/r ratio = {np.mean(ratios):.3f}")

# Check if focus position differs between r and g
from astropy.io import fits
print("\n=== Focus position (LVDTC) per band ===")
for band in ['r', 'g']:
    mask = np.array([f == band for f in t['filter']])
    files = t['sci_file'][mask]
    focvals = []
    for fn in files:
        path = f"/Users/Jenny/projects/observation/bok/20251111/{fn}"
        hdr = fits.getheader(path)
        focvals.append(hdr.get('LVDTC', float('nan')))
    focvals = np.array(focvals)
    print(f"  {band}-band: mean focus={np.nanmean(focvals):.1f}  "
          f"std={np.nanstd(focvals):.1f}  "
          f"range=[{np.nanmin(focvals):.0f}, {np.nanmax(focvals):.0f}]")
