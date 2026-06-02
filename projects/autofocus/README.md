# New90Prime Autofocus Dashboard

This directory contains a command-line focus pipeline and live dashboard for
New90Prime focus sequences.  The pipeline reduces raw FITS images, measures
per-amplifier PSF/FWHM statistics, fits the best focus, and can solve the
focal-plane tilt from the measured FWHM pattern.

The current example dashboard is here:

- [Interactive focus dashboard on GitHub Pages](https://jianiding.github.io/New90prime_Autofocus/projects/autofocus/)
- [Static focus time series](focus_output/focus_time_series.png)
- [Current tilt map](focus_output/tilt_map.png)

If the GitHub Pages link shows `ERR_CONNECTION_RESET`, try a different network
or phone hotspot; some networks block `github.io` even when `github.com` works.

In the dashboard, hover over a point to inspect the stacked PSF contour. Click a
point to open the tilt/focus solution with the per-amplifier FWHM detail table.
Use Plotly box/lasso select on the FWHM panel and click **Fit Selected** to fit
a best-focus value from chosen frames.

## Requirements

Install dependencies with:

```bash
python3 -m venv autofocus
source autofocus/bin/activate  # or `autofocus\Scripts\activate` on Windows
pip install -r requirements.txt
```

## Preparing Your Data

1. Place all relevant FITS files inside a single directory (e.g. `./20251111`).
2. Ensure filenames contain identifiable image numbers (e.g. `bs.OBJECT.0093.fits` → number `93`).
3. Provide bad-pixel masks per amplifier and filter under `./bad_pixel_masks` (or point `--mask-dir` elsewhere). The script expects files named `bad_pixel_mask_amp_<FILTER><AMP>.npy`, such as `bad_pixel_mask_amp_R1.npy`.

## Running the Pipeline

For a one-time command-line run on a single filter, provide the matching flat
numbers explicitly:

```bash
python focus_pipeline.py \
  --data-dir ./20251111 \
  --filter R \
  --bias-nums 11-13 \
  --flat-nums 41-43 \
  --sci-nums 93-96 \
  --mask-dir bad_pixel_masks \
  --outdir focus_output \
  --pixscale 0.455 \
  --solve-tilt
```

For a multi-band run, omit `--flat-nums`. The pipeline will classify flats by
their FITS `FILTER`/`FILTERS` and `OBJECT` headers, then build one master flat
per filter and amplifier:

```bash
python focus_pipeline.py \
  --data-dir ./20251111 \
  --filter all \
  --bias-nums 1-10 \
  --sci-nums 101-281 \
  --mask-dir bad_pixel_masks \
  --outdir focus_output \
  --pixscale 0.455 \
  --solve-tilt
```

`--dark-nums` is optional. Omit it when no dark correction is desired.

If you pass explicit `--flat-nums` in a multi-band run, those flat numbers must
include flats for every science band being processed. Bands with no matching
requested flats are skipped rather than calibrated with the wrong filter.

## Live Monitor

For observing, use the monitor. It watches for new science images, reruns the
pipeline, serves the dashboard, and uses incremental mode by default:

```bash
source autofocus/bin/activate

python realtime_focus_monitor.py \
  --data-dir ./20251111 \
  --bias-nums 1-10 \
  --name-contains object \
  --mask-dir bad_pixel_masks \
  --outdir focus_output \
  --date-key DATE-OBS \
  --pixscale 0.455 \
  --solve-tilt
```

Then open:

```text
http://localhost:8000/focus_time_series.html
```

For a completely fresh reprocess that ignores cached catalogs, use a new output
directory or add `--no-incremental`:

```bash
python realtime_focus_monitor.py \
  --data-dir ./20251111 \
  --bias-nums 1-10 \
  --name-contains object \
  --mask-dir bad_pixel_masks \
  --outdir focus_output_fresh \
  --date-key DATE-OBS \
  --pixscale 0.455 \
  --solve-tilt \
  --no-incremental
```

Incremental cache metadata includes the detection threshold, amp list,
calibration files, masks, active bands, and mask-generation settings. If those
inputs change, cached detections and master calibrations are cleared.

To preview the generated dashboard locally:

```bash
cd focus_output
python serve.py 8080
```

Then open `http://localhost:8080/focus_time_series.html`.

On GitHub Pages the dashboard is static: hover thumbnails and click-through
tilt maps work from committed output files. Live server actions such as
**Fit Selected** and on-demand per-amplifier lookup require
`realtime_focus_monitor.py` to be running locally.

You can mix single numbers and ranges (e.g. `--sci-nums 90 93-96 101`). Ranges may be ascending or descending.

Add `--auto-generate-masks` to let the script build 2-D bad-pixel maps from the master flats on the fly (saved beneath `--mask-dir`). Tweak the thresholds with `--mask-sat-mult`, `--mask-black-mult`, `--mask-sat-frac`, and `--mask-black-frac` if you need stricter or looser masking. Use `--skip-reduced` to explicitly suppress writing reduced FITS products even if `--write-reduced` appears elsewhere in your workflow.

### Argument Reference

| Argument | Required | Description |
| --- | --- | --- |
| `--data-dir` | No (default `.`) | Folder containing all FITS files. |
| `--filter` | Yes | Photometric filter label, comma-separated list, or `all`. |
| `--bias-nums` | Yes | Bias image numbers; numbers are matched against filenames. |
| `--dark-nums` | No | Optional dark image numbers. Omit if no dark correction is needed. |
| `--flat-nums` | No for multi-band auto mode | Flat image numbers. Omit in `--filter all` mode to auto-discover flats by filter. |
| `--sci-nums` | Yes | Science image numbers to include in the focus curve. |
| `--focus-key` | No (default `LVDTC`) | FITS header keyword that stores the focus position. |
| `--amps` | No (default `1 2 3 4 5 6 7 8`) | Amplifier IDs (HDU numbers) to analyze. |
| `--threshold` | No (default `25`) | SEP detection threshold. |
| `--mask-dir` | No (default `bad_pixel_masks`) | Directory containing the bad-pixel mask `.npy` files. |
| `--outdir` | No (default `focus_output`) | Directory for reduced products and plots. |
| `--write-reduced` / `--skip-reduced` | Optional flags | Force writing (or skipping) `_amp*_reduced.fits` files for each science image and amplifier. Default behavior skips writing unless `--write-reduced` is present. |

### What the Script Produces

- `focus_output/focus_sources.fits`: merged catalog of detected sources across all amps and science frames.
- `focus_output/focus_fit.png`: scatter + fitted parabola plus residuals.
- `focus_output/focus_time_series.html`: interactive dashboard with hover PSF contours.
- `focus_output/focus_time_series.ecsv`: per-exposure FWHM, ellipticity, filter, airmass, and focus table.
- `focus_output/tilt_map_<frame>.png`: tilt/focus solution figure for clicked frames.
- `focus_output/tilt_result_<frame>.json`: per-frame tilt/focus fit values and actuator correction summary.
- `focus_output/focus_per_amp_points.ecsv`: per-amplifier FWHM table created by the selected-frame/per-amp fitting workflow.
- `focus_output/focus_per_amp_best_focus.ecsv`: per-amp best-focus fit summary.
- Console summary of quadratic coefficients and best-focus estimate.
- (Optional) Per-amplifier reduced FITS files when `--write-reduced` is used.

## Per-Amp and Repeat-Trial Summaries

After the monitor has produced `focus_sources.fits` and `focus_time_series.ecsv`,
create a per-amplifier table for selected frames:

```bash
python fit_focus_per_amp_from_monitor.py \
  --sources focus_output/focus_sources.fits \
  --time-series focus_output/focus_time_series.ecsv \
  --fit-nums 150-154 \
  --outdir focus_output
```

Average repeated exposures by amplifier and make an averaged tilt map:

```bash
python repeated_tilt_trials.py \
  --points-ecsv focus_output/focus_per_amp_points.ecsv \
  --average-images 150-154 \
  --average-label repeat_150_154 \
  --average-outdir focus_output
```

This writes files such as:

- `average_fwhm_per_amp_<label>.csv`
- `average_tilt_result_<label>.json`
- `average_tilt_map_<label>.png`

## Troubleshooting Tips

- If the script reports missing masks, either supply the `.npy` files or remove the warning by creating zero masks with the expected filenames.
- Ensure at least three science frames have valid focus header values; the parabola fit needs ≥3 points.
- Set `--threshold` lower if no sources are found, but expect more spurious detections.
- If per-amplifier details do not appear when clicking a point, open the
  dashboard through the running monitor server, not by double-clicking the HTML
  file. The live endpoint is needed for on-demand amp FWHM lookup.
- If explicit `--flat-nums` skips a band, either include flat numbers for that
  band or omit `--flat-nums` so multi-band flats are auto-discovered.
- If results look stale, use a fresh `--outdir` or `--no-incremental`.

For additional customization, edit `focus_pipeline.py` directly—its logic mirrors the original notebook but with CLI parameterization.
