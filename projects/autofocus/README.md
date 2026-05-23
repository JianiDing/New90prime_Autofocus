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

## Running the Script

Basic command:

```bash
python focus_pipeline.py \
  --data-dir ./20251111 \
  --filter R \
  --bias-nums 11-13 \
  --dark-nums 21-22 \
  --flat-nums 41-43 \
  --sci-nums 93-96 \
  --mask-dir bad_pixel_masks \
  --outdir focus_output \
  --write-reduced
```

For tilt solving and the interactive dashboard:

```bash
python focus_pipeline.py \
  --data-dir ./20251111 \
  --filter R \
  --bias-nums 11-13 \
  --dark-nums 21-22 \
  --flat-nums 41-43 \
  --sci-nums 93-96 \
  --mask-dir bad_pixel_masks \
  --outdir focus_output \
  --pixscale 0.455 \
  --solve-tilt
```

To preview the generated dashboard locally:

```bash
cd focus_output
python serve.py 8080
```

Then open `http://localhost:8080/focus_time_series.html`.

On GitHub Pages the dashboard is static: hover thumbnails, click-through tilt
maps, and per-amplifier tables work from the committed output files. Live
server actions are intentionally omitted from the shared dashboard.

You can mix single numbers and ranges (e.g. `--sci-nums 90 93-96 101`). Ranges may be ascending or descending.

Add `--auto-generate-masks` to let the script build 2-D bad-pixel maps from the master flats on the fly (saved beneath `--mask-dir`). Tweak the thresholds with `--mask-sat-mult`, `--mask-black-mult`, `--mask-sat-frac`, and `--mask-black-frac` if you need stricter or looser masking. Use `--skip-reduced` to explicitly suppress writing reduced FITS products even if `--write-reduced` appears elsewhere in your workflow.

### Argument Reference

| Argument | Required | Description |
| --- | --- | --- |
| `--data-dir` | No (default `.`) | Folder containing all FITS files. |
| `--filter` | Yes | Photometric filter label (e.g. `R`, `G`). |
| `--bias-nums` `--dark-nums` `--flat-nums` | Yes | Image numbers for calibration frames; numbers are matched against filenames. |
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
- `focus_output/focus_per_amp_points.ecsv`: per-amplifier FWHM table used by the dashboard.
- `focus_output/tilt_map_<frame>.png`: tilt/focus solution figure for clicked frames.
- Console summary of quadratic coefficients and best-focus estimate.
- (Optional) Per-amplifier reduced FITS files when `--write-reduced` is used.

## Troubleshooting Tips

- If the script reports missing masks, either supply the `.npy` files or remove the warning by creating zero masks with the expected filenames.
- Ensure at least three science frames have valid focus header values; the parabola fit needs ≥3 points.
- Set `--threshold` lower if no sources are found, but expect more spurious detections.

For additional customization, edit `focus_pipeline.py` directly—its logic mirrors the original notebook but with CLI parameterization.
