# ccdtools

Work in progress: New 90 Inch Reduction Pipeline:
1. ccdtools provides simple CCD data reduction helpers and a CLI for detecting/masking bad columns as well as a focus-quality analysis and GUI per amplifier.

## Features

- Scan a directory of observation data (FITS files) and categorize bias/dark/flat frames 
- Build master bias/dark/flat per amplifier
- Detect bad columns by saturation/black thresholds
- Produce per-band, per-amp bad pixel maps and masked FITS outputs
- Analyse stellar FWHM / ellipticity per amplifier using SEP + GMM clustering
- Optional Tk-based GUI to compare amplifier statistics
- CLI with modes for bad-pixel masking or focus analysis

## Installation (development / editable)

It's recommended to use a virtual environment.

```bash
cd /path/to/project/root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
```

This creates a console script `ccdtools` and lets you import the package in Python.

## CLI usage

### Prerequisites

- Activate the virtual environment you installed `ccdtools` into (see Installation above).
- Change into the repository root so relative defaults like `./20250427/` resolve correctly:

	```bash
	cd /Users/Jenny/projects/observation/bok/new90prime_reduction
	```

- Organize your nightly FITS under a directory that contains bias/dark/flat/science frames, for example `/Users/Jenny/projects/observation/bok/20250427`.

All examples below use the Python module entry point so they work even if the console script is not on your `PATH`.

### Focus / amplifier quality analysis (amps 1–8 only)

GUI run for the R-band science frame `bs.OBJECT.0079.fits` (only HDUs 1–8 are analysed automatically):

```bash
python -m ccdtools.cli --mode focus \
	--directory /Users/Jenny/projects/observation/bok/20250427 \
	--focus-band R \
	--focus-science-files /Users/Jenny/projects/observation/bok/20250427/bs.OBJECT.0079.fits
```

Headless batch run with summary export and no GUI:

```bash
python -m ccdtools.cli --mode focus \
	--directory /Users/Jenny/projects/observation/bok/20250427 \
	--focus-band R \
	--focus-no-gui \
	--focus-export ./focus_summary.csv
```

Provide explicit amplifier indices (any outside 1–8 are skipped with a warning):

```bash
python -m ccdtools.cli --mode focus \
	--directory /Users/Jenny/projects/observation/bok/20250427 \
	--focus-band R \
	--focus-amp-nums 1 2 5
```

Key focus options:

- `--focus-science-files` bypasses auto-selection and attempts the files in order.
- `--focus-threshold`, `--focus-cutout`, `--focus-max-fwhm`, `--focus-min-flux-ratio`, `--focus-max-e` tune candidate detection.
- `--focus-sf` / `--focus-bf` control the bad-column fractions (defaults 0.25 / 0.2 to mirror the original notebook mask).
- `--focus-sat-sigma` / `--focus-black-sigma` set the saturation/black thresholds as `median ± sigma·std` (defaults 1.0 / 4.0).
- `--focus-write-regions --focus-region-dir ./regions` emits DS9 ellipse overlays.
- `--focus-no-gui` skips the Tk window (useful on headless systems).
- `--focus-export` writes the per-amplifier summary CSV.

### Bad pixel masking

Automatic thresholds with plots suppressed:

```bash
python -m ccdtools.cli --mode badpixels \
	--directory /Users/Jenny/projects/observation/bok/20250427 \
	--bands-to-try U,R,Z \
	--amp-num 2 \
	--sf 0.5 \
	--bf 0.25 \
	--outdir ./test_mask \
	--no-show
```

Explicit thresholds and interactive plots (omit `--no-show`):

```bash
python -m ccdtools.cli --mode badpixels \
	--directory /Users/Jenny/projects/observation/bok/20250427 \
	--bands-to-try U,R,Z \
	--amp-num 2 \
	--sat-thresh 12345 \
	--black-thresh 100 \
	--outdir ./test_mask
```

## API quick-start (python)

```python
from ccdtools.file_utils import skim_fits_files
from ccdtools.utilities import diff_amp, flat_reduction_b
from ccdtools.bad_pixel_mask import find_bad_columns, process_and_save_calibrated_image

categorized = skim_fits_files(directory='./20250427/', Keyword='OBJECT', target_bands=['u','r','z'])
# pick appropriate band
flats = categorized['U']['flat']
amp = 2
biases, darks, flats_list, sciences = diff_amp(amp, categorized['bias_frames'], categorized['dark_frames'], flats, categorized['U']['other'][:1])
Master_bias, Master_dark, Unbiased_dark, Master_flat = flat_reduction_b(biases, darks, flats_list)
bad_cols = find_bad_columns(Master_flat, np.median(Master_flat)+0.5*np.std(Master_flat), np.median(Master_flat)-np.std(Master_flat), 0.5, 0.25)
process_and_save_calibrated_image(Master_flat, bad_cols, amp, './test_mask/')
```

## Notes

- The FITS scanner may need tuning for your headers.
- The CLI computes thresholds automatically by default (median +/- std per band), but you can override.

## Contributing / GitHub

To publish on GitHub:

1. Create a new repository on GitHub (via web or `gh repo create`).
2. Add the remote and push:

```bash
git init
git add .
git commit -m "Initial commit: ccdtools"
git remote add origin git@github.com:YOUR_USER/YOUR_REPO.git
git branch -M main
git push -u origin main
```

Or, use the GitHub CLI (`gh`) to create & push in one step:

```bash
gh repo create YOUR_USER/ccdtools --public --source=. --remote=origin --push
```