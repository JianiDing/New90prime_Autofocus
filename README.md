# ccdtools

Work in progress: New 90 Inch Reduction Pipeline:
1. ccdtools provides simple CCD data reduction helpers and a small CLI for detecting and masking bad columns in amplifier-specific images. 

## Features

- Scan a directory of observation data (FITS files) and categorize bias/dark/flat frames 
- Build master bias/dark/flat per amplifier
- Detect bad columns by saturation/black thresholds
- Produce per-band, per-amp bad pixel maps and masked FITS outputs
- CLI with options for thresholds, bands to try, and batch mode

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

Basic example (uses automatic thresholds):

```bash
python -m ccdtools.cli --directory ./20250427/ --bands-to-try U,R,Z --amp-num 2 --sf 0.5 --bf 0.25 --outdir ./test_mask --no-show
```

Override thresholds explicitly:

```bash
ccdtools --directory ./20250427/ --bands-to-try U,R,Z --amp-num 2 --sat-thresh 12345 --black-thresh 100 --outdir ./test_mask --no-show
```

Run interactively (show plots): omit `--no-show`.

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
- The CLI computes thresholds automatically by default (median +0.5*std /-std per band), but you can override.

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
