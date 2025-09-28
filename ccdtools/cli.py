"""Command-line interface for ccdtools pipeline."""
import argparse
import os
import sys
import numpy as np

from .utilities import diff_amp, flat_reduction_b
from .bad_pixel_mask import find_bad_columns, process_and_save_calibrated_image
from .file_utils import skim_fits_files, get_band_frames


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run bad-pixel detection pipeline")
    p.add_argument("--directory", default="./20250427/", help="directory with FITS files")
    p.add_argument("--Keyword", default="OBJECT", help="keyword for skimming fits files")
    p.add_argument("--target-bands", nargs='+', default=['u', 'r', 'z'], help="target bands to include")
    p.add_argument("--exclusion-list", nargs='*', default=[], help="files to exclude")
    p.add_argument("--amp-num", type=int, default=2, help="amplifier HDU index to process")
    p.add_argument("--sf", type=float, default=0.5, help="saturated-fraction threshold (e.g. 0.5)")
    p.add_argument("--bf", type=float, default=0.25, help="black-fraction threshold (e.g. 0.25)")
    p.add_argument("--outdir", default='./test_mask', help="output directory for masks")
    p.add_argument("--no-show", action='store_true', help="disable plotting (batch mode)")
    p.add_argument("--sat-thresh", type=float, default=None, help="explicit saturation threshold (overrides auto)")
    p.add_argument("--black-thresh", type=float, default=None, help="explicit black column threshold (overrides auto)")
    p.add_argument("--flat-band", default='U', help="which band key to use for flats (e.g. U)")
    p.add_argument("--science-band", default='U', help="which band key to use for science/other frames (e.g. U)")
    p.add_argument("--bands-to-try", default=None, help="comma-separated list of bands to try in order (e.g. U,R,Z)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    # `skim_fits_files` is expected to be a user-provided function in the project root.
    # use our bundled skim_fits_files
    categorized_files = skim_fits_files(directory=args.directory, Keyword=args.Keyword, target_bands=args.target_bands, exclusion_list=args.exclusion_list)

    # Prepare list of bands to try (in order)
    if args.bands_to_try:
        bands = [b.strip() for b in args.bands_to_try.split(',') if b.strip()]
    else:
        bands = [args.flat_band]

    biases = categorized_files.get('bias_frames', [])
    darks = categorized_files.get('dark_frames', [])

    if not (biases and darks):
        print('Missing required frame lists (bias/dark). Aborting.')
        return 4

    overall_bad = {}
    for band in bands:
        print(f"\nProcessing band: {band}")
        flats = get_band_frames(categorized_files, band, kind='flat')
        if not flats:
            print(f'  No flats found for band {band}; skipping.')
            continue

        sciences = get_band_frames(categorized_files, band, kind='other')[:1]

        biases_list, darks_list, flats_list, sciences_list = diff_amp(args.amp_num, biases, darks, flats, sciences)

        try:
            Master_bias, Master_dark, Unbiased_dark, Master_flat = flat_reduction_b(biases_list, darks_list, flats_list, plot_master_bias=False, plot_unbias_dark=False, plot_master_flat=False)
        except Exception as e:
            print(f'  Failed to compute masters for band {band}: {e}; skipping.')
            continue

        # thresholds (user override or automatic per requested formula)
        auto_sat = float(np.median(Master_flat) + 0.5 * np.std(Master_flat))
        auto_black = float(np.median(Master_flat) - 1.0 * np.std(Master_flat))
        sat_thresh = args.sat_thresh if args.sat_thresh is not None else auto_sat
        black_thresh = args.black_thresh if args.black_thresh is not None else auto_black

        bad_columns = find_bad_columns(Master_flat, saturation_threshold=sat_thresh, black_column_threshold=black_thresh, sf=args.sf, bf=args.bf)

        # per-band output directory so each band+amp produces unique masks
        band_out = os.path.join(args.outdir, str(band))
        os.makedirs(band_out, exist_ok=True)

        show_plots = not args.no_show
        bad_map, masked = process_and_save_calibrated_image(Master_flat, bad_columns, args.amp_num, band_out, show_plots=show_plots)
        overall_bad[band] = bad_columns
        print(f'  Done for band {band} — bad columns: {bad_columns}')

    print('\nAll done. Summary of bad columns per band:')
    for b, cols in overall_bad.items():
        print(f'  {b}: {cols}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
