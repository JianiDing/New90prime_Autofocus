"""Command-line interface for ccdtools pipeline."""
import argparse
import csv
import os
from pathlib import Path
from typing import List
import sys
import numpy as np

from .utilities import diff_amp, flat_reduction_b
from .bad_pixel_mask import find_bad_columns, process_and_save_calibrated_image
from .file_utils import skim_fits_files, get_band_frames
from .focus import (
    FocusConfig,
    aggregate_results,
    analyze_amplifier,
    launch_gui,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CCD tools command-line utilities")
    p.add_argument("--directory", default="./20250427/", help="directory with FITS files")
    p.add_argument("--Keyword", default="OBJECT", help="keyword for skimming fits files")
    p.add_argument("--target-bands", nargs='+', default=['u', 'r', 'z'], help="target bands to include")
    p.add_argument("--exclusion-list", nargs='*', default=[], help="files to exclude")
    p.add_argument("--amp-num", type=int, default=2, help="amplifier HDU index to process")
    p.add_argument("--sf", type=float, default=0.25, help="saturated-fraction threshold (e.g. 0.5)")
    p.add_argument("--bf", type=float, default=0.2, help="black-fraction threshold (e.g. 0.25)")
    p.add_argument("--outdir", default='./test_mask', help="output directory for masks")
    p.add_argument("--no-show", action='store_true', help="disable plotting (batch mode)")
    p.add_argument("--sat-thresh", type=float, default=None, help="explicit saturation threshold (overrides auto)")
    p.add_argument("--black-thresh", type=float, default=None, help="explicit black column threshold (overrides auto)")
    p.add_argument("--flat-band", default='U', help="which band key to use for flats (e.g. U)")
    p.add_argument("--science-band", default='U', help="which band key to use for science/other frames (e.g. U)")
    p.add_argument("--bands-to-try", default=None, help="comma-separated list of bands to try in order (e.g. U,R,Z)")
    p.add_argument("--mode", choices=["badpixels", "focus"], default="badpixels", help="which pipeline to run")

    # focus-specific options (namespaced to avoid clashes)
    p.add_argument("--focus-band", default='Z', help="band key to use for focus analysis")
    p.add_argument("--focus-science-count", type=int, default=1, help="number of science frames to analyse")
    p.add_argument("--focus-science-files", nargs='+', default=None, help="explicit science frame paths (overrides auto selection)")
    p.add_argument("--focus-amp-nums", nargs='+', type=int, default=None, help="explicit amplifier HDU indices (e.g. 1 2 3 4)")
    p.add_argument("--focus-threshold", type=float, default=5.0, help="SEP detection threshold (sigma)")
    p.add_argument("--focus-cutout", type=int, default=15, help="half-size for radial profile cutouts")
    p.add_argument("--focus-max-fwhm", type=float, default=15.0, help="maximum FWHM allowed for candidates")
    p.add_argument("--focus-min-flux-ratio", type=float, default=2.0, help="minimum flux-in-FWHM / peak ratio for candidates")
    p.add_argument("--focus-max-e", type=float, default=0.9, help="maximum ellipticity (e) for candidates")
    p.add_argument("--focus-sf", type=float, default=0.25, help="saturated-column fraction threshold for bad column mask")
    p.add_argument("--focus-bf", type=float, default=0.2, help="black-column fraction threshold for bad column mask")
    p.add_argument("--focus-sat-sigma", type=float, default=1.0, help="sigma multiplier for saturation threshold")
    p.add_argument("--focus-black-sigma", type=float, default=4.0, help="sigma multiplier for black threshold")
    p.add_argument("--focus-min-samples", type=int, default=12, help="minimum candidates needed for GMM clustering")
    p.add_argument("--focus-random-state", type=int, default=0, help="random seed for GMM")
    p.add_argument("--focus-no-gui", action='store_true', help="skip launching the GUI (useful on headless systems)")
    p.add_argument("--focus-write-regions", action='store_true', help="write DS9 region files for each amplifier")
    p.add_argument("--focus-region-dir", default=None, help="directory for DS9 region outputs")
    p.add_argument("--focus-export", default=None, help="optional CSV path to export summary metrics")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "focus":
        return run_focus(args)
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


def _infer_amp_indices(science_file: str) -> List[int]:
    from astropy.io import fits

    with fits.open(science_file, memmap=False) as hdul:
        # skip primary HDU (0)
        return [idx for idx in range(1, len(hdul)) if 1 <= idx <= 8]


def run_focus(args) -> int:
    categorized_files = skim_fits_files(directory=args.directory, Keyword=args.Keyword, target_bands=args.target_bands, exclusion_list=args.exclusion_list)

    biases = categorized_files.get('bias_frames', [])
    darks = categorized_files.get('dark_frames', [])
    if not biases or not darks:
        print('Missing required frame lists (bias/dark). Aborting.')
        return 4

    band = args.focus_band.upper()
    flats = get_band_frames(categorized_files, band, kind='flat')

    if args.focus_science_files:
        sciences = [str(Path(p)) for p in args.focus_science_files]
        missing = [p for p in sciences if not Path(p).is_file()]
        if missing:
            print("The following science files could not be found:")
            for miss in missing:
                print(f"  {miss}")
            return 8
    else:
        sciences = get_band_frames(categorized_files, band, kind='other')[: args.focus_science_count]

    if not flats:
        print(f"No suitable flats found for band {band}.")
        return 5

    if not sciences:
        if args.focus_science_files:
            print("No usable science frames were provided.")
        else:
            print(f"No suitable science frames found for band {band}.")
        return 5

    if args.focus_amp_nums:
        requested_indices = list(args.focus_amp_nums)
    else:
        requested_indices = _infer_amp_indices(sciences[0])

    amp_indices = [idx for idx in requested_indices if 1 <= idx <= 8]
    skipped_indices = [idx for idx in requested_indices if idx not in amp_indices]
    if skipped_indices:
        skipped_str = ', '.join(str(idx) for idx in skipped_indices)
        print(f"Skipping unsupported amplifier(s): {skipped_str} (only 1-8 are analysed)")

    if not amp_indices:
        print('No supported amplifier indices (1-8) were found to analyse.')
        return 6

    region_dir = Path(args.focus_region_dir) if args.focus_region_dir else None
    config = FocusConfig(
        threshold=args.focus_threshold,
        cutout_size=args.focus_cutout,
        max_fwhm=args.focus_max_fwhm,
        min_flux_ratio=args.focus_min_flux_ratio,
        max_ellipticity=args.focus_max_e,
        sf=args.focus_sf,
        bf=args.focus_bf,
        sat_sigma=args.focus_sat_sigma,
        black_sigma=args.focus_black_sigma,
        min_samples_for_gmm=args.focus_min_samples,
        random_state=args.focus_random_state,
        write_regions=args.focus_write_regions,
        region_dir=region_dir,
    )

    results = []
    for amp in amp_indices:
        print(f"Analysing amplifier {amp}...")
        try:
            res = analyze_amplifier(amp, biases, darks, flats, sciences, config, region_basename=f"amp{amp}")
        except Exception as exc:
            print(f"  Failed for amp {amp}: {exc}")
            continue
        results.append(res)
        print(f"  Median FWHM: {res.median('fwhm'):.3f}  (stars: {res.star_count})")

    if not results:
        print('No amplifier analyses succeeded.')
        return 7

    summary = aggregate_results(results)
    print('\nSummary:')
    for row in summary:
        print(
            f"  Amp {row['amp']}: median FWHM={row['median_fwhm']:.3f}, median e={row['median_e']:.3f}, stars={row['star_count']} (candidates={row['candidate_count']})"
        )

    if args.focus_export:
        export_path = Path(args.focus_export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open('w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
        print(f"Summary exported to {export_path}")

    if not args.focus_no_gui:
        try:
            launch_gui(results)
        except Exception as exc:
            print(f"GUI launch failed: {exc}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
