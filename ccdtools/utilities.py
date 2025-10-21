"""Utility functions for CCD data handling."""
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits
import glob
import os

# The single keyword that contains both image type (flat, dark, bias) 
# and band information (u, g, i) in your specific FITS files.
     

def skim_fits_files(directory='.', Keyword='OBJECT' , target_bands=['U', 'G', 'I'], exclusion_list=[]):
    """
    Skims through FITS files, applies an exclusion filter, and categorizes
    calibration frames (flat, dark, bias) by the specified photometry bands,
    using the OBJECT keyword for both type and band identification.

    Parameters:
    - keyword (str): header name for the filter e.g. 'OBJECT' 
    - directory (str): Path to the directory containing the FITS files. Defaults to the current directory.
    - target_bands (list): List of photometry bands to categorize (e.g., ['U', 'G', 'I']).
    - exclusion_list (list): List of strings. Files containing these strings
                             in their name will be skipped.

    Returns:
    - dict: A dictionary containing the categorized file paths.
    """
    # 1. Initialize the data structure
    # Band-specific lists for flats and any other band-specific files
    KEYWORD_COMPOUND = Keyword
    categorized_files = {band.upper(): {'flat': [], 'other': []} for band in target_bands}
    
    # Top-level lists for band-independent frames
    categorized_files['bias_frames'] = []
    categorized_files['dark_frames'] = []
    
    # Lists for unclassified/unwanted files
    categorized_files['unmatched'] = []
    categorized_files['excluded'] = []

    # Ensure bands are uppercase for robust matching
    target_bands = [b.upper() for b in target_bands]

    fits_files = glob.glob(os.path.join(directory, '*.fits'))
    print(f"Found {len(fits_files)} FITS files to check in '{directory}'.")

    for filename in fits_files:
        basename = os.path.basename(filename)
        
        # 2. Exclusion Check
        is_excluded = False
        for exclusion_str in exclusion_list:
            if exclusion_str in basename:
                categorized_files['excluded'].append(filename)
                is_excluded = True
                break
        
        if is_excluded:
            continue # Skip to the next file if excluded

        # 3. Process the FITS File
        try:
            with fits.open(filename, memmap=False) as hdul:
                header = hdul[0].header
                
                # Get the COMPOSITE value (e.g., 'flat_g  ' or 'bias  ')
                composite_value = str(header.get(KEYWORD_COMPOUND, 'UNKNOWN')).lower().strip()

                # --- 4. Categorize by Image Type ---
                
                # A. Band-Independent Frames (Dark/Bias)
                if 'bias' in composite_value or 'zero' in composite_value:
                    categorized_files['bias_frames'].append(filename)
                    continue # Finished with this file, move to the next

                elif 'dark' in composite_value:
                    categorized_files['dark_frames'].append(filename)
                    continue # Finished with this file, move to the next

                # B. Band-Dependent Frames (Flats and other science/standard frames)
                
                image_type = 'other'
                found_band = None
                
                # Check for FLAT frames
                if 'flat' in composite_value:
                    image_type = 'flat'
                
                # Identify the band for flats/other types by checking if a target band 
                # letter (u, g, i) is present in the object string
                for band in target_bands:
                     if band.lower() in composite_value:
                         found_band = band.upper()
                         break
                
                # Store flats and other band-specific frames
                if found_band:
                    categorized_files[found_band][image_type].append(filename)
                
                # If no target band was found or it's a completely unclassified type
                else:
                    categorized_files['unmatched'].append(f"{filename} (OBJECT: {composite_value})")


        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    # 5. Print Summary and Return
    print(f"\n--- Skimming Complete ---")
    print(f"Total FITS files checked: {len(fits_files)}")
    print(f"Total files processed (not excluded): {len(fits_files) - len(categorized_files['excluded'])}")
    print(f"Total files excluded: {len(categorized_files['excluded'])}")
    
    return categorized_files

def diff_amp(
    amp_num: int,
    bias_files: Sequence[str],
    dark_files: Sequence[str],
    flat_files: Sequence[str],
    sciences: Sequence[str],
    *,
    overscan_slice: Optional[slice] = slice(2040, 2060),
    dtype: np.dtype = np.float32,
) -> Tuple[List[Optional[np.ndarray]], ...]:
    """Load amplifier cutouts from FITS lists with optional overscan removal.

    Parameters
    ----------
    amp_num:
        HDU index (amplifier number) to extract from each FITS file.
    bias_files, dark_files, flat_files, sciences:
        Sequences of FITS paths for each calibration category.
    overscan_slice:
        Column slice marking the overscan region to median-subtract. Pass ``None``
        to disable overscan correction.
    dtype:
        Target NumPy dtype for returned arrays (defaults to ``float32``).

    Returns
    -------
    tuple
        Four lists of NumPy arrays (or ``None`` when a file could not be read):
        ``(biases, darks, flats, sciences)``.
    """

    def _load_collection(file_list: Sequence[str]) -> List[Optional[np.ndarray]]:
        loaded: List[Optional[np.ndarray]] = []
        for path in file_list:
            try:
                with fits.open(path, memmap=False) as hdul:
                    data = hdul[amp_num].data
                    if data is None:
                        raise ValueError("HDU contains no data")
                    array = np.asarray(data, dtype=dtype)
                    if overscan_slice is not None and array.ndim == 2:
                        try:
                            overscan_region = array[:, overscan_slice]
                        except Exception:
                            overscan_region = None
                        if overscan_region is not None and overscan_region.size:
                            overscan_level = float(np.nanmedian(overscan_region))
                            array = array - overscan_level
                    loaded.append(array)
            except Exception as exc:  # pragma: no cover - log and continue
                print(f"Warning: could not read HDU {amp_num} from {path}: {exc}")
                loaded.append(None)
        return loaded

    biases = _load_collection(bias_files)
    darks = _load_collection(dark_files)
    flats = _load_collection(flat_files)
    sciences_out = _load_collection(sciences)

    return biases, darks, flats, sciences_out


def data_reduction(
    amp_num: int,
    master_bias: np.ndarray,
    master_flat: np.ndarray,
    science_arrays: Sequence[Optional[np.ndarray]],
    science_paths: Sequence[str],
    *,
    master_dark: Optional[np.ndarray] = None,
    subtract_bias_from_flat: bool = True,
    plot_normalize_flat: bool = False,
    plot_final_sci: bool = False,
    write_output: bool = True,
    output_suffix: str = "_reduced",
    output_dir: Optional[os.PathLike] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Apply bias/flat (and optional dark) correction to science frames.

    Parameters
    ----------
    amp_num:
        Amplifier HDU index used for naming reduced outputs.
    master_bias, master_flat:
        Calibration frames computed by :func:`flat_reduction_b`.
    science_arrays:
        Sequence of in-memory science images (as returned by :func:`diff_amp`).
    science_paths:
        Parallel sequence of FITS paths, used for headers/output naming.
    master_dark:
        Optional master dark frame to subtract after bias removal.
    subtract_bias_from_flat:
        When ``True`` the master flat has the master bias removed before
        normalisation (mirrors legacy notebook behaviour).
    plot_normalize_flat, plot_final_sci:
        When ``True`` display diagnostic matplotlib plots.
    write_output:
        If ``True`` write calibrated FITS files alongside the source data (or in
        ``output_dir`` when provided).
    output_suffix:
        String appended to the base filename for saved products.
    output_dir:
        Optional override directory for reduced FITS files.

    Returns
    -------
    tuple
        ``(normalized_flat, reduced_images)`` where ``reduced_images`` is a list
        of calibrated ``float32`` arrays in the same order as the input paths.
    """

    if len(science_arrays) != len(science_paths):
        raise ValueError("science_arrays and science_paths must have the same length")

    master_bias = np.asarray(master_bias, dtype=np.float32)
    master_flat = np.asarray(master_flat, dtype=np.float32)
    flat_for_norm = master_flat - master_bias if subtract_bias_from_flat else master_flat

    flat_median = float(np.nanmedian(flat_for_norm))
    if not np.isfinite(flat_median) or flat_median == 0:
        flat_median = 1.0
    normalized_flat = flat_for_norm / flat_median

    if plot_normalize_flat:
        import matplotlib.pyplot as plt  # local import to avoid hard dependency at import time

        plt.figure()
        plt.title('Normalized Flat')
        plt.imshow(normalized_flat, origin='lower', cmap='viridis')
        plt.colorbar()

    output_base = Path(output_dir) if output_dir is not None else None
    reduced_images: List[np.ndarray] = []

    for array, path in zip(science_arrays, science_paths):
        if array is None and not path:
            continue

        data: np.ndarray
        header = None
        if path:
            with fits.open(path, memmap=False) as hdul:
                header = hdul[amp_num].header.copy()
                if array is None:
                    data = np.asarray(hdul[amp_num].data, dtype=np.float32)
                else:
                    data = np.asarray(array, dtype=np.float32)
        else:
            data = np.asarray(array, dtype=np.float32)

        calibrated = data - master_bias
        if master_dark is not None:
            calibrated = calibrated - np.asarray(master_dark, dtype=np.float32)

        with np.errstate(divide='ignore', invalid='ignore'):
            final = np.divide(
                calibrated,
                normalized_flat,
                out=np.zeros_like(calibrated, dtype=np.float32),
                where=normalized_flat != 0,
            )

        reduced_images.append(final)

        if plot_final_sci:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.title(f'Reduced Science Image (amp {amp_num})')
            vmax = np.nanmean(final) * 2 if np.isfinite(np.nanmean(final)) else None
            plt.imshow(final, origin='lower', cmap='gray', vmax=vmax)
            plt.colorbar()

        if write_output and path:
            destination_dir = output_base if output_base is not None else Path(path).parent
            destination_dir.mkdir(parents=True, exist_ok=True)
            output_name = Path(path).stem + f"_amp{amp_num}{output_suffix}.fits"
            output_path = destination_dir / output_name
            fits.writeto(output_path, final, header=header, overwrite=True)

    if plot_normalize_flat or plot_final_sci:
        import matplotlib.pyplot as plt

        plt.show()

    return normalized_flat, reduced_images


def flat_reduction_b(biases, darks, flats, plot_master_bias: bool = False, plot_unbias_dark: bool = False, plot_master_flat: bool = False):
    """Compute master bias/dark/flat frames from lists of arrays."""
    import matplotlib.pyplot as plt

    # Basic validations
    for name, arrs in (('biases', biases), ('darks', darks), ('flats', flats)):
        if not arrs:
            raise ValueError(f"{name} list is empty - need at least one frame to compute masters")
        if any(a is None for a in arrs):
            bad_idx = [i for i, a in enumerate(arrs) if a is None]
            raise ValueError(f"{name} contains unreadable frames at indices: {bad_idx}; check earlier warnings")

    # verify shapes are consistent within each set
    def _check_and_stack(arrs, label):
        shapes = [getattr(a, 'shape', None) for a in arrs]
        unique_shapes = set(shapes)
        if None in unique_shapes:
            raise ValueError(f"{label} contains None arrays: shapes={shapes}")
        if len(unique_shapes) != 1:
            raise ValueError(f"Inconsistent shapes in {label}: {shapes}")
        return np.stack(arrs, axis=0)

    biases_stack = _check_and_stack(biases, 'biases')
    darks_stack = _check_and_stack(darks, 'darks')
    flats_stack = _check_and_stack(flats, 'flats')
    
    Master_bias = np.median(biases_stack, axis=0)
    # Subtract Master_bias from each dark and flat before computing masters
    # This produces unbiased dark frames and unbiased flat frames
    darks_unbiased = darks_stack - Master_bias[np.newaxis, ...]
    flats_unbiased = flats_stack - Master_bias[np.newaxis, ...]

    # Compute master frames from the bias-subtracted stacks
    Master_dark = np.median(darks_unbiased, axis=0)
    Unbiased_dark = Master_dark  # already has bias removed per-frame

    Master_flat = np.median(flats_unbiased, axis=0)


    if plot_master_bias:
        plt.figure()
        plt.title('Master Bias')
        plt.imshow(Master_bias, vmax=1200)

    if plot_unbias_dark:
        plt.figure()
        plt.title('Unbiased Dark')
        plt.imshow(Unbiased_dark, vmax=1200)

    if plot_master_flat:
        plt.figure()
        plt.title('Master Flat')
        plt.imshow(Master_flat, vmax=2.5e4)

    return Master_bias, Master_dark, Unbiased_dark, Master_flat
