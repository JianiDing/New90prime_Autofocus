"""Utility functions for CCD data handling."""
from typing import List, Tuple
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




def diff_amp(amp_num: int, bias_files: List[str], dark_files: List[str], flat_files: List[str], sciences: List[str]) -> Tuple[List[np.ndarray], ...]:
    """Load amp-specific data arrays from lists of FITS files."""
    def _load_list(file_list):
        out = []
        for f in file_list:
            try:
                with fits.open(f, memmap=False) as hdul:
                    data = hdul[amp_num].data
            except Exception as e:
                # store None for problematic files and continue; caller will validate
                data = None
                print(f"Warning: could not read HDU {amp_num} from {f}: {e}")
            out.append(data)
        return out

    biases = _load_list(bias_files)
    darks = _load_list(dark_files)
    flats = _load_list(flat_files)
    sciencesf = _load_list(sciences)

    return biases, darks, flats, sciencesf


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
    Master_dark = np.median(darks_stack, axis=0)
    Unbiased_dark = Master_dark - Master_bias
    Master_flat = np.median(flats_stack, axis=0)

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
