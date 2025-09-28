"""Bad pixel detection and masking tools."""
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# relative import from the same package
from .utilities import flat_reduction_b


def find_bad_columns(ccd_data: np.ndarray, saturation_threshold: float, black_column_threshold: float, sf: float, bf: float) -> List[int]:
    rows, cols = ccd_data.shape
    bad_columns = []

    for j in range(cols):
        column_data = ccd_data[:, j]
        saturated_pixel_count = np.sum(column_data >= saturation_threshold)
        black_pixel_count = np.sum(column_data <= black_column_threshold)

        is_partially_saturated = saturated_pixel_count > rows * sf
        is_partially_black = black_pixel_count > rows * bf

        if is_partially_saturated or is_partially_black:
            bad_columns.append(j)

    return bad_columns


def create_bad_pixel_map(shape: tuple, bad_columns: List[int]) -> np.ndarray:
    bad_map = np.zeros(shape, dtype=int)
    for col_index in bad_columns:
        bad_map[:, col_index] = 1
    return bad_map


def mask_bad_columns(ccd_data: np.ndarray, bad_columns: List[int]) -> np.ndarray:
    masked_data = np.copy(ccd_data)
    for j in bad_columns:
        masked_data[:, j] = np.nan
    return masked_data


def process_and_save_calibrated_image(calibrated_image: np.ndarray, bad_columns: List[int], amp_num: float, output_path: str = './', show_plots: bool = True):
    """
    Processes a calibrated image by masking bad columns and saving the results.

    This function generates a bad pixel map, masks the bad columns in the image,
    visualizes the results, and saves the masked image and the bad pixel map
    as FITS files.

    Args:
        calibrated_image (np.ndarray): The calibrated 2D NumPy array (e.g., Final_sci).
        bad_columns (List[int]): A list of column indices to mask.
        output_path (str): The directory path to save the output files.
    """
    import os

    # ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Create the bad pixel map
    bad_pixel_map = create_bad_pixel_map(calibrated_image.shape, bad_columns)

    # Mask the bad columns in the calibrated image
    masked_image = mask_bad_columns(calibrated_image, bad_columns)

    # --- Visualize the results (optional) ---
    if show_plots:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].set_title("Masked Image")
        axes[0].imshow(masked_image, cmap='hot')
        axes[1].set_title("Bad Pixel Map")
        axes[1].imshow(bad_pixel_map, cmap='hot', alpha=0.5, vmin=0, vmax=1)
        plt.show()
    
    # --- Save the files to FITS format ---
    # To save the masked image
    masked_hdu = fits.PrimaryHDU(masked_image)
    masked_hdul = fits.HDUList([masked_hdu])
    masked_hdul.writeto(f'{output_path}/masked_image_ampnum_'+str(amp_num)+'.fits', overwrite=True)
    print(f"Saved masked image to: {output_path}/masked_image_ampnum_"+str(amp_num)+'.fits')

    # To save the bad pixel map
    map_hdu = fits.PrimaryHDU(bad_pixel_map)
    map_hdul = fits.HDUList([map_hdu])
    map_hdul.writeto(f'{output_path}/bad_pixel_map_ampnum_'+str(amp_num)+'.fits', overwrite=True)
    print(f"Saved bad pixel map to: {output_path}/bad_pixel_map_ampnum_"+str(amp_num)+'.fits')

    return bad_pixel_map, masked_image


def main():
    """Small demo when running as a module."""
    # This demo expects a fits file named 'mosaic.fits' in the package root or current dir
    try:
        with fits.open('mosaic.fits') as hdul:
            data = hdul[0].data
    except Exception as e:
        print('Demo: could not open mosaic.fits in current directory:', e)
        return

    bad_cols = find_bad_columns(data, saturation_threshold=60000, black_column_threshold=5, sf=0.5, bf=0.25)
    print('Found bad columns:', bad_cols)
    process_and_save_calibrated_image(data, bad_cols)


if __name__ == '__main__':

    main()
