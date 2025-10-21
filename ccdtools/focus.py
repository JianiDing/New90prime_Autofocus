"""Focus analysis and amplifier quality tools.

This module extracts notebook logic to evaluate focus/seeing quality per
amplifier. It calibrates frames, detects point sources, clusters candidates,
reports median FWHM and ellipticity, and optionally launches a Tk GUI to
visualise the results across amplifier subregions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import sep
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .bad_pixel_mask import find_bad_columns, mask_bad_columns
from .utilities import diff_amp, flat_reduction_b


def _ensure_native_float(array: np.ndarray) -> np.ndarray:
    """Return a float32 array in native endianness (handles FITS big-endian data)."""
    arr = np.asarray(array)
    if arr.dtype.byteorder not in ("=", "|"):
        arr = arr.byteswap().newbyteorder()
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    else:
        arr = arr.astype(np.float32, copy=False)
    return arr


def _radial_profile(data: np.ndarray, center: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(int)
    r_max = int(r.max()) + 1
    radial_mean = np.array([data[r == i].mean() if np.any(r == i) else np.nan for i in range(r_max)])
    return np.arange(r_max, dtype=np.float32), radial_mean


def _gaussian(r: np.ndarray, a: float, mu: float, sigma: float, c: float) -> np.ndarray:
    return a * np.exp(-((r - mu) ** 2) / (2 * sigma ** 2)) + c


@dataclass
class FocusConfig:
    threshold: float = 5.0
    cutout_size: int = 15
    max_fwhm: float = 15.0
    min_flux_ratio: float = 2.0
    max_ellipticity: float = 1.0
    sf: float = 0.25
    bf: float = 0.2
    sat_sigma: float = 1.0
    black_sigma: float = 4.0
    sat_threshold: Optional[float] = None
    black_threshold: Optional[float] = None
    min_samples_for_gmm: int = 12
    random_state: int = 0
    write_regions: bool = False
    region_dir: Optional[Path] = None


@dataclass
class FocusAmpAnalysis:
    amp_index: int
    x: np.ndarray
    y: np.ndarray
    flux: np.ndarray
    fwhm: np.ndarray
    ellipticity: np.ndarray
    flux_ratio: np.ndarray
    flags: np.ndarray
    candidate_mask: np.ndarray
    star_mask: np.ndarray
    labels: np.ndarray

    def median(self, field: str) -> float:
        data = getattr(self, field)
        values = data[self.star_mask]
        if values.size == 0:
            return float("nan")
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return float("nan")
        return float(np.nanmedian(finite))

    @property
    def star_count(self) -> int:
        return int(np.count_nonzero(self.star_mask))

    @property
    def candidate_count(self) -> int:
        return int(np.count_nonzero(self.candidate_mask))


def _calibrate_science_frame(
    science: np.ndarray,
    master_bias: np.ndarray,
    master_dark: np.ndarray,
    master_flat: np.ndarray,
) -> np.ndarray:
    """Basic calibration: subtract bias/dark and divide by a normalised flat."""
    sci = _ensure_native_float(science)
    bias = _ensure_native_float(master_bias)
    dark = _ensure_native_float(master_dark)
    flat = _ensure_native_float(master_flat)

    calibrated = sci - bias - dark
    flat_median = np.nanmedian(flat)
    if not np.isfinite(flat_median) or flat_median == 0:
        flat_median = 1.0
    normalised_flat = flat / flat_median
    normalised_flat = np.where(normalised_flat == 0, 1.0, normalised_flat)
    return calibrated / normalised_flat


def _sep_detect(
    image: np.ndarray,
    threshold: float,
    cutout_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = _ensure_native_float(image)
    finite_mask = np.isfinite(data)
    if not np.all(finite_mask):
        fill = np.nanmedian(data[finite_mask]) if np.any(finite_mask) else 0.0
        data = np.where(finite_mask, data, fill)

    mask = data > (np.median(data) + 5 * np.std(data))
    bkg = sep.Background(data, bw=64, bh=64, fw=3, fh=3, mask=mask)
    data_sub = data - bkg
    objects, segmap = sep.extract(data_sub, thresh=threshold, err=bkg.globalrms, segmentation_map=True, minarea=20)

    fwhm_radial: List[float] = []
    flux_in_fwhm: List[float] = []
    flux_peak_ratio: List[float] = []

    from scipy.optimize import curve_fit  # Lazy import to reduce top-level dependency cost

    for i in range(len(objects)):
        x, y = int(objects['x'][i]), int(objects['y'][i])
        if (
            y - cutout_size < 0
            or y + cutout_size >= data_sub.shape[0]
            or x - cutout_size < 0
            or x + cutout_size >= data_sub.shape[1]
        ):
            fwhm_radial.append(np.nan)
            flux_in_fwhm.append(np.nan)
            flux_peak_ratio.append(np.nan)
            continue

        cutout = data_sub[y - cutout_size : y + cutout_size, x - cutout_size : x + cutout_size]
        rp_r, rp_flux = _radial_profile(cutout, (cutout_size, cutout_size))

        try:
            popt, _ = curve_fit(
                _gaussian,
                rp_r,
                rp_flux,
                p0=[float(np.nanmax(rp_flux)), 0.0, 2.0, float(np.nanmedian(rp_flux[-5:]))],
                maxfev=5000,
            )
            sigma = abs(popt[2])
            fwhm = 2.355 * sigma
        except Exception:
            fwhm = np.nan

        fwhm_radial.append(fwhm)

        if np.isnan(fwhm):
            flux_in_fwhm.append(np.nan)
            flux_peak_ratio.append(np.nan)
            continue

        yy, xx = np.indices(cutout.shape)
        rr = np.sqrt((xx - cutout_size) ** 2 + (yy - cutout_size) ** 2)
        mask_fwhm = rr <= fwhm
        flux_fwhm = float(np.sum(cutout[mask_fwhm]))
        peak_pix = float(np.nanmax(cutout))
        flux_in_fwhm.append(flux_fwhm)
        flux_peak_ratio.append(flux_fwhm / peak_pix if peak_pix != 0 else np.nan)

    ellipticity = 1.0 - objects['b'] / objects['a']
    return (
        data_sub,
        objects,
        np.array(fwhm_radial, dtype=np.float32),
        np.array(flux_in_fwhm, dtype=np.float32),
        np.array(flux_peak_ratio, dtype=np.float32),
        ellipticity,
    )


def analyze_amplifier(
    amp_index: int,
    bias_files: Sequence[str],
    dark_files: Sequence[str],
    flat_files: Sequence[str],
    science_files: Sequence[str],
    config: FocusConfig,
    region_basename: Optional[str] = None,
) -> FocusAmpAnalysis:
    biases, darks, flats, sciences = diff_amp(amp_index, list(bias_files), list(dark_files), list(flat_files), list(science_files))

    usable_sciences = [s for s in sciences if s is not None]
    if not usable_sciences:
        raise ValueError(f"No usable science frames for amp {amp_index}")

    Master_bias, Master_dark, _, Master_flat = flat_reduction_b(
        biases,
        darks,
        flats,
        plot_master_bias=False,
        plot_unbias_dark=False,
        plot_master_flat=False,
    )
    calibrated = _calibrate_science_frame(usable_sciences[0], Master_bias, Master_dark, Master_flat)

    sat_thresh = config.sat_threshold
    if sat_thresh is None:
        sat_thresh = float(np.nanmedian(Master_flat) + config.sat_sigma * np.nanstd(Master_flat))
    black_thresh = config.black_threshold
    if black_thresh is None:
        black_thresh = float(np.nanmedian(Master_flat) - config.black_sigma * np.nanstd(Master_flat))

    bad_columns = find_bad_columns(
        Master_flat,
        saturation_threshold=sat_thresh,
        black_column_threshold=black_thresh,
        sf=config.sf,
        bf=config.bf,
    )
    masked_image = mask_bad_columns(calibrated, bad_columns)

    _, objects, fwhm_radial, flux_in_fwhm, flux_ratio, ellipticity = _sep_detect(
        masked_image,
        threshold=config.threshold,
        cutout_size=config.cutout_size,
    )

    flux = objects['flux'].astype(np.float32)
    flags = objects['flag'].astype(np.int16)
    x = objects['x'].astype(np.float32)
    y = objects['y'].astype(np.float32)

    candidate_mask = (
        np.isfinite(fwhm_radial)
        & np.isfinite(flux_ratio)
        & np.isfinite(ellipticity)
        & np.isfinite(flux)
        & (fwhm_radial > 0)
        & (fwhm_radial < config.max_fwhm)
        & (flux_ratio > config.min_flux_ratio)
        & (ellipticity < config.max_ellipticity)
        & (flags == 0)
    )

    labels = np.full(len(flux), -1, dtype=int)

    candidate_indices = np.where(candidate_mask)[0]
    if candidate_indices.size >= config.min_samples_for_gmm:
        features = np.column_stack(
            (
                fwhm_radial[candidate_indices],
                flux_ratio[candidate_indices],
                -2.5 * np.log10(np.clip(flux[candidate_indices], 1e-10, None)),
                ellipticity[candidate_indices],
            )
        )
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        gmm = GaussianMixture(n_components=2, random_state=config.random_state)
        labels_candidate = gmm.fit_predict(scaled)

        medians = []
        for lbl in np.unique(labels_candidate):
            medians.append((lbl, np.nanmedian(fwhm_radial[candidate_indices][labels_candidate == lbl])))
        medians.sort(key=lambda item: item[1])
        star_label = medians[0][0]

        labels_candidate_binary = np.where(labels_candidate == star_label, 0, 1)
        labels[candidate_indices] = labels_candidate_binary
    else:
        labels[candidate_mask] = 0

    star_mask = labels == 0

    if config.write_regions and region_basename:
        region_dir = config.region_dir or Path('.')
        region_dir.mkdir(parents=True, exist_ok=True)
        region_path = region_dir / f"amp{amp_index}_sources_ellipse.reg"
        with region_path.open('w') as reg:
            reg.write("# Region file format: DS9 version 4.1\n")
            reg.write("image\n")
            for i in range(len(objects)):
                reg.write(
                    f"ellipse({objects['x'][i]:.2f},{objects['y'][i]:.2f},{objects['a'][i]:.2f},{objects['b'][i]:.2f},{objects['theta'][i] * 180.0 / np.pi:.2f})\n"
                )

    return FocusAmpAnalysis(
        amp_index=amp_index,
        x=x,
        y=y,
        flux=flux,
        fwhm=fwhm_radial,
        ellipticity=ellipticity.astype(np.float32),
        flux_ratio=flux_ratio,
        flags=flags,
        candidate_mask=candidate_mask,
        star_mask=star_mask,
        labels=labels,
    )


def aggregate_results(results: Iterable[FocusAmpAnalysis]) -> List[Dict[str, float]]:
    summary: List[Dict[str, float]] = []
    for res in results:
        summary.append(
            {
                "amp": res.amp_index,
                "median_fwhm": res.median("fwhm"),
                "median_e": res.median("ellipticity"),
                "star_count": res.star_count,
                "candidate_count": res.candidate_count,
            }
        )
    return summary


def launch_gui(results: Sequence[FocusAmpAnalysis]) -> None:
    """Launch the amplifier grid GUI matching the notebook focus analysis layout."""
    import tkinter as tk

    class AmpGridGUI(tk.Tk):
        def __init__(self, amp_results: Sequence[FocusAmpAnalysis]):
            super().__init__()
            self.title("Amplifier Focus Grid (FWHM / Ellipticity)")
            self.geometry("1500x900")
            self.amp_results = sorted(amp_results, key=lambda r: r.amp_index)
            self.grid_mode = 1
            self.active_param = 'FWHM'

            ctrl_frame = tk.Frame(self)
            ctrl_frame.pack(side=tk.TOP, pady=10)
            tk.Button(ctrl_frame, text="Show FWHM", font=("Arial", 14), command=lambda: self.show_param('FWHM')).pack(side=tk.LEFT, padx=10)
            tk.Button(ctrl_frame, text="Show Ellipticity", font=("Arial", 14), command=lambda: self.show_param('e')).pack(side=tk.LEFT, padx=10)
            tk.Label(ctrl_frame, text="Subgrids:", font=("Arial", 14)).pack(side=tk.LEFT, padx=5)
            self.grid_var = tk.IntVar(value=1)
            grid_menu = tk.OptionMenu(ctrl_frame, self.grid_var, 1, 2, 4, command=self.update_grid_mode)
            grid_menu.config(font=("Arial", 14), width=3)
            grid_menu.pack(side=tk.LEFT)

            self.amp_frame = tk.Frame(self)
            self.amp_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.amp_panels: List[tk.Frame] = []
            self.show_param('FWHM')

        def update_grid_mode(self, value):
            self.grid_mode = int(value)
            self.show_param(self.active_param)

        def show_param(self, param):
            self.active_param = param
            for widget in self.amp_frame.winfo_children():
                widget.destroy()
            self.amp_panels.clear()
            grid_per_amp = self.grid_mode

            total_amps = len(self.amp_results)
            for idx, result in enumerate(self.amp_results):
                amp = result.amp_index
                row = idx // 4
                col = idx % 4
                amp_outer_frame = tk.Frame(self.amp_frame, width=200, height=400, bd=4, relief=tk.RIDGE, bg="#cccccc")
                amp_outer_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
                amp_outer_frame.grid_propagate(False)
                tk.Label(amp_outer_frame, text=f"Amp {amp}", font=("Arial", 16, "bold"), bg="#cccccc").pack(side=tk.TOP, pady=3)

                subgrid_frame = tk.Frame(amp_outer_frame, bg="#dddddd")
                subgrid_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

                mask = result.labels == 0
                x = result.x[mask]
                y = result.y[mask]
                if param == 'FWHM':
                    vals = result.fwhm[mask]
                else:
                    vals = result.ellipticity[mask]

                if grid_per_amp == 1:
                    median = float(np.nanmedian(vals)) if vals.size else float('nan')
                    N = int(np.count_nonzero(~np.isnan(vals)))
                    panel = tk.Frame(subgrid_frame, width=160, height=320, bd=2, relief=tk.GROOVE, bg="#eeeeee")
                    panel.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
                    label_param = 'FWHM' if param == 'FWHM' else 'e'
                    tk.Label(panel, text=f"Median {label_param}:\n{median:.3f}", font=("Arial", 14), bg="#eeeeee").pack(pady=30)
                    tk.Label(panel, text=f"N = {N}", font=("Arial", 13), bg="#eeeeee", fg="gray").pack(pady=5)
                elif grid_per_amp == 2:
                    if y.size:
                        ymid = 0.5 * (np.nanmin(y) + np.nanmax(y))
                        in_top = y < ymid
                        in_bottom = y >= ymid
                    else:
                        in_top = in_bottom = np.zeros(0, dtype=bool)
                    for split, mask_region in enumerate([in_top, in_bottom]):
                        sub_vals = vals[mask_region]
                        med = float(np.nanmedian(sub_vals)) if sub_vals.size else float('nan')
                        N = int(np.count_nonzero(~np.isnan(sub_vals)))
                        panel = tk.Frame(subgrid_frame, width=160, height=150, bd=2, relief=tk.GROOVE, bg="#f7f7f7")
                        panel.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
                        tk.Label(panel, text=f"{'Top' if split == 0 else 'Bottom'}", font=("Arial", 12, "bold"), bg="#f7f7f7").pack(pady=2)
                        tk.Label(panel, text=f"{med:.3f}", font=("Arial", 13), bg="#f7f7f7").pack(pady=4)
                        tk.Label(panel, text=f"N = {N}", font=("Arial", 12), bg="#f7f7f7", fg="gray").pack(pady=2)
                else:
                    if x.size and y.size:
                        xbins = np.linspace(np.nanmin(x), np.nanmax(x), 3)
                        ybins = np.linspace(np.nanmin(y), np.nanmax(y), 3)
                    else:
                        xbins = np.linspace(0, 1, 3)
                        ybins = np.linspace(0, 1, 3)
                    for row2 in range(2):
                        for col2 in range(2):
                            in_panel = (
                                (x >= xbins[col2])
                                & (x < xbins[col2 + 1])
                                & (y >= ybins[row2])
                                & (y < ybins[row2 + 1])
                            )
                            sub_vals = vals[in_panel]
                            med = float(np.nanmedian(sub_vals)) if sub_vals.size else float('nan')
                            N = int(np.count_nonzero(~np.isnan(sub_vals)))
                            panel = tk.Frame(subgrid_frame, width=70, height=70, bd=2, relief=tk.GROOVE, bg="#f7f7f7")
                            panel.grid(row=row2, column=col2, padx=2, pady=2, sticky="nsew")
                            tk.Label(panel, text=f"({['Top','Bottom'][row2]}-{['Left','Right'][col2]})", font=("Arial", 10, "bold"), bg="#f7f7f7").pack(pady=1)
                            tk.Label(panel, text=f"{med:.3f}", font=("Arial", 11), bg="#f7f7f7").pack(pady=2)
                            tk.Label(panel, text=f"N = {N}", font=("Arial", 10), bg="#f7f7f7", fg="gray").pack(pady=1)
                        subgrid_frame.grid_rowconfigure(row2, weight=1)
                    for col2 in range(2):
                        subgrid_frame.grid_columnconfigure(col2, weight=1)

                self.amp_panels.append(amp_outer_frame)

            max_cols = min(4, total_amps)
            for i in range(max_cols):
                self.amp_frame.grid_columnconfigure(i, weight=1)
            max_rows = (total_amps + max_cols - 1) // max_cols
            for i in range(max_rows):
                self.amp_frame.grid_rowconfigure(i, weight=1)

    app = AmpGridGUI(results)
    app.mainloop()


__all__ = [
    "FocusConfig",
    "FocusAmpAnalysis",
    "analyze_amplifier",
    "aggregate_results",
    "launch_gui",
    "SeeingConfig",
    "AmpAnalysis",
]


# Backwards compatibility aliases (older code may import from ccdtools.seeing)
SeeingConfig = FocusConfig
AmpAnalysis = FocusAmpAnalysis

