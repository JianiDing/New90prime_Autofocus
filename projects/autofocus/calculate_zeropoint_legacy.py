#!/usr/bin/env python3
"""Estimate Bok zero points by matching pipeline stars to Legacy Survey.

This script uses the ``focus_sources.fits`` catalog produced by
``focus_pipeline.py``. Source ``x``/``y`` positions are transformed to sky
coordinates with the WCS in the matching raw science FITS amplifier header,
then cross-matched against a Legacy/Tractor catalog.

The most reliable mode is to pass a local Legacy Survey Tractor/sweep catalog:

  python calculate_zeropoint_legacy.py \\
    --sources focus_output/focus_sources.fits \\
    --time-series focus_output/focus_time_series.ecsv \\
    --data-dir /path/to/night \\
    --legacy-catalog legacy_sweep_subset.fits \\
    --filter r \\
    --image-nums 172-201 \\
    --out zeropoint_legacy.csv

If ``astroquery`` is installed, ``--query-vizier`` can query a VizieR Legacy
Survey table instead, but local Tractor/sweep files are preferred for speed and
repeatability.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table, vstack
from astropy.wcs import WCS
import astropy.units as u


_NUM_RE = re.compile(r"(\d+)")


def parse_num_ranges(tokens: Iterable[str] | None) -> set[int] | None:
    if not tokens:
        return None
    out: set[int] = set()
    for token in tokens:
        for part in str(token).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                out.update(range(int(a), int(b) + 1))
            else:
                out.add(int(part))
    return out


def image_number(path_or_name: str | Path) -> int | None:
    nums = _NUM_RE.findall(Path(path_or_name).stem)
    return int(nums[-1]) if nums else None


def discover_images(data_dir: Path, name_contains: str | None = "object") -> list[Path]:
    files = sorted(data_dir.glob("*.fits"))
    if name_contains:
        needle = name_contains.lower()
        files = [p for p in files if needle in p.name.lower()]
    return sorted(files, key=lambda p: (image_number(p) is None, image_number(p) or -1, p.name))


def image_map_from_time_series(time_series: Path, data_dir: Path) -> dict[int, Path]:
    tab = Table.read(time_series)
    if "sci_file" not in tab.colnames:
        raise ValueError(f"{time_series} has no sci_file column")
    mapping: dict[int, Path] = {}
    for idx, name in enumerate(tab["sci_file"]):
        mapping[idx] = data_dir / str(name)
    return mapping


def image_map_from_directory(data_dir: Path, name_contains: str | None) -> dict[int, Path]:
    return {idx: path for idx, path in enumerate(discover_images(data_dir, name_contains))}


def parse_subset_ids(sources: Table) -> tuple[np.ndarray, np.ndarray]:
    if "subset_id" not in sources.colnames:
        raise ValueError("source catalog has no subset_id column; cannot map rows to images/amps")
    file_idx = np.full(len(sources), -1, dtype=int)
    amp = np.full(len(sources), -1, dtype=int)
    for i, sid in enumerate(sources["subset_id"]):
        match = re.search(r"(\d+)_amp(\d+)", str(sid))
        if match:
            file_idx[i] = int(match.group(1))
            amp[i] = int(match.group(2))
    return file_idx, amp


def apply_basic_source_cuts(tab: Table, max_fwhm: float, max_e: float) -> np.ndarray:
    mask = np.ones(len(tab), dtype=bool)
    for col in ("x", "y", "flux", "FWHM", "e", "flux_ratio"):
        if col in tab.colnames:
            mask &= np.isfinite(np.array(tab[col], dtype=float))
    if "flag" in tab.colnames:
        mask &= np.array(tab["flag"], dtype=int) == 0
    if "flux" in tab.colnames:
        mask &= np.array(tab["flux"], dtype=float) > 0
    if "flux_ratio" in tab.colnames:
        mask &= np.array(tab["flux_ratio"], dtype=float) > 1
    if "FWHM" in tab.colnames:
        mask &= np.array(tab["FWHM"], dtype=float) < max_fwhm
    if "e" in tab.colnames:
        mask &= np.array(tab["e"], dtype=float) < max_e
    return mask


def gmm_star_mask(tab: Table) -> np.ndarray:
    try:
        from focus_pipeline import compute_gmm_labels
    except Exception as exc:
        print(f"[warn] could not import pipeline GMM selection ({exc}); using basic cuts only")
        return np.ones(len(tab), dtype=bool)
    labels = compute_gmm_labels(tab)
    return labels == 0


def hdu_for_amp(hdul: fits.HDUList, amp: int):
    suffix = str(amp)
    for hdu in hdul:
        name = str(hdu.header.get("EXTNAME") or hdu.header.get("AMPNAME") or hdu.name)
        if name.upper().endswith(suffix) and hdu.data is not None:
            return hdu
    if amp < len(hdul) and hdul[amp].data is not None:
        return hdul[amp]
    raise ValueError(f"could not find amp {amp} HDU")


def wcs_is_reasonable(wcs: WCS) -> bool:
    try:
        scales = np.array([float(s.to_value(u.arcsec)) for s in wcs.proj_plane_pixel_scales()])
        return bool(np.all(np.isfinite(scales)) and np.nanmax(scales) > 0.01)
    except Exception:
        return False


def add_sky_positions(
    sources: Table,
    file_idx: np.ndarray,
    amp: np.ndarray,
    image_map: dict[int, Path],
) -> Table:
    rows: list[Table] = []
    for idx in sorted(set(file_idx[file_idx >= 0])):
        image_path = image_map.get(int(idx))
        if image_path is None or not image_path.exists():
            print(f"[warn] no image file for catalog file_idx={idx}; skipping")
            continue
        rows_for_image: list[Table] = []
        with fits.open(image_path, memmap=False) as hdul:
            for amp_num in sorted(set(amp[(file_idx == idx) & (amp > 0)])):
                row_mask = (file_idx == idx) & (amp == amp_num)
                sub = sources[row_mask].copy()
                if len(sub) == 0:
                    continue
                try:
                    hdu = hdu_for_amp(hdul, int(amp_num))
                    wcs = WCS(hdu.header)
                    if not wcs_is_reasonable(wcs):
                        print(f"[warn] unreasonable WCS for {image_path.name} amp{amp_num}; skipping")
                        continue
                    ra, dec = wcs.all_pix2world(
                        np.array(sub["x"], dtype=float),
                        np.array(sub["y"], dtype=float),
                        0,
                    )
                except Exception as exc:
                    print(f"[warn] WCS failed for {image_path.name} amp{amp_num}: {exc}")
                    continue
                finite = np.isfinite(ra) & np.isfinite(dec)
                if not np.any(finite):
                    continue
                sub = sub[finite]
                sub["ra"] = ra[finite]
                sub["dec"] = dec[finite]
                sub["sci_file"] = image_path.name
                sub["image_num"] = image_number(image_path) or -1
                sub["amp"] = int(amp_num)
                rows_for_image.append(sub)
        if rows_for_image:
            rows.append(vstack(rows_for_image, join_type="outer"))
    if not rows:
        raise RuntimeError("No source rows with valid WCS positions")
    return vstack(rows, join_type="outer")


def read_legacy_catalog(path: Path) -> Table:
    tab = Table.read(path)
    lower = {c.lower(): c for c in tab.colnames}
    if not ({"ra", "dec"} <= set(lower) or {"raj2000", "dej2000"} <= set(lower)):
        raise ValueError("Legacy catalog must contain RA/DEC or RAJ2000/DEJ2000 columns")
    return tab


def query_vizier_legacy(center: SkyCoord, radius_deg: float, catalog: str) -> Table:
    try:
        from astroquery.vizier import Vizier
    except Exception as exc:
        raise RuntimeError("astroquery is required for --query-vizier") from exc
    Vizier.ROW_LIMIT = -1
    result = Vizier.query_region(center, radius=radius_deg * u.deg, catalog=catalog)
    if not result:
        raise RuntimeError(f"No VizieR rows returned from {catalog}")
    if len(result) > 1:
        print(f"[info] VizieR returned {len(result)} tables; using the first")
    return result[0]


def column_lookup(tab: Table, *names: str) -> str | None:
    lower = {c.lower(): c for c in tab.colnames}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def legacy_ra_dec(tab: Table) -> tuple[np.ndarray, np.ndarray]:
    ra_col = column_lookup(tab, "ra", "RA", "RAJ2000")
    dec_col = column_lookup(tab, "dec", "DEC", "DEJ2000")
    if not ra_col or not dec_col:
        raise ValueError("Could not find RA/Dec columns in Legacy catalog")
    return np.array(tab[ra_col], dtype=float), np.array(tab[dec_col], dtype=float)


def legacy_mag(tab: Table, band: str) -> tuple[np.ndarray, str]:
    b = band.lower()
    direct = column_lookup(tab, f"{b}mag", f"mag_{b}", f"{b}_mag")
    if direct:
        return np.array(tab[direct], dtype=float), direct
    flux_col = column_lookup(tab, f"flux_{b}", f"{b}_flux")
    if flux_col:
        flux = np.array(tab[flux_col], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 22.5 - 2.5 * np.log10(flux), flux_col
    raise ValueError(
        f"Could not find Legacy magnitude for band {band}. "
        f"Expected {b}mag/mag_{b} or flux_{b}."
    )


def star_like_mask(tab: Table, band: str) -> np.ndarray:
    mask = np.ones(len(tab), dtype=bool)
    type_col = column_lookup(tab, "type", "TYPE")
    if type_col:
        typ = np.char.upper(np.array(tab[type_col]).astype(str))
        mask &= np.isin(typ, ["PSF", "STAR"])
    # Tractor maskbits: reject BRIGHT, SATUR, ALLMASK_G/R/I/Z where possible.
    maskbits = column_lookup(tab, "maskbits", "MASKBITS")
    if maskbits:
        mask &= np.array(tab[maskbits], dtype=int) == 0
    anymask = column_lookup(tab, f"anymask_{band.lower()}", f"ANYMASK_{band.upper()}")
    if anymask:
        mask &= np.array(tab[anymask], dtype=int) == 0
    return mask


def estimate_zeropoints(
    sources: Table,
    legacy: Table,
    band: str,
    exptime_by_file: dict[str, float],
    match_radius_arcsec: float,
    mag_min: float | None,
    mag_max: float | None,
) -> Table:
    leg_ra, leg_dec = legacy_ra_dec(legacy)
    cat_mag, mag_col = legacy_mag(legacy, band)
    leg_good = np.isfinite(leg_ra) & np.isfinite(leg_dec) & np.isfinite(cat_mag)
    leg_good &= star_like_mask(legacy, band)
    if mag_min is not None:
        leg_good &= cat_mag >= mag_min
    if mag_max is not None:
        leg_good &= cat_mag <= mag_max

    legacy_good = legacy[leg_good]
    cat_mag_good = cat_mag[leg_good]
    if len(legacy_good) == 0:
        raise RuntimeError("No usable Legacy catalog rows after cuts")

    src_coord = SkyCoord(np.array(sources["ra"], dtype=float) * u.deg,
                         np.array(sources["dec"], dtype=float) * u.deg)
    leg_coord = SkyCoord(leg_ra[leg_good] * u.deg, leg_dec[leg_good] * u.deg)
    idx, sep2d, _ = src_coord.match_to_catalog_sky(leg_coord)
    matched = sep2d < match_radius_arcsec * u.arcsec

    rows = []
    for src_i, leg_i, sep in zip(np.where(matched)[0], idx[matched], sep2d[matched].arcsec):
        src = sources[src_i]
        flux = float(src["flux"])
        exptime = exptime_by_file.get(str(src["sci_file"]), 1.0)
        if not np.isfinite(flux) or flux <= 0 or exptime <= 0:
            continue
        inst_mag = -2.5 * math.log10(flux / exptime)
        zp = float(cat_mag_good[leg_i] - inst_mag)
        if not np.isfinite(zp):
            continue
        rows.append({
            "sci_file": str(src["sci_file"]),
            "image_num": int(src["image_num"]),
            "amp": int(src["amp"]),
            "x": float(src["x"]),
            "y": float(src["y"]),
            "ra": float(src["ra"]),
            "dec": float(src["dec"]),
            "legacy_ra": float(leg_coord.ra.deg[leg_i]),
            "legacy_dec": float(leg_coord.dec.deg[leg_i]),
            "sep_arcsec": float(sep),
            "flux": flux,
            "exptime": float(exptime),
            "inst_mag": float(inst_mag),
            "legacy_mag": float(cat_mag_good[leg_i]),
            "legacy_mag_col": mag_col,
            "zeropoint": zp,
        })
    if not rows:
        raise RuntimeError("No Bok/Legacy matches within the requested radius")
    return Table(rows)


def exptime_map(image_map: dict[int, Path]) -> dict[str, float]:
    out = {}
    for path in image_map.values():
        try:
            hdr = fits.getheader(path)
            out[path.name] = float(hdr.get("EXPTIME", hdr.get("EXPOSURE", 1.0)))
        except Exception:
            out[path.name] = 1.0
    return out


def summarize_matches(matches: Table, out_summary: Path) -> Table:
    rows = []
    for image_num in sorted(set(matches["image_num"])):
        sub = matches[matches["image_num"] == image_num]
        clipped = sigma_clip(np.array(sub["zeropoint"], dtype=float), sigma=3.0, maxiters=5)
        good = ~np.ma.getmaskarray(clipped)
        vals = np.array(sub["zeropoint"], dtype=float)[good]
        rows.append({
            "image_num": int(image_num),
            "sci_file": str(sub["sci_file"][0]),
            "zeropoint_median": float(np.nanmedian(vals)) if vals.size else np.nan,
            "zeropoint_std": float(np.nanstd(vals)) if vals.size else np.nan,
            "n_match": int(len(sub)),
            "n_clip": int(vals.size),
            "median_sep_arcsec": float(np.nanmedian(sub["sep_arcsec"])),
        })
    summary = Table(rows)
    summary.write(out_summary, overwrite=True)
    return summary


def plot_diagnostics(matches: Table, summary: Table, out_png: Path):
    zp = np.array(matches["zeropoint"], dtype=float)
    cat_mag = np.array(matches["legacy_mag"], dtype=float)
    sep = np.array(matches["sep_arcsec"], dtype=float)

    clipped = sigma_clip(zp, sigma=3.0, maxiters=5)
    good = ~np.ma.getmaskarray(clipped)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].scatter(cat_mag, zp, c=sep, s=14, alpha=0.75)
    axes[0].scatter(cat_mag[~good], zp[~good], s=18, facecolors="none", edgecolors="tab:red")
    axes[0].set_xlabel("Legacy catalog magnitude")
    axes[0].set_ylabel("Zero point")
    axes[0].grid(True, alpha=0.25)

    axes[1].hist(zp[good], bins=30, color="0.25", alpha=0.85)
    axes[1].axvline(np.nanmedian(zp[good]), color="tab:red", lw=2)
    axes[1].set_xlabel("Clipped zero point")
    axes[1].set_ylabel("N")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(summary["image_num"], summary["zeropoint_median"], "o-")
    axes[2].set_xlabel("Image number")
    axes[2].set_ylabel("Median zero point")
    axes[2].grid(True, alpha=0.25)

    fig.suptitle("Bok / Legacy Survey zero-point diagnostic")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", required=True, help="Pipeline focus_sources.fits")
    p.add_argument("--data-dir", required=True, help="Directory containing raw science FITS")
    p.add_argument("--time-series", help="Matching focus_time_series.ecsv; best way to map catalog rows to images")
    p.add_argument("--legacy-catalog", help="Local Legacy/Tractor/sweep catalog FITS/ECSV/CSV")
    p.add_argument("--query-vizier", action="store_true", help="Query VizieR instead of reading --legacy-catalog")
    p.add_argument("--vizier-catalog", default="VII/292/north", help="VizieR Legacy table for --query-vizier")
    p.add_argument("--filter", required=True, help="Photometric band, e.g. g/r/i/z")
    p.add_argument("--image-nums", nargs="*", help="Optional science image numbers to include, e.g. 172-201")
    p.add_argument("--name-contains", default="object", help="Filename filter if --time-series is not given")
    p.add_argument("--match-radius", type=float, default=1.0, help="Match radius in arcsec")
    p.add_argument("--max-fwhm", type=float, default=15.0)
    p.add_argument("--max-e", type=float, default=0.5)
    p.add_argument("--mag-min", type=float, default=None)
    p.add_argument("--mag-max", type=float, default=None)
    p.add_argument("--out", default="zeropoint_legacy_matches.ecsv")
    p.add_argument("--summary-out", default="zeropoint_legacy_summary.ecsv")
    p.add_argument("--plot-out", default="zeropoint_legacy_diagnostic.png")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    sources = Table.read(args.sources)
    file_idx, amp = parse_subset_ids(sources)
    if args.time_series:
        image_map = image_map_from_time_series(Path(args.time_series), data_dir)
    else:
        image_map = image_map_from_directory(data_dir, args.name_contains)

    wanted_nums = parse_num_ranges(args.image_nums)
    if wanted_nums is not None:
        keep_file_indices = {
            idx for idx, path in image_map.items()
            if image_number(path) in wanted_nums
        }
        row_keep = np.isin(file_idx, sorted(keep_file_indices))
        sources = sources[row_keep]
        file_idx = file_idx[row_keep]
        amp = amp[row_keep]
        image_map = {idx: path for idx, path in image_map.items() if idx in keep_file_indices}

    basic = apply_basic_source_cuts(sources, args.max_fwhm, args.max_e)
    sources = sources[basic]
    file_idx = file_idx[basic]
    amp = amp[basic]

    selected_rows: list[Table] = []
    selected_file_idx: list[np.ndarray] = []
    selected_amp: list[np.ndarray] = []
    for idx in sorted(set(file_idx[file_idx >= 0])):
        sub_mask = file_idx == idx
        sub = sources[sub_mask]
        gmm_mask = gmm_star_mask(sub)
        selected_rows.append(sub[gmm_mask])
        selected_file_idx.append(file_idx[sub_mask][gmm_mask])
        selected_amp.append(amp[sub_mask][gmm_mask])

    if not selected_rows:
        raise RuntimeError("No GMM-selected source rows")
    sources = vstack(selected_rows, join_type="outer")
    file_idx = np.concatenate(selected_file_idx)
    amp = np.concatenate(selected_amp)
    print(f"Using {len(sources)} GMM-selected Bok sources")

    sources_sky = add_sky_positions(sources, file_idx, amp, image_map)
    print(f"{len(sources_sky)} sources have valid WCS coordinates")

    if args.query_vizier:
        center = SkyCoord(np.nanmedian(sources_sky["ra"]) * u.deg, np.nanmedian(sources_sky["dec"]) * u.deg)
        radius = max(
            0.05,
            float(np.nanmax(center.separation(
                SkyCoord(np.array(sources_sky["ra"], dtype=float) * u.deg,
                         np.array(sources_sky["dec"], dtype=float) * u.deg)
            ).deg)) + 0.02,
        )
        legacy = query_vizier_legacy(center, radius, args.vizier_catalog)
    elif args.legacy_catalog:
        legacy = read_legacy_catalog(Path(args.legacy_catalog))
    else:
        raise ValueError("Provide --legacy-catalog, or use --query-vizier if astroquery is installed")

    matches = estimate_zeropoints(
        sources_sky,
        legacy,
        args.filter,
        exptime_map(image_map),
        args.match_radius,
        args.mag_min,
        args.mag_max,
    )
    matches.write(args.out, overwrite=True)
    summary = summarize_matches(matches, Path(args.summary_out))
    plot_diagnostics(matches, summary, Path(args.plot_out))

    print(f"Wrote matches: {args.out}")
    print(f"Wrote summary: {args.summary_out}")
    print(f"Wrote plot: {args.plot_out}")
    for row in summary:
        print(
            f"{row['sci_file']}: ZP={row['zeropoint_median']:.3f} "
            f"+/- {row['zeropoint_std']:.3f}, N={row['n_clip']}/{row['n_match']}"
        )


if __name__ == "__main__":
    main()
