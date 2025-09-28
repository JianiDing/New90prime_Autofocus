"""Small file utilities for scanning a directory of FITS files and categorizing them."""
from typing import List, Dict, Any
import os
from astropy.io import fits


def _safe_header_value(header, keys):
    for k in keys:
        v = header.get(k)
        if v:
            return str(v)
    return ''


def skim_fits_files(directory: str = './', Keyword: str = 'OBJECT', target_bands: List[str] = None, exclusion_list: List[str] = None) -> Dict[str, Any]:
    """Scan a directory and return a simple categorization of FITS files.

    The function uses simple heuristics: header keyword values and filename
    substrings to classify files as bias, dark, flat, or 'other'. Flats are grouped
    by band (detected from header keys FILTER/FILTER1/BAND or from filename).

    Returns a dict with keys: 'bias_frames', 'dark_frames', and one key per
    uppercase band in target_bands with a dict {'flat': [...], 'other': [...]}.
    """
    if target_bands is None:
        target_bands = []
    if exclusion_list is None:
        exclusion_list = []

    target_bands = [b.lower() for b in target_bands]

    categorized = { 'bias_frames': [], 'dark_frames': [] }
    for b in target_bands:
        categorized[b.upper()] = {'flat': [], 'other': []}

    for name in sorted(os.listdir(directory)):
        if name in exclusion_list:
            continue
        lname = name.lower()
        if not (lname.endswith('.fits') or lname.endswith('.fit')):
            continue
        path = os.path.join(directory, name)

        try:
            hdr = fits.getheader(path, ext=0)
        except Exception:
            # If header can't be read, skip
            continue

        # Look for common header keys
        imagetyp = _safe_header_value(hdr, ['IMAGETYP', 'IMAGETYPE', 'OBSTYPE', Keyword])
        obj = _safe_header_value(hdr, ['OBJECT', 'TARGET', 'IMAGETYP'])
        filt = _safe_header_value(hdr, ['FILTER', 'FILTER1', 'BAND'])

        keytext = (imagetyp or obj or name).lower()

        if 'bias' in keytext or 'zero' in keytext:
            categorized['bias_frames'].append(path)
            continue
        if 'dark' in keytext:
            categorized['dark_frames'].append(path)
            continue
        if 'flat' in keytext or 'twilight' in keytext or 'dome' in keytext:
            band = (filt or name).lower()
            # choose band by tokenizing
            selected = None
            for b in target_bands:
                if b in band:
                    selected = b.upper()
                    break
            if selected is None and target_bands:
                selected = target_bands[0].upper()
            if selected is None:
                # put in 'OTHER' bucket if no bands defined
                categorized.setdefault('OTHER', {'flat': [], 'other': []})['flat'].append(path)
            else:
                categorized[selected]['flat'].append(path)
            continue

        # default: add to first matching band 'other' bucket if we can
        assigned = False
        for b in target_bands:
            if b in keytext or b in name.lower():
                categorized[b.upper()]['other'].append(path)
                assigned = True
                break
        if not assigned:
            # fallback - put in first band other or an 'UNASSIGNED' list
            if target_bands:
                categorized[target_bands[0].upper()]['other'].append(path)
            else:
                categorized.setdefault('UNASSIGNED', []).append(path)

    return categorized


def get_band_frames(categorized: Dict[str, Any], band: str, kind: str = 'flat') -> List[str]:
    """Return the list of frames for a given band and kind ('flat' or 'other').

    band lookup is case-insensitive; returns empty list if not found.
    """
    if not band:
        return []
    key = band.upper()
    entry = categorized.get(key)
    if not entry:
        # try lowercase key
        entry = categorized.get(band.lower())
    if not entry:
        return []
    return entry.get(kind, [])
