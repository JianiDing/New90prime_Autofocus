# Autofocus Pipeline — Session Changes

Summary of fixes and features added to `focus_pipeline.py` and
`realtime_focus_monitor.py` during this session.

---

## 1. Real-time monitor (`realtime_focus_monitor.py`)

- **`--filter` defaults to `all`** — multi-band dashboard out of the box.
  Single-band (`--filter R`) and comma-list (`--filter R,G,I`) still work.
- **`--solve-tilt` flag** — forwarded to the pipeline so per-frame tilt
  artifacts are pre-computed.
- **HTTP server endpoints** built into the monitor (port `--port`,
  default 8000):
  - `GET *.html / *.png / *.json` → static files from `--outdir`.
  - `POST /fit_selected` → fast parabola fit from
    `focus_time_series.ecsv` (no full pipeline rerun).
  - `POST /solve_tilt` → kept as a legacy fallback.
- **Empty-start watching** — launching with no science frames is fine.
  As soon as a new `*object*.fits` lands, watchdog detects it,
  debounces (`--debounce-seconds`, default 10 s), waits for writes to
  settle (`--settle-seconds`, default 3 s), then runs the pipeline and
  updates the dashboard. Calibrations (bias/dark/flat) must already be
  in the directory.

## 2. Focus fit

- **Focus axis = mean of LVDTA, LVDTB, LVDTC** when `--focus-key` is
  the default (`LVDTC`) and all three keys are present in the header.
  Override with e.g. `--focus-key LVDTA` to use a single actuator.
- Parabola fit `FWHM(x) = A·(x − h)² + k` with `h` = best focus,
  reported in LVDT units.
- **"Fit Selected" button** in dashboard fits a parabola through
  user-selected frames in milliseconds (reads ECSV, no pipeline rerun).

## 3. Tilt + focus solver (`solve_tilt_focus`)

Model

$$
\text{FWHM}^2(x,y) = \text{FWHM}_0^2 + \alpha\,\delta z(x,y)^2,
\qquad
\delta z(x,y) = z_0 + a\,x + b\,y
$$

- **Fixed degenerate-fit bug**: the previous initial guess
  `(z₀, a, b) = 0` made the LM Jacobian vanish, so every frame returned
  zero tilt and zero corrections. Now the seed comes from a linear
  plane fit to FWHM², plus a sign-flipped retry to break the
  ±(z₀, a, b) sign degeneracy. R² and tilt magnitudes are now non-zero
  and physically meaningful.
- **Per-frame pre-compute**: `--solve-tilt` (no `--solve-tilt-frame`)
  loops over every science frame, writing
  - `tilt_result_<num>.json` — current/optimal LVDT, corrections,
    tilt magnitude, R², seeing floor, etc.
  - `tilt_map_<num>.png` — 3-panel figure (FWHM map, defocus map,
    actuator bar chart).
  - Default `tilt_result.json` / `tilt_map.png` point at the middle
    frame.

### Tilt-map figure tweaks

- Larger figure (`figsize=(22, 8)`), bigger circles (`s=900`).
- `set_aspect("equal")` removed; padded x/y limits so the 8 amps aren't
  squashed to the edges.
- Larger fonts on titles, axis labels, ticks.
- Per-circle text overlays removed (cleaner look — colour + colorbar
  are sufficient).

## 4. Dashboard UX (`focus_time_series.html`)

- **Per-filter colored traces** (g/i/r/u/z) — works automatically when
  the `filter` column exists in the ECSV.
- **Bottom-right tilt info panel removed**.
- **Click a data point → tilt map opens directly** (full-screen
  overlay, click anywhere to close). Loads `tilt_map_<frame>.png` for
  the clicked frame.
- **Hover → PSF thumbnail overlay** (unchanged, still works from
  `psf_thumbnails.json`).
- **Box / lasso select → "Fit Selected"** runs the fast ECSV parabola
  fit and updates `focus_fit.png`.

## 5. Output artifact summary (per pipeline run)

```
focus_output/
  focus_sources.fits           # SEP catalog
  focus_time_series.ecsv       # one row per (frame, amp)
  focus_time_series.html       # interactive dashboard
  focus_time_series.png        # static QA plot
  focus_fit.png                # parabola through selected frames
  psf_thumbnails.json          # base64 PSF cutouts (cached)
  tilt_result.json             # default tilt (middle frame)
  tilt_map.png                 # default tilt map
  tilt_result_<num>.json       # per-frame tilt JSON   (one per frame)
  tilt_map_<num>.png           # per-frame tilt figure (one per frame)
```

## 6. How to use the actuator corrections

For each frame, `tilt_result_<num>.json["corrections"]` gives the
LVDT-unit deltas to apply to actuators A, B, C so the focal plane is
both pistoned and tilted to flat:

```json
"corrections": {"A": +1.10, "B": +0.70, "C": +0.22}
```

Equivalently, set each actuator to `optimal_lvdt[A/B/C]`.

**Recommended practice**: average corrections across several
high-`R²` (> ~0.4) frames at the same nominal LVDT setting before
commanding a move — single-frame fits are noisy and the sign of
`(z₀, a, b)` can flip frame to frame because only `δz²` enters the
model.

## 7. Typical launch command

```bash
python realtime_focus_monitor.py \
  --data-dir /path/to/night/ \
  --bias-nums 1-10 --dark-nums 21-22 --flat-nums 91-100 \
  --name-contains object --mask-dir bad_pixel_masks \
  --date-key DATE-OBS --pixscale 0.455 --solve-tilt
# Open: http://localhost:8000/focus_time_series.html
```

`--filter all` is implicit; add `--port 8001` (or any free port) to
avoid a clash with another instance.
