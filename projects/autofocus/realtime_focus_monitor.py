#!/usr/bin/env python3
"""Watch a data directory and re-run focus_pipeline whenever new science images arrive."""

from __future__ import annotations

import argparse
import errno
import fcntl
import http.server
import json
import os
import re
import signal
import socketserver
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Sequence, Set

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError as exc:  # pragma: no cover - user guidance path
    raise SystemExit(
        "watchdog is required: pip install watchdog"
    ) from exc

IMAGE_REGEX = re.compile(r"(\d+)")
PIPELINE_BUSY_MESSAGE = (
    "Pipeline is already running for this dashboard. Wait for it to finish "
    "before starting another fit/tilt request."
)


def extract_image_number(filename: str) -> Optional[int]:
    stem = Path(filename).stem
    matches = IMAGE_REGEX.findall(stem)
    if not matches:
        return None
    return int(matches[-1])


def build_pipeline_command(args: argparse.Namespace, sci_numbers: Sequence[int]) -> List[str]:
    base_cmd: List[str] = []
    if getattr(args, "pipeline_nice", 0):
        base_cmd += ["nice", "-n", str(args.pipeline_nice)]
    base_cmd += [args.python_bin, str(Path(args.pipeline_script).resolve())]
    base_cmd += ["--data-dir", str(Path(args.data_dir).resolve())]
    base_cmd += ["--filter", args.filter]
    base_cmd += ["--bias-nums", *args.bias_nums]
    if args.dark_nums:
        base_cmd += ["--dark-nums", *args.dark_nums]
    if args.flat_nums:
        base_cmd += ["--flat-nums", *args.flat_nums]
    base_cmd += ["--sci-nums", *(str(num) for num in sci_numbers)]
    base_cmd += ["--mask-dir", str(Path(args.mask_dir).resolve())]
    base_cmd += ["--outdir", str(Path(args.outdir).resolve())]
    base_cmd += ["--focus-key", args.focus_key]
    base_cmd += ["--time-key", args.time_key]
    base_cmd += ["--date-key", args.date_key]
    base_cmd += ["--pixscale", str(args.pixscale)]
    base_cmd += ["--airmass-key", args.airmass_key]
    base_cmd += ["--threshold", str(args.threshold)]
    base_cmd += ["--fwhm-method", args.fwhm_method]
    base_cmd += ["--gmm-fwhm-method", args.gmm_fwhm_method]
    if args.amps:
        base_cmd += ["--amps", *(str(amp) for amp in args.amps)]
    if args.auto_generate_masks:
        base_cmd.append("--auto-generate-masks")
    base_cmd += ["--mask-sat-mult", str(args.mask_sat_mult)]
    base_cmd += ["--mask-black-mult", str(args.mask_black_mult)]
    base_cmd += ["--mask-sat-frac", str(args.mask_sat_frac)]
    base_cmd += ["--mask-black-frac", str(args.mask_black_frac)]
    if args.write_reduced:
        base_cmd.append("--write-reduced")
    if args.skip_reduced:
        base_cmd.append("--skip-reduced")
    if args.incremental:
        base_cmd.append("--incremental")
    if args.no_fit:
        base_cmd.append("--no-fit")
    if getattr(args, "solve_tilt", False):
        base_cmd.append("--solve-tilt")
    if getattr(args, "global_tilt_fit", False):
        base_cmd.append("--global-tilt-fit")
    return base_cmd


@contextmanager
def outdir_pipeline_lock(outdir: str):
    """Non-blocking lockfile so two monitor processes cannot share one outdir."""
    lock_path = Path(outdir) / ".pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise RuntimeError(
                    f"Another pipeline process is already using {lock_path.parent}"
                ) from exc
            raise
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"{time.time():.3f}\n")
            handle.flush()
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def monitor_pipeline_slot(shared: dict):
    """Non-blocking in-process lock shared by watcher and HTTP handlers."""
    lock = shared.setdefault("pipeline_lock", threading.Lock())
    acquired = lock.acquire(blocking=False)
    if not acquired:
        raise RuntimeError(PIPELINE_BUSY_MESSAGE)
    try:
        yield
    finally:
        lock.release()


def pipeline_is_busy(shared: dict) -> bool:
    lock = shared.setdefault("pipeline_lock", threading.Lock())
    return lock.locked()


def run_pipeline_subprocess(
    cmd: List[str],
    shared: dict,
    *,
    capture_output: bool = False,
    timeout: Optional[float] = None,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Run one child pipeline and remember it so shutdown can terminate it."""
    stdout = subprocess.PIPE if capture_output else None
    stderr = subprocess.PIPE if capture_output else None
    proc = subprocess.Popen(
        cmd,
        stdout=stdout,
        stderr=stderr,
        text=True,
        start_new_session=True,
    )
    shared["active_process"] = proc
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate_active_process(shared, reason="timeout")
        out, err = proc.communicate()
        raise
    finally:
        if shared.get("active_process") is proc:
            shared["active_process"] = None

    result = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr
        )
    return result


def terminate_active_process(shared: dict, reason: str = "shutdown") -> None:
    proc = shared.get("active_process")
    if proc is None or proc.poll() is not None:
        return
    print(f"[monitor] Terminating active pipeline process ({reason})")
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print("[monitor] Active pipeline did not stop; killing it")
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait(timeout=10)


def make_http_handler(args: argparse.Namespace, shared: dict):
    """Return a combined request handler: serves static files from outdir + POST /fit_selected."""
    outdir_str = str(Path(args.outdir).resolve())

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=outdir_str, **kw)

        def do_GET(self):  # type: ignore[override]
            if self.path in {"/favicon.ico", "favicon.ico"}:
                self.send_response(204)
                self.end_headers()
                return
            super().do_GET()

        def do_POST(self):  # type: ignore[override]
            if self.path == "/fit_selected":
                self._handle_fit()
            elif self.path == "/solve_tilt":
                self._handle_solve_tilt()
            elif self.path == "/amp_fwhm":
                self._handle_amp_fwhm()
            else:
                self.send_error(404)

        def do_OPTIONS(self):  # type: ignore[override]
            self.send_response(204)
            self.end_headers()

        def _handle_fit(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                frames = [int(x) for x in body.get("frames", [])]
            except Exception as exc:
                self._json(400, {"error": f"Bad request: {exc}"})
                return
            if len(frames) < 3:
                self._json(400, {"error": "Need at least 3 frames for a parabola fit"})
                return
            if pipeline_is_busy(shared):
                self._json(409, {"error": PIPELINE_BUSY_MESSAGE})
                return
            Path(outdir_str, "selected_frames.txt").write_text(
                " ".join(str(x) for x in frames)
            )
            # Fast path: fit directly from existing ECSV — no pipeline re-run needed
            ecsv_path = Path(outdir_str) / "focus_time_series.ecsv"
            if ecsv_path.exists():
                try:
                    self._fit_from_ecsv(ecsv_path, frames)
                    return
                except Exception as exc:
                    print(f"[server] Fast fit failed ({exc}), falling back to pipeline")
            # Fallback: run full pipeline
            all_nums = sorted(shared.get("image_numbers", set()) or set(frames))
            cmd = [c for c in build_pipeline_command(args, all_nums) if c != "--no-fit"]
            cmd += ["--fit-nums"] + [str(f) for f in frames]
            print(f"[server] Running pipeline fit for {len(frames)} selected frames: {frames}")
            try:
                with monitor_pipeline_slot(shared), outdir_pipeline_lock(outdir_str):
                    result = run_pipeline_subprocess(
                        cmd,
                        shared,
                        capture_output=True,
                        timeout=600,
                    )
                if result.returncode != 0:
                    err = (result.stderr or result.stdout or "")[-800:]
                    print(f"[server] Fit failed:\n{err}")
                    self._json(500, {"error": err})
                else:
                    print("[server] Fit complete")
                    per_amp = None
                    if ecsv_path.exists():
                        per_amp = self._fit_per_amp_for_frames(ecsv_path, frames)
                    payload = {"status": "ok", "frames": frames}
                    if per_amp:
                        payload["per_amp"] = per_amp
                    self._json(200, payload)
            except RuntimeError as exc:
                self._json(409, {"error": str(exc)})
            except subprocess.TimeoutExpired:
                self._json(500, {"error": "Pipeline timed out (600 s)"})

        def _fit_from_ecsv(self, ecsv_path: Path, frames: list) -> None:
            """Fit parabola directly from focus_time_series.ecsv — instant, no pipeline re-run."""
            import re as _re
            from astropy.table import Table as _Table
            from scipy.optimize import curve_fit as _curve_fit
            import numpy as _np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt

            tbl = _Table.read(str(ecsv_path), format="ascii.ecsv")
            if "focus_position" not in tbl.colnames:
                raise ValueError("focus_position column not in ECSV — regenerate dashboard first")

            _rnum = _re.compile(r"(\d+)")
            frame_set = set(frames)
            mask = _np.array([
                bool(_rnum.findall(Path(str(row["sci_file"])).stem) and
                     int(_rnum.findall(Path(str(row["sci_file"])).stem)[-1]) in frame_set)
                for row in tbl
            ])
            sub = tbl[mask]
            # Deduplicate by sci_file (ECSV may have one row per amp)
            seen = set()
            rows_x, rows_y = [], []
            for row in sub:
                fn = str(row["sci_file"])
                if fn not in seen:
                    seen.add(fn)
                    rows_x.append(float(row["focus_position"]))
                    rows_y.append(float(row["avg_fwhm"]) * args.pixscale)
            if len(rows_x) < 3:
                self._json(400, {"error": "Not enough frames with valid focus_position"})
                return

            x = _np.array(rows_x)
            y = _np.array(rows_y)
            finite = _np.isfinite(x) & _np.isfinite(y)
            x, y = x[finite], y[finite]
            if len(x) < 3:
                self._json(400, {"error": "Not enough finite focus positions"})
                return
            sort_idx = _np.argsort(x)
            x, y = x[sort_idx], y[sort_idx]

            # Parabola fit
            def _parabola(xx, A, h, k):
                return A * (xx - h) ** 2 + k
            try:
                h0 = x[_np.argmin(y)]
                p0 = [1.0, h0, float(_np.min(y))]
                popt, pcov = _curve_fit(_parabola, x, y, p0=p0, maxfev=10000)
                A, h, k = popt
                y_pred = _parabola(x, *popt)
                ss_res = _np.sum((y - y_pred) ** 2)
                ss_tot = _np.sum((y - _np.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else _np.nan
            except Exception as exc:
                self._json(500, {"error": f"Fit failed: {exc}"})
                return

            # Plot
            fig, ax = _plt.subplots(figsize=(7, 4))
            ax.scatter(x, y, color="#1f77b4", zorder=5, label="Selected frames")
            x_fine = _np.linspace(x.min(), x.max(), 300)
            ax.plot(x_fine, _parabola(x_fine, A, h, k), color="crimson",
                    label=f"Best focus: {h:.3f} (R²={r2:.3f})")
            ax.axvline(h, color="crimson", linestyle="--", alpha=0.5)
            ax.set_xlabel("Focus position")
            ax.set_ylabel("FWHM (arcsec)")
            ax.set_title("Focus curve (selected frames)")
            ax.legend()
            fig.tight_layout()
            plot_path = Path(outdir_str) / "focus_fit.png"
            fig.savefig(str(plot_path), dpi=120)
            _plt.close(fig)

            per_amp = self._fit_per_amp_for_frames(ecsv_path, frames)

            print(f"[server] Fast fit complete: best focus={h:.3f}, R²={r2:.3f}")
            payload = {"status": "ok", "frames": frames,
                       "best_focus": round(h, 4), "r2": round(r2, 4)}
            if per_amp:
                payload["per_amp"] = per_amp
            self._json(200, payload)

        def _fit_per_amp_for_frames(self, ecsv_path: Path, frames: list) -> Optional[dict]:
            """Create the per-amplifier best-focus figure for selected frames."""
            sources_path = Path(outdir_str) / "focus_sources.fits"
            if not sources_path.exists():
                print("[server] Per-amp fit skipped: focus_sources.fits not found")
                return None
            try:
                from astropy.table import Table as _Table
                import numpy as _np
                from fit_focus_per_amp_from_monitor import (
                    average_focus_and_error as _average_focus_and_error,
                    build_per_amp_points as _build_per_amp_points,
                    fit_per_amp as _fit_per_amp,
                    plot_per_amp_fits as _plot_per_amp_fits,
                )

                sources = _Table.read(str(sources_path))
                time_series = _Table.read(str(ecsv_path), format="ascii.ecsv")
                amps = list(args.amps) if args.amps else list(range(1, 9))
                points = _build_per_amp_points(
                    sources=sources,
                    time_series=time_series,
                    amps=amps,
                    fit_numbers=set(int(f) for f in frames),
                    fwhm_max=15.0,
                    flag_max=0,
                    flux_ratio_min=1.0,
                )
                fits = _fit_per_amp(points, amps, min_points=3)
                avg, avg_err, n_good = _average_focus_and_error(fits)
                fits.meta["average_best_focus"] = avg
                fits.meta["average_best_focus_error_sem"] = avg_err
                fits.meta["n_good_amps"] = n_good
                fits.meta["fit_numbers"] = list(frames)

                points_path = Path(outdir_str) / "selected_focus_per_amp_points.ecsv"
                summary_path = Path(outdir_str) / "selected_focus_per_amp_best_focus.ecsv"
                plot_path = Path(outdir_str) / "selected_focus_per_amp_best_focus.png"
                points.write(points_path, format="ascii.ecsv", overwrite=True)
                fits.write(summary_path, format="ascii.ecsv", overwrite=True)
                _plot_per_amp_fits(
                    points,
                    fits,
                    plot_path,
                    average_best_focus=avg,
                    average_best_focus_error=avg_err,
                    n_good_amps=n_good,
                )
                print(
                    f"[server] Per-amp fit complete: average={avg:.3f}"
                    + (f" ± {avg_err:.3f}" if _np.isfinite(avg_err) else "")
                )
                return {
                    "average_best_focus": round(float(avg), 4) if _np.isfinite(avg) else None,
                    "average_best_focus_error_sem": (
                        round(float(avg_err), 4) if _np.isfinite(avg_err) else None
                    ),
                    "n_good_amps": int(n_good),
                    "plot": plot_path.name,
                    "summary": summary_path.name,
                    "points": points_path.name,
                }
            except Exception as exc:
                print(f"[server] Per-amp fit failed: {exc}")
                return None

        def _amp_fwhm_for_frame(self, ecsv_path: Path, frame: int) -> dict:
            """Return per-amplifier median FWHM for one clicked frame, no files written."""
            sources_path = Path(outdir_str) / "focus_sources.fits"
            if not sources_path.exists():
                raise FileNotFoundError("focus_sources.fits not found; run the pipeline first")

            from astropy.table import Table as _Table
            import numpy as _np
            from fit_focus_per_amp_from_monitor import (
                build_per_amp_points as _build_per_amp_points,
            )

            cache_key = (
                int(frame),
                sources_path.stat().st_mtime,
                ecsv_path.stat().st_mtime,
                tuple(args.amps) if args.amps else tuple(range(1, 9)),
            )
            amp_cache = shared.setdefault("amp_fwhm_cache", {})
            if cache_key in amp_cache:
                return amp_cache[cache_key]

            def _cached_table(name: str, path: Path, fmt: Optional[str] = None):
                table_key = f"{name}_table"
                mtime_key = f"{name}_mtime"
                mtime = path.stat().st_mtime
                if shared.get(mtime_key) != mtime or table_key not in shared:
                    if fmt:
                        shared[table_key] = _Table.read(str(path), format=fmt)
                    else:
                        shared[table_key] = _Table.read(str(path))
                    shared[mtime_key] = mtime
                    amp_cache.clear()
                return shared[table_key]

            sources = _cached_table("focus_sources", sources_path)
            time_series = _cached_table("focus_time_series", ecsv_path, fmt="ascii.ecsv")
            amps = list(args.amps) if args.amps else list(range(1, 9))
            points = _build_per_amp_points(
                sources=sources,
                time_series=time_series,
                amps=amps,
                fit_numbers={int(frame)},
                fwhm_max=15.0,
                flag_max=0,
                flux_ratio_min=1.0,
            )
            if len(points) == 0:
                raise ValueError(f"No per-amplifier FWHM data found for frame {frame}")

            rows = []
            sci_file = str(points["sci_file"][0])
            focus_position = float(points["focus_position"][0])
            for row in points:
                fwhm_pix = float(row["median_fwhm"])
                fwhm_arcsec = fwhm_pix * args.pixscale if _np.isfinite(fwhm_pix) else _np.nan
                rows.append({
                    "amp": int(row["amp"]),
                    "fwhm_pix": round(fwhm_pix, 4) if _np.isfinite(fwhm_pix) else None,
                    "fwhm_arcsec": round(float(fwhm_arcsec), 4) if _np.isfinite(fwhm_arcsec) else None,
                    "median_e": (
                        round(float(row["median_e"]), 4)
                        if _np.isfinite(float(row["median_e"]))
                        else None
                    ),
                    "n_stars": int(row["n_stars"]),
                })

            valid = _np.array([r["fwhm_pix"] for r in rows if r["fwhm_pix"] is not None], dtype=float)
            avg_pix = float(_np.nanmedian(valid)) if valid.size else _np.nan
            avg_arcsec = avg_pix * args.pixscale if _np.isfinite(avg_pix) else _np.nan
            payload = {
                "frame": int(frame),
                "sci_file": sci_file,
                "focus_position": round(focus_position, 4),
                "pixscale": args.pixscale,
                "avg_fwhm_pix": round(avg_pix, 4) if _np.isfinite(avg_pix) else None,
                "avg_fwhm_arcsec": round(float(avg_arcsec), 4) if _np.isfinite(avg_arcsec) else None,
                "amps": rows,
            }
            amp_cache[cache_key] = payload
            return payload

        def _handle_amp_fwhm(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                frame = int(body.get("frame"))
            except Exception as exc:
                self._json(400, {"error": f"Bad request: {exc}"})
                return
            ecsv_path = Path(outdir_str) / "focus_time_series.ecsv"
            if not ecsv_path.exists():
                self._json(404, {"error": "focus_time_series.ecsv not found"})
                return
            if pipeline_is_busy(shared):
                self._json(409, {"error": PIPELINE_BUSY_MESSAGE})
                return
            try:
                payload = self._amp_fwhm_for_frame(ecsv_path, frame)
                self._json(200, payload)
            except Exception as exc:
                print(f"[server] Amp FWHM lookup failed for frame {frame}: {exc}")
                self._json(500, {"error": str(exc)})

        def _json(self, code: int, data: dict) -> None:
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle_solve_tilt(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                frame = int(body.get("frame"))
            except Exception as exc:
                self._json(400, {"error": f"Bad request: {exc}"})
                return
            if pipeline_is_busy(shared):
                self._json(409, {"error": PIPELINE_BUSY_MESSAGE})
                return
            all_nums = sorted(shared.get("image_numbers", set()) or {frame})
            if frame not in all_nums:
                all_nums = sorted(set(all_nums) | {frame})
            cmd = [c for c in build_pipeline_command(args, all_nums) if c != "--no-fit"]
            cmd += ["--solve-tilt", "--solve-tilt-frame", str(frame), "--no-fit"]
            print(f"[server] Solving tilt for frame {frame} ...")
            try:
                with monitor_pipeline_slot(shared), outdir_pipeline_lock(outdir_str):
                    result = run_pipeline_subprocess(
                        cmd,
                        shared,
                        capture_output=True,
                        timeout=600,
                    )
                if result.returncode != 0:
                    err = (result.stderr or result.stdout or "")[-800:]
                    print(f"[server] Tilt solve failed:\n{err}")
                    self._json(500, {"error": err})
                    return
                print(f"[server] Tilt solve complete for frame {frame}")
                self._json(200, {"status": "ok", "frame": frame})
            except RuntimeError as exc:
                self._json(409, {"error": str(exc)})
            except subprocess.TimeoutExpired:
                self._json(500, {"error": "Pipeline timed out (600 s)"})

        def log_message(self, fmt, *a):  # type: ignore[override]
            pass  # suppress per-request log noise

    return _Handler


def should_track(path: Path, includes: Optional[List[str]], excludes: Optional[List[str]]) -> bool:
    stem = path.name.lower()
    if includes and not any(token in stem for token in includes):
        return False
    if excludes and any(token in stem for token in excludes):
        return False
    return path.suffix.lower() in {".fits", ".fit"}


def collect_existing_numbers(args: argparse.Namespace) -> Set[int]:
    files = sorted(Path(args.data_dir).glob(args.science_glob))
    numbers: Set[int] = set()
    for path in files:
        if not should_track(path, args.name_contains, args.ignore_contains):
            continue
        num = extract_image_number(path.name)
        if num is not None:
            numbers.add(num)
    return numbers


def collect_previous_output_numbers(outdir: str) -> Set[int]:
    """Read frame numbers from an existing dashboard time series, if present."""
    path = Path(outdir) / "focus_time_series.ecsv"
    if not path.exists():
        return set()
    try:
        from astropy.table import Table as _Table

        table = _Table.read(str(path), format="ascii.ecsv")
        if "sci_file" in table.colnames:
            numbers = set()
            for value in table["sci_file"]:
                num = extract_image_number(str(value))
                if num is not None:
                    numbers.add(num)
            return numbers
    except Exception:
        pass

    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return set()
    numbers: Set[int] = set()
    fits_re = re.compile(r"([A-Za-z0-9_.-]*?(\d+)\.fits?)")
    for match in fits_re.finditer(text):
        try:
            numbers.add(int(match.group(2)))
        except ValueError:
            continue
    return numbers


class ScienceWatcher(FileSystemEventHandler):
    def __init__(
        self,
        args: argparse.Namespace,
        seen_numbers: Set[int],
        initial_pipeline_numbers: Set[int],
        shared: dict,
    ) -> None:
        super().__init__()
        self.args = args
        self.shared = shared
        self.seen_numbers: Set[int] = set(seen_numbers)
        self.image_numbers: Set[int] = set(initial_pipeline_numbers)
        self.lock = threading.Lock()
        self.timer: Optional[threading.Timer] = None
        self.pipeline_running = False
        self.rerun_requested = False
        self.running_numbers: Set[int] = set()

    def on_created(self, event):  # type: ignore[override]
        if getattr(event, "is_directory", False):
            return
        self._handle_event(Path(event.src_path))

    def on_moved(self, event):  # type: ignore[override]
        if getattr(event, "is_directory", False):
            return
        self._handle_event(Path(event.dest_path))

    def _handle_event(self, path: Path) -> None:
        if not should_track(path, self.args.name_contains, self.args.ignore_contains):
            return
        num = extract_image_number(path.name)
        if num is None:
            return
        with self.lock:
            if num in self.seen_numbers:
                return
            self.seen_numbers.add(num)
            self.image_numbers.add(num)
        print(f"[monitor] Detected new science file {path.name} (#{num})")
        self._schedule_run()

    def _schedule_run(self) -> None:
        with self.lock:
            if self.timer and self.timer.is_alive():
                self.timer.cancel()
            self.timer = threading.Timer(self.args.debounce_seconds, self._run_pipeline)
            self.timer.start()

    def _run_pipeline(self) -> None:
        with self.lock:
            if self.pipeline_running:
                self.rerun_requested = True
                print("[monitor] Pipeline already running; queued one follow-up run")
                return
            self.pipeline_running = True
            numbers = sorted(self.image_numbers)
            self.running_numbers = set(numbers)
        if not numbers:
            with self.lock:
                self.pipeline_running = False
                self.running_numbers = set()
            return
        time.sleep(self.args.settle_seconds)
        cmd = build_pipeline_command(self.args, numbers)
        print(f"[monitor] Running pipeline for {len(numbers)} science frames...")
        try:
            with monitor_pipeline_slot(self.shared), outdir_pipeline_lock(self.args.outdir):
                run_pipeline_subprocess(cmd, self.shared, check=True)
            print("[monitor] Pipeline run complete")
        except RuntimeError as exc:
            print(f"[monitor] Pipeline skipped: {exc}")
        except subprocess.CalledProcessError as exc:
            print(f"[monitor] Pipeline failed with code {exc.returncode}")
        finally:
            with self.lock:
                pending_numbers = set(self.image_numbers) - self.running_numbers
                self.pipeline_running = False
                rerun = self.rerun_requested
                self.rerun_requested = False
                self.running_numbers = set()
            if rerun:
                if pending_numbers:
                    print(
                        f"[monitor] Starting queued follow-up run "
                        f"({len(pending_numbers)} new frame(s): "
                        f"{sorted(pending_numbers)[:10]})"
                    )
                    self._schedule_run()
                else:
                    print("[monitor] Queued follow-up skipped; no new frames pending")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run focus_pipeline whenever new science exposures arrive.",
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing FITS files")
    parser.add_argument("--filter", default="all",
                        help="Photometric filter(s): single band (e.g. R), "
                             "comma-separated list (e.g. R,G,I), or 'all' to "
                             "show every band in one dashboard (default: all)")
    parser.add_argument("--bias-nums", nargs="+", required=True, help="Bias image numbers")
    parser.add_argument("--dark-nums", nargs="*", default=None, help="Dark image numbers (optional; omit if no darks)")
    parser.add_argument(
        "--flat-nums",
        nargs="+",
        default=None,
        help=(
            "Flat image numbers. If omitted in multi-band/all mode, "
            "focus_pipeline.py auto-discovers flats by band."
        ),
    )
    parser.add_argument("--mask-dir", default="bad_pixel_masks", help="Bad mask directory")
    parser.add_argument("--outdir", default="focus_output", help="Output directory for products")
    parser.add_argument("--focus-key", default="LVDTC", help="Focus header keyword")
    parser.add_argument("--time-key", default="TIME-OBS", help="Time header keyword")
    parser.add_argument("--date-key", default="DATE-OBS", help="Date header keyword")
    parser.add_argument("--airmass-key", default="AIRMASS", help="Airmass header keyword")
    parser.add_argument(
        "--pixscale",
        type=float,
        default=0.455,
        help="Plate scale in arcsec/pixel (default: 0.455 for Bok 90Prime)",
    )
    parser.add_argument("--threshold", type=float, default=25.0, help="SEP threshold")
    parser.add_argument(
        "--fwhm-method",
        choices=("direct", "moffat", "gaussian"),
        default="direct",
        help=(
            "Per-star FWHM method passed to focus_pipeline.py. 'direct' is "
            "the default half-maximum crossing; 'moffat' fits a Moffat "
            "profile; 'gaussian' uses the previous Gaussian radial fit."
        ),
    )
    parser.add_argument(
        "--gmm-fwhm-method",
        choices=("same", "direct", "moffat", "gaussian"),
        default="same",
        help=(
            "FWHM feature passed to the GMM star selector. 'same' reuses "
            "--fwhm-method; use 'gaussian' with '--fwhm-method direct' for "
            "Gaussian selection plus direct reported seeing."
        ),
    )
    parser.add_argument(
        "--allow-slow-moffat",
        action="store_true",
        help=(
            "Allow realtime monitor runs that use Moffat FWHM fitting. This can "
            "be extremely slow for full-frame/all-band reductions."
        ),
    )
    parser.add_argument("--amps", nargs="+", type=int, default=None, help="Amplifier list")
    parser.add_argument(
        "--mask-sat-mult",
        type=float,
        default=1.0,
        help="Mask saturation multiplier",
    )
    parser.add_argument(
        "--mask-black-mult",
        type=float,
        default=4.0,
        help="Mask black-column multiplier",
    )
    parser.add_argument(
        "--mask-sat-frac",
        type=float,
        default=0.25,
        help="Saturated column fraction",
    )
    parser.add_argument(
        "--mask-black-frac",
        type=float,
        default=0.2,
        help="Black column fraction",
    )
    parser.add_argument("--auto-generate-masks", action="store_true", help="Force mask regen")
    parser.add_argument("--write-reduced", action="store_true", help="Write reduced FITS files")
    parser.add_argument(
        "--no-fit",
        action="store_true",
        default=False,
        help="Skip automatic focus fit; select frames manually from the dashboard",
    )
    parser.add_argument(
        "--solve-tilt",
        action="store_true",
        default=False,
        help="Pre-compute focal-plane tilt + actuator corrections for every "
             "science frame.  Writes tilt_result_<num>.json and tilt_map_<num>.png "
             "so the dashboard can show per-frame tilt instantly on click.",
    )
    parser.add_argument(
        "--global-tilt-fit",
        action="store_true",
        default=False,
        help="With --solve-tilt: jointly fit one (FWHM_0, alpha) across all "
             "frames and per-frame (z0, a, b).  Breaks the seeing/piston "
             "degeneracy; recommended for focus-scan sequences (≥ 5 frames "
             "at varying LVDT).",
    )
    parser.add_argument(
        "--skip-reduced",
        action="store_true",
        help="Skip writing reduced FITS (overrides --write-reduced)",
    )
    parser.add_argument(
        "--pipeline-script",
        default="focus_pipeline.py",
        help="Path to focus_pipeline.py",
    )
    parser.add_argument(
        "--pipeline-nice",
        type=int,
        default=5,
        help=(
            "Run child pipeline jobs through nice with this priority increment "
            "(default: 5). Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used to run the pipeline",
    )
    parser.add_argument(
        "--science-glob",
        default="*.fits",
        help="Glob used to find candidate science files",
    )
    parser.add_argument(
        "--name-contains",
        nargs="*",
        default=None,
        help="Only react to filenames containing any of these substrings",
    )
    parser.add_argument(
        "--ignore-contains",
        nargs="*",
        default=None,
        help="Ignore filenames containing these substrings",
    )
    parser.add_argument(
        "--debounce-seconds",
        type=float,
        default=10.0,
        help="Wait time before rerunning after a burst of files",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=3.0,
        help="Delay to allow file writes to finish before running",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=True,
        help="Use incremental mode: only reduce new exposures (default: on)",
    )
    parser.add_argument(
        "--no-incremental",
        dest="incremental",
        action="store_false",
        help="Disable incremental mode; reprocess everything each run",
    )
    parser.add_argument(
        "--no-initial-run",
        action="store_true",
        help=(
            "Deprecated alias for the default behavior: serve/watch without "
            "processing files that already existed when the monitor started."
        ),
    )
    parser.add_argument(
        "--initial-run",
        action="store_true",
        help=(
            "Immediately process all existing matching science files, then watch "
            "for new ones. Use this only when you intentionally want to reduce "
            "the current backlog."
        ),
    )
    parser.add_argument(
        "--no-resume-output",
        action="store_true",
        help=(
            "Do not seed the next pipeline run from an existing "
            "focus_time_series.ecsv in --outdir. By default, the monitor resumes "
            "the previous dashboard frames and appends new exposures."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the web dashboard (default: 8000).  Open "
             "http://localhost:PORT/focus_time_series.html in your browser.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data directory {data_dir} does not exist")
    if (
        (args.fwhm_method == "moffat" or args.gmm_fwhm_method == "moffat")
        and not args.allow_slow_moffat
    ):
        raise SystemExit(
            "Refusing realtime Moffat mode because it can overload the server. "
            "Use --allow-slow-moffat only for small subsets or intentional tests."
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve paths so the HTTP handler can find them regardless of cwd
    for attr in ("data_dir", "mask_dir", "outdir", "pipeline_script"):
        if getattr(args, attr, None):
            setattr(args, attr, str(Path(getattr(args, attr)).resolve()))

    # Clean up any stale selection file from previous run
    stale = outdir / "selected_frames.txt"
    if stale.exists():
        stale.unlink()
        print("[monitor] Cleared stale selected_frames.txt")

    # Shared state between HTTP handler and file watcher
    shared: dict = {"image_numbers": set()}

    # Start combined HTTP server: static files from outdir + POST /fit_selected
    handler_cls = make_http_handler(args, shared)
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(("0.0.0.0", args.port), handler_cls)
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True, name="http-server")
    http_thread.start()
    print(f"[monitor] Dashboard → http://localhost:{args.port}/focus_time_series.html")

    existing_numbers = collect_existing_numbers(args)
    if existing_numbers:
        print(f"[monitor] Found {len(existing_numbers)} existing science files")
    else:
        print("[monitor] No science files found yet; waiting for first exposure")

    previous_output_numbers = (
        set() if args.no_resume_output else collect_previous_output_numbers(args.outdir)
    )
    if previous_output_numbers:
        print(
            f"[monitor] Resuming dashboard with {len(previous_output_numbers)} "
            "previous output frame(s)"
        )

    initial_pipeline_numbers = (
        set(existing_numbers)
        if args.initial_run
        else set(previous_output_numbers)
    )
    handler = ScienceWatcher(args, existing_numbers, initial_pipeline_numbers, shared)
    shared["image_numbers"] = handler.image_numbers  # live reference, updated by watcher
    observer = Observer()
    observer.schedule(handler, str(data_dir), recursive=False)
    observer.start()

    try:
        if existing_numbers and args.initial_run and not args.no_initial_run:
            handler._schedule_run()
        elif existing_numbers:
            print(
                "[monitor] Existing files will be ignored for processing; "
                "new files will be processed as they arrive. Previous dashboard "
                "frames are kept if found in --outdir. Use --initial-run to "
                "reduce the existing backlog."
            )
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[monitor] Shutting down")
    finally:
        terminate_active_process(shared)
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
