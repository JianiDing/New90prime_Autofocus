#!/usr/bin/env python3
"""Simple HTTP server that reliably serves files to remote clients."""
import http.server
import json
import math
import os
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIRECTORY)

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Suppress noisy error tracebacks from dropped connections."""
    def log_message(self, format, *args):
        # Only log successful requests
        if len(args) >= 2 and isinstance(args[1], str) and args[1].startswith("2"):
            super().log_message(format, *args)
        elif len(args) >= 1 and "200" in str(args):
            super().log_message(format, *args)

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass  # client disconnected, ignore

    def do_POST(self):
        if self.path != "/amp_fwhm":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            frame = int(body.get("frame"))
            payload = self._amp_fwhm_for_frame(frame)
            self._json(200, payload)
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def _json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _amp_fwhm_for_frame(self, frame):
        points_path = os.path.join(DIRECTORY, "focus_per_amp_points.ecsv")
        if not os.path.exists(points_path):
            raise FileNotFoundError("focus_per_amp_points.ecsv not found")

        pixscale = 0.455
        rows = []
        with open(points_path) as fh:
            header = None
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if header is None:
                    header = line.split()
                    continue
                values = line.split()
                if len(values) != len(header):
                    continue
                row = dict(zip(header, values))
                if int(row["image_number"]) != frame:
                    continue
                fwhm_pix = float(row["median_fwhm"])
                fwhm_arcsec = fwhm_pix * pixscale
                rows.append({
                    "amp": int(row["amp"]),
                    "fwhm_pix": round(fwhm_pix, 4) if math.isfinite(fwhm_pix) else None,
                    "fwhm_arcsec": round(fwhm_arcsec, 4) if math.isfinite(fwhm_arcsec) else None,
                    "median_e": round(float(row["median_e"]), 4),
                    "n_stars": int(row["n_stars"]),
                    "sci_file": row["sci_file"],
                    "focus_position": float(row["focus_position"]),
                })
        if not rows:
            raise ValueError(f"No per-amplifier FWHM data found for frame {frame}")

        valid = [r["fwhm_pix"] for r in rows if r["fwhm_pix"] is not None]
        valid_sorted = sorted(valid)
        mid = len(valid_sorted) // 2
        if len(valid_sorted) % 2:
            avg_pix = valid_sorted[mid]
        else:
            avg_pix = 0.5 * (valid_sorted[mid - 1] + valid_sorted[mid])
        avg_arcsec = avg_pix * pixscale
        return {
            "frame": frame,
            "sci_file": rows[0]["sci_file"],
            "focus_position": round(rows[0]["focus_position"], 4),
            "pixscale": pixscale,
            "avg_fwhm_pix": round(avg_pix, 4),
            "avg_fwhm_arcsec": round(avg_arcsec, 4),
            "amps": [
                {k: v for k, v in row.items() if k not in ("sci_file", "focus_position")}
                for row in rows
            ],
        }

# Key fix: allow port reuse and use a forking/threading server
socketserver.TCPServer.allow_reuse_address = True
with socketserver.ThreadingTCPServer(("0.0.0.0", PORT), QuietHandler) as httpd:
    print(f"Serving on http://0.0.0.0:{PORT}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
