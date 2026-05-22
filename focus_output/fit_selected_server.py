#!/usr/bin/env python3
"""Flask backend: receive selected frame numbers from the dashboard and run the focus fit."""
from flask import Flask, request, jsonify
import json
import os
import subprocess
import sys

app = Flask(__name__)

OUTDIR = os.path.dirname(os.path.abspath(__file__))
ARGS_FILE = os.path.join(OUTDIR, "monitor_args.json")


def _add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

app.after_request(_add_cors)


@app.route("/fit_selected", methods=["OPTIONS"])
def fit_selected_options():
    return "", 204


def _build_pipeline_cmd(args_dict, sci_numbers):
    python_bin = args_dict.get("python_bin", sys.executable)
    pipeline   = args_dict.get("pipeline_script", "focus_pipeline.py")
    cmd = [python_bin, pipeline]
    cmd += ["--data-dir",   args_dict["data_dir"]]
    cmd += ["--filter",     args_dict["filter"]]
    cmd += ["--bias-nums"]  + args_dict["bias_nums"]
    cmd += ["--dark-nums"]  + args_dict["dark_nums"]
    if args_dict.get("flat_nums"):
        cmd += ["--flat-nums"] + args_dict["flat_nums"]
    cmd += ["--sci-nums"] + [str(n) for n in sci_numbers]
    cmd += ["--mask-dir",   args_dict["mask_dir"]]
    cmd += ["--outdir",     args_dict["outdir"]]
    cmd += ["--focus-key",  args_dict.get("focus_key", "LVDTC")]
    cmd += ["--time-key",   args_dict.get("time_key", "TIME-OBS")]
    cmd += ["--date-key",   args_dict.get("date_key", "DATE-OBS")]
    cmd += ["--airmass-key",args_dict.get("airmass_key", "AIRMASS")]
    cmd += ["--pixscale",   str(args_dict.get("pixscale", 0.455))]
    cmd += ["--threshold",  str(args_dict.get("threshold", 25.0))]
    if args_dict.get("amps"):
        cmd += ["--amps"] + [str(a) for a in args_dict["amps"]]
    cmd += ["--mask-sat-mult",   str(args_dict.get("mask_sat_mult", 1.0))]
    cmd += ["--mask-black-mult", str(args_dict.get("mask_black_mult", 4.0))]
    cmd += ["--mask-sat-frac",   str(args_dict.get("mask_sat_frac", 0.25))]
    cmd += ["--mask-black-frac", str(args_dict.get("mask_black_frac", 0.2))]
    if args_dict.get("auto_generate_masks"):
        cmd.append("--auto-generate-masks")
    if args_dict.get("write_reduced"):
        cmd.append("--write-reduced")
    elif args_dict.get("skip_reduced"):
        cmd.append("--skip-reduced")
    if args_dict.get("incremental"):
        cmd.append("--incremental")
    # NOTE: do NOT pass --no-fit here — this run IS the fit
    return cmd


@app.route("/fit_selected", methods=["POST"])
def fit_selected():
    data = request.get_json(force=True)
    frames = data.get("frames", [])
    if len(frames) < 3:
        return jsonify({"error": "Need at least 3 frames for a valid parabola fit"}), 400

    # Save selected frames list
    with open(os.path.join(OUTDIR, "selected_frames.txt"), "w") as f:
        f.write(" ".join(str(x) for x in frames))

    # Read monitor args
    if not os.path.exists(ARGS_FILE):
        return jsonify({"error": "monitor_args.json not found — is the monitor running?"}), 500
    with open(ARGS_FILE) as f:
        args_dict = json.load(f)

    cmd = _build_pipeline_cmd(args_dict, frames)
    print(f"[fit-server] Running pipeline for {len(frames)} selected frames: {frames}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "")[-800:]
            print(f"[fit-server] Pipeline failed:\n{err}")
            return jsonify({"error": err}), 500
        print("[fit-server] Fit complete")
        return jsonify({"status": "ok", "frames": frames}), 200
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Pipeline timed out (600 s)"}), 500


if __name__ == "__main__":
    print(f"[fit-server] Serving on http://0.0.0.0:8000")
    print(f"[fit-server] Reading pipeline args from: {ARGS_FILE}")
    app.run(host="0.0.0.0", port=8000)
