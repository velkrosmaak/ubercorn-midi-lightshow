#!/usr/bin/env python3
"""
webapp.py – Ubercorn MIDI Light Show web control interface
===========================================================
A Flask app that serves the control UI and exposes a REST API.
Reads/writes ubercorn_config.json which the main script hot-reloads.
Optionally manages the main script process via subprocess.

Install:
    pip3 install flask

Run:
    python3 webapp.py [--port 8080]

Then visit http://<pi-ip>:8080 in a browser on the same network.
"""

import argparse
import os
import signal
import subprocess
import sys
import time

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg_module

app = Flask(__name__, static_folder="static")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app = Flask(__name__, static_folder=STATIC_DIR)

# ── process management ─────────────────────────────────────────────────────────

SHOW_SCRIPT = os.path.join(os.path.dirname(__file__), "ubercorn_midi_show.py")
_show_proc = None

def show_is_running() -> bool:
    global _show_proc
    if _show_proc is None:
        return False
    poll = _show_proc.poll()
    return poll is None

def start_show():
    global _show_proc
    if show_is_running():
        return {"ok": False, "msg": "Already running"}
    _show_proc = subprocess.Popen(
        [sys.executable, SHOW_SCRIPT],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True
    )
    return {"ok": True, "pid": _show_proc.pid}

def stop_show():
    global _show_proc
    if not show_is_running():
        return {"ok": False, "msg": "Not running"}
    _show_proc.terminate()
    try:
        _show_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _show_proc.kill()
    _show_proc = None
    return {"ok": True}

def restart_show():
    """Write restart flag; the main script re-execs itself."""
    cfg_module.update({"restart": True})
    return {"ok": True, "msg": "Restart flag set"}

# ── REST API ───────────────────────────────────────────────────────────────────

@app.route("/api/config", methods=["GET"])
def api_get_config():
    return jsonify(cfg_module.load())

@app.route("/api/config", methods=["POST"])
def api_set_config():
    data = request.get_json(force=True)
    # whitelist of settable keys
    allowed = {
        "brightness", "fps", "rotation", "speed", "hue_shift", "saturation",
        "blur", "particle_den", "strobe_rate", "sustain", "attract_mode",
        "attract_interval", "max_layers", "midi_host", "midi_port", "blackout",
    }
    filtered = {k: v for k, v in data.items() if k in allowed}
    updated = cfg_module.update(filtered)
    return jsonify({"ok": True, "config": updated})

@app.route("/api/config/<key>", methods=["POST"])
def api_set_one(key):
    data = request.get_json(force=True)
    cfg_module.update({key: data.get("value")})
    return jsonify({"ok": True})

@app.route("/api/process/start", methods=["POST"])
def api_start():
    return jsonify(start_show())

@app.route("/api/process/stop", methods=["POST"])
def api_stop():
    return jsonify(stop_show())

@app.route("/api/process/restart", methods=["POST"])
def api_restart():
    return jsonify(restart_show())

@app.route("/api/process/status", methods=["GET"])
def api_status():
    return jsonify({
        "running": show_is_running(),
        "pid": _show_proc.pid if show_is_running() else None,
    })

@app.route("/api/blackout", methods=["POST"])
def api_blackout():
    data = request.get_json(force=True)
    cfg_module.update({"blackout": bool(data.get("value", True))})
    return jsonify({"ok": True})

@app.route("/api/reset", methods=["POST"])
def api_reset():
    defaults = cfg_module.DEFAULTS.copy()
    defaults.pop("restart", None)
    defaults.pop("blackout", None)
    cfg_module.save(defaults)
    return jsonify({"ok": True, "config": defaults})

# ── serve the SPA ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return app.send_static_file("index.html")

# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--autostart", action="store_true",
                        help="Auto-start the light show on launch")
    args = parser.parse_args()

    # ensure config file exists
    cfg_module.load()

    if args.autostart:
        start_show()
        print(f"[WEB] Light show started (PID {_show_proc.pid if _show_proc else '?'})")

    print(f"[WEB] Control UI at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
