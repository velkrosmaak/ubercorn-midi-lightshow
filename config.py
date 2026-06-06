"""
config.py – shared config contract between ubercorn_midi_show.py and webapp.py
==============================================================================
Both processes read/write CONFIG_PATH (JSON).  The main script polls it every
second and hot-reloads any changed values without restarting.  A "restart"
flag tells the main script to exec() itself fresh.
"""

import json
import os
import time

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ubercorn_config.json")

DEFAULTS = {
    # ── display ──────────────────────────────────────────────────────────────
    "brightness":     0.8,    # 0.0 – 1.0
    "fps":            60,     # 10 – 60
    "rotation":       0,      # 0 / 90 / 180 / 270

    # ── animation ────────────────────────────────────────────────────────────
    "speed":          1.0,    # 0.2 – 4.0  (animation time multiplier)
    "hue_shift":      0.0,    # 0 – 360 degrees
    "saturation":     1.0,    # 0.0 – 1.0
    "blur":           0.0,    # 0.0 – 1.0
    "particle_den":   64,     # 0 – 127

    # ── effects ───────────────────────────────────────────────────────────────
    "strobe_rate":    0.0,    # Hz, 0 = off, max 30
    "sustain":        False,  # freeze decay / bloom mode
    "attract_mode":   True,   # auto-fire when MIDI silent > 10 s
    "attract_interval": 4.0,  # seconds between attract spawns
    "max_layers":     12,     # 1 – 24

    # ── midi / network ────────────────────────────────────────────────────────
    "midi_host":      "0.0.0.0",
    "midi_port":      5004,

    # ── meta / control ────────────────────────────────────────────────────────
    "blackout":       False,  # all pixels off
    "restart":        False,  # main script will exec() itself when True
    "_updated_at":    0.0,    # epoch timestamp, set on every write
}


def load() -> dict:
    """Load config from disk, falling back to defaults for missing keys."""
    if not os.path.exists(CONFIG_PATH):
        save(DEFAULTS.copy())
        return DEFAULTS.copy()
    try:
        with open(CONFIG_PATH) as f:
            data = json.load(f)
        # fill in any missing keys from defaults
        for k, v in DEFAULTS.items():
            if k not in data:
                data[k] = v
        return data
    except Exception:
        return DEFAULTS.copy()


def save(cfg: dict):
    """Atomically write config to disk."""
    cfg["_updated_at"] = time.time()
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG_PATH)


def update(partial: dict):
    """Merge partial dict into existing config and save."""
    cfg = load()
    cfg.update(partial)
    save(cfg)
    return cfg
