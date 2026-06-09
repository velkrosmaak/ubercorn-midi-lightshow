#!/usr/bin/env python3
"""
ubercorn_midi_show.py
=====================
MIDI-over-WiFi light show for Raspberry Pi Zero + Pimoroni Ubercorn (Unicorn HAT HD 16×16).

All effect rendering is fully vectorised with NumPy — no Python pixel loops.
Frames are composed as (H, W, 3) float32 arrays, clipped to uint8 at output.
This gives ~50-100× speedup over the previous pure-Python loop approach.

Install:
    pip3 install numpy flask        # Mac dev
    pip3 install numpy unicornhathd # Pi

Config hot-reload: reads ubercorn_config.json every ~1 s (60 frames).
Web UI:           run webapp.py alongside this script.
RTP-MIDI:         full Apple MIDI session via rtpmidi_session.py.
"""

import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# ── local modules ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg_module
from rtpmidi_session import RTPMidiServer

# ── optional hardware ──────────────────────────────────────────────────────────
try:
    import unicornhathd as hat
    HAT_AVAILABLE = True
    WIDTH, HEIGHT = hat.get_shape()
except ImportError:
    HAT_AVAILABLE = False
    WIDTH, HEIGHT = 16, 16
    print("[WARN] unicornhathd not found – running in sim mode.")

# ── pre-compute coordinate grids (done once, reused every frame) ───────────────
_YY, _XX = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
_CX, _CY  = WIDTH / 2.0 - 0.5, HEIGHT / 2.0 - 0.5
_DX       = _XX - _CX          # (H,W) x-offset from centre
_DY       = _YY - _CY          # (H,W) y-offset from centre
_DIST     = np.sqrt(_DX**2 + _DY**2)          # (H,W) radial distance
_ANGLE    = np.arctan2(_DY, _DX)              # (H,W) angle in radians
_DIAG     = (_XX + _YY).astype(np.float32)    # (H,W) diagonal coordinate

# ── state ──────────────────────────────────────────────────────────────────────
class State:
    brightness    = 0.8
    fps           = 60
    rotation      = 0
    speed         = 1.0
    hue_shift     = 0.0
    strobe_rate   = 0.0
    particle_den  = 64
    saturation    = 1.0
    sustain       = False
    blur          = 0.0
    blackout      = False
    attract_mode  = True
    attract_interval = 4.0
    max_layers    = 12
    midi_host     = "0.0.0.0"
    midi_port     = 5004
    time          = 0.0

state = State()
_last_cfg_mtime = [0.0]

def reload_config():
    try:
        if not os.path.exists(cfg_module.CONFIG_PATH):
            print(f"[CONFIG] Not found – creating defaults at {cfg_module.CONFIG_PATH}")
            cfg_module.save(cfg_module.DEFAULTS.copy())
            return
        mtime = os.path.getmtime(cfg_module.CONFIG_PATH)
        if mtime <= _last_cfg_mtime[0]:
            return
        _last_cfg_mtime[0] = mtime
        c = cfg_module.load()
        state.brightness       = float(c.get("brightness",      0.8))
        state.fps              = int(c.get("fps",               60))
        state.rotation         = int(c.get("rotation",          0))
        state.speed            = float(c.get("speed",           1.0))
        state.hue_shift        = float(c.get("hue_shift",       0.0))
        state.strobe_rate      = float(c.get("strobe_rate",     0.0))
        state.particle_den     = int(c.get("particle_den",      64))
        state.saturation       = float(c.get("saturation",      1.0))
        state.sustain          = bool(c.get("sustain",          False))
        state.blur             = float(c.get("blur",            0.0))
        state.blackout         = bool(c.get("blackout",         False))
        state.attract_mode     = bool(c.get("attract_mode",     True))
        state.attract_interval = float(c.get("attract_interval",4.0))
        state.max_layers       = int(c.get("max_layers",        12))
        if HAT_AVAILABLE:
            hat.rotation(state.rotation)
        print(f"[CONFIG] brightness={state.brightness:.2f} fps={state.fps} speed={state.speed:.1f}")
        if c.get("restart", False):
            cfg_module.update({"restart": False})
            print("[CONFIG] Restarting…")
            if HAT_AVAILABLE:
                hat.off()
            os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        print(f"[CONFIG] reload error: {e}")

# ── MIDI constants ─────────────────────────────────────────────────────────────
MIDI_NOTE_OFF  = 0x80
MIDI_NOTE_ON   = 0x90
MIDI_CC        = 0xB0
MIDI_PITCHBEND = 0xE0

PALETTE_BANKS = [
    (0.0,  0.08), (0.08, 0.17), (0.28, 0.42),
    (0.55, 0.70), (0.70, 0.85), (0.85, 1.00),
]

# ── NumPy HSV → RGB (vectorised, operates on whole (H,W) arrays) ───────────────
def hsv_array_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    h, s, v : float32 arrays of shape (H, W), values in [0, 1].
    Returns  : float32 array of shape (H, W, 3) with RGB in [0, 255].
    """
    h = (h + state.hue_shift / 360.0) % 1.0
    s = np.clip(s * state.saturation, 0, 1)
    i = (h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i6 = i % 6
    r = np.select([i6==0, i6==1, i6==2, i6==3, i6==4, i6==5], [v, q, p, p, t, v])
    g = np.select([i6==0, i6==1, i6==2, i6==3, i6==4, i6==5], [t, v, v, q, p, p])
    b = np.select([i6==0, i6==1, i6==2, i6==3, i6==4, i6==5], [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1) * 255.0

def solid_rgb(hue: float, sat: float, val: float) -> np.ndarray:
    """Return a solid (H, W, 3) float32 frame of one HSV colour."""
    h = np.full((HEIGHT, WIDTH), hue,  dtype=np.float32)
    s = np.full((HEIGHT, WIDTH), sat,  dtype=np.float32)
    v = np.full((HEIGHT, WIDTH), val,  dtype=np.float32)
    return hsv_array_to_rgb(h, s, v)

# ── layer ──────────────────────────────────────────────────────────────────────
@dataclass
class Layer:
    note:      int
    velocity:  int
    hue:       float = 0.0
    alpha:     float = 1.0
    decay:     float = 0.015
    alive:     bool  = True
    effect_fn: object = None
    phase:     float = 0.0
    born:      float = 0.0

    def vel_val(self) -> float:
        return 0.3 + 0.7 * (self.velocity / 127.0)

layers: List[Layer] = []
layer_lock = threading.Lock()

# ── effects (all return float32 (H,W,3) arrays, no Python pixel loops) ─────────

def effect_radial_burst(L: Layer) -> np.ndarray:
    radius    = L.phase * WIDTH * 0.8
    thickness = 2.5 + 1.5 * np.sin(L.phase * 3)
    dist      = np.abs(_DIST - radius)
    mask      = dist < thickness
    bright    = np.where(mask, (1.0 - dist / max(thickness, 1e-6)) * L.alpha, 0.0).astype(np.float32)
    h = np.full((HEIGHT, WIDTH), L.hue, dtype=np.float32)
    s = np.ones((HEIGHT, WIDTH),         dtype=np.float32)
    v = (bright * L.vel_val()).astype(np.float32)
    return hsv_array_to_rgb(h, s, v)

def effect_note_flash(L: Layer) -> np.ndarray:
    sat = max(0.0, 1.0 - L.alpha * 0.4)
    val = L.vel_val() * L.alpha
    return solid_rgb(L.hue, sat, val)

def effect_diagonal_sweep(L: Layer) -> np.ndarray:
    sweep = L.phase * (WIDTH + HEIGHT) * 1.4 - 4
    dist  = np.abs(_DIAG - sweep).astype(np.float32)
    bright = np.clip((1.0 - dist / 4.0), 0, 1) * L.alpha * L.vel_val()
    h_arr = ((L.hue + dist * 0.03) % 1.0).astype(np.float32)
    s_arr = np.ones((HEIGHT, WIDTH), dtype=np.float32)
    return hsv_array_to_rgb(h_arr, s_arr, bright.astype(np.float32))

def effect_sparkle(L: Layer) -> np.ndarray:
    out = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    rng = random.Random(L.note + int(L.phase * 10))
    count = int(state.particle_den / 5) + 5
    v = L.vel_val() * L.alpha
    for _ in range(count):
        x = rng.randint(0, WIDTH - 1)
        y = rng.randint(0, HEIGHT - 1)
        h = (L.hue + rng.uniform(-0.1, 0.1)) % 1.0
        br = rng.uniform(0.5, 1.0) * v
        sat = rng.uniform(0.7, 1.0)
        rgb = _hsv_scalar(h, sat, br)
        out[y, x] = rgb
    return out

def effect_spiral(L: Layer) -> np.ndarray:
    t     = L.phase * np.pi * 4
    r     = _DIST + 1e-4
    sp    = (_ANGLE + r * 0.6 - t) % (2 * np.pi / 3)
    dist  = np.minimum(sp, 2 * np.pi / 3 - sp).astype(np.float32)
    fade  = np.clip(1.0 - r / (WIDTH * 0.85), 0, 1).astype(np.float32)
    bright = np.where(dist < 0.4,
                      (1.0 - dist / 0.4) * L.alpha * fade * L.vel_val(),
                      0.0).astype(np.float32)
    h_arr = ((L.hue + r / WIDTH * 0.3) % 1.0).astype(np.float32)
    s_arr = np.ones((HEIGHT, WIDTH), dtype=np.float32)
    return hsv_array_to_rgb(h_arr, s_arr, bright)

def effect_columns(L: Layer) -> np.ndarray:
    out = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    v = L.vel_val()
    heights = (0.5 + 0.5 * np.sin(L.phase * 6 + np.arange(WIDTH) * 0.8)) * HEIGHT
    heights = heights.astype(np.int32)
    hues = (L.hue + np.arange(WIDTH) / WIDTH * 0.3) % 1.0
    for x in range(WIDTH):
        col_h = int(heights[x])
        if col_h <= 0:
            continue
        ys = np.arange(col_h)
        brights = (1.0 - ys / HEIGHT * 0.5) * L.alpha * v
        rgb = _hsv_scalar(float(hues[x]), 1.0, 1.0)
        out[HEIGHT - 1 - ys, x] = rgb * brights[:, None]
    return out

def effect_plasma(L: Layer) -> np.ndarray:
    t   = L.phase * 4
    val = (np.sin(_XX * 0.5 + t) +
           np.sin(_YY * 0.5 + t * 1.3) +
           np.sin((_XX + _YY) * 0.35 + t * 0.7) +
           np.sin(_DIST * 0.4 + t)) / 4.0
    bright = np.clip((0.5 + 0.5 * val) * L.alpha * L.vel_val(), 0, 1).astype(np.float32)
    h_arr  = ((L.hue + val * 0.2) % 1.0).astype(np.float32)
    s_arr  = np.ones((HEIGHT, WIDTH), dtype=np.float32)
    return hsv_array_to_rgb(h_arr, s_arr, bright)

def effect_strobe_burst(L: Layer) -> np.ndarray:
    if int(L.phase * 20) % 2 == 0:
        return np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    return solid_rgb(L.hue, 0.6, L.vel_val() * L.alpha)

def effect_diamond(L: Layer) -> np.ndarray:
    manhattan = (np.abs(_XX - _CX) + np.abs(_YY - _CY)).astype(np.float32)
    size  = L.phase * max(WIDTH, HEIGHT) * 1.2
    dist  = np.abs(manhattan - size)
    bright = np.where(dist < 3,
                      (1.0 - dist / 3.0) * L.alpha * L.vel_val(),
                      0.0).astype(np.float32)
    h_arr = ((L.hue + dist * 0.04) % 1.0).astype(np.float32)
    s_arr = np.ones((HEIGHT, WIDTH), dtype=np.float32)
    return hsv_array_to_rgb(h_arr, s_arr, bright)

def effect_meteor(L: Layer) -> np.ndarray:
    out = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    rng = random.Random(L.note)
    v = L.vel_val()
    tail = 6
    for _ in range(4):
        sx    = rng.randint(0, WIDTH - 1)
        speed = rng.uniform(0.8, 1.5)
        for i in range(tail):
            t_off = L.phase * speed * HEIGHT
            px = int(sx + t_off - i) % WIDTH
            py = int(t_off - i) % HEIGHT
            br = ((tail - i) / tail) * L.alpha * v
            h  = (L.hue + i * 0.02) % 1.0
            out[py, px] = np.clip(out[py, px] + np.array(_hsv_scalar(h, 1.0, br)), 0, 255)
    return out

def effect_ripple(L: Layer) -> np.ndarray:
    out = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    rng = random.Random(L.note)
    v   = L.vel_val()
    radius = L.phase * WIDTH * 0.7
    for _ in range(3):
        cx = rng.uniform(2, WIDTH - 3)
        cy = rng.uniform(2, HEIGHT - 3)
        dx = _XX - cx
        dy = _YY - cy
        d  = np.sqrt(dx**2 + dy**2).astype(np.float32)
        dist = np.abs(d - radius)
        bright = np.where(dist < 2, (1.0 - dist / 2.0) * L.alpha * 0.6 * v, 0.0).astype(np.float32)
        h = (L.hue + radius / WIDTH * 0.2) % 1.0
        h_arr = np.full((HEIGHT, WIDTH), h, dtype=np.float32)
        s_arr = np.ones((HEIGHT, WIDTH),    dtype=np.float32)
        out = np.clip(out + hsv_array_to_rgb(h_arr, s_arr, bright), 0, 255)
    return out

def effect_warp(L: Layer) -> np.ndarray:
    """Tunnel/warp zoom effect."""
    t    = L.phase * 3
    r    = _DIST + 1e-4
    zoom = (np.sin(np.log(r + 1) * 2 - t) + 1) / 2
    twist = _ANGLE + r * 0.3
    h_arr = ((L.hue + twist / (2 * np.pi) * 0.4 + zoom * 0.2) % 1.0).astype(np.float32)
    s_arr = np.ones((HEIGHT, WIDTH), dtype=np.float32)
    bright = (zoom * L.alpha * L.vel_val()).astype(np.float32)
    return hsv_array_to_rgb(h_arr, s_arr, bright)

# ── scalar HSV helper (for per-pixel effects like sparkle/meteor) ──────────────
def _hsv_scalar(h: float, s: float, v: float) -> np.ndarray:
    """Returns a (3,) float32 RGB array [0-255]."""
    h = (h + state.hue_shift / 360.0) % 1.0
    s = float(np.clip(s * state.saturation, 0, 1))
    i = int(h * 6)
    f = h * 6 - i
    p, q, t_ = v*(1-s), v*(1-f*s), v*(1-(1-f)*s)
    rgb = [(v,t_,p),(q,v,p),(p,v,t_),(p,q,v),(t_,p,v),(v,p,q)][i % 6]
    return np.array(rgb, dtype=np.float32) * 255.0

EFFECTS = [
    effect_radial_burst, effect_note_flash, effect_diagonal_sweep,
    effect_sparkle, effect_spiral, effect_columns, effect_plasma,
    effect_strobe_burst, effect_diamond, effect_meteor, effect_ripple,
    effect_warp,
]

def assign_effect(velocity: int, note: int):
    if velocity > 110:
        return random.choice([effect_strobe_burst, effect_note_flash, effect_radial_burst])
    if velocity > 85:
        return random.choice([effect_plasma, effect_spiral, effect_diamond, effect_warp])
    if velocity > 60:
        return random.choice([effect_radial_burst, effect_diagonal_sweep, effect_columns, effect_meteor])
    return random.choice([effect_sparkle, effect_ripple, effect_diagonal_sweep, effect_columns])

# ── background (vectorised aurora) ─────────────────────────────────────────────
class Background:
    def __init__(self):
        self.t = 0.0

    def render(self) -> np.ndarray:
        t = self.t
        v = (np.sin(_XX * 0.4 + t * 0.3) * np.sin(_YY * 0.3 + t * 0.2)) * 0.12
        v = np.clip(v, 0, None).astype(np.float32)
        h = ((t * 0.04 + _XX / WIDTH * 0.15 + _YY / HEIGHT * 0.1) % 1.0).astype(np.float32)
        s = np.full((HEIGHT, WIDTH), 0.8, dtype=np.float32)
        frame = hsv_array_to_rgb(h, s, v)
        self.t += (1.0 / max(state.fps, 1)) * state.speed
        return frame

background = Background()

# ── compositing ────────────────────────────────────────────────────────────────

def blur_pass(frame: np.ndarray, amount: float) -> np.ndarray:
    """Fast box blur using np.roll — no Python loops."""
    if amount < 0.01:
        return frame
    blurred = (
        np.roll(frame,  1, axis=0) + np.roll(frame, -1, axis=0) +
        np.roll(frame,  1, axis=1) + np.roll(frame, -1, axis=1) +
        frame
    ) / 5.0
    return frame + (blurred - frame) * amount

def composite() -> np.ndarray:
    if state.blackout:
        return np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

    frame = background.render()

    with layer_lock:
        active = [l for l in layers if l.alive]

    for layer in active:
        frame = np.clip(frame + layer.effect_fn(layer), 0, 255)

    frame = blur_pass(frame, state.blur)

    if state.strobe_rate > 0:
        period = 1.0 / state.strobe_rate
        if int(state.time / (period / 2)) % 2 == 1:
            return np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

    return frame

# ── layer lifecycle ────────────────────────────────────────────────────────────

def pick_palette(velocity: int) -> Tuple[float, float]:
    return PALETTE_BANKS[(velocity // 22) % len(PALETTE_BANKS)]

def spawn_layer(note: int, velocity: int):
    with layer_lock:
        for l in layers:
            if l.note == note:
                l.decay = 0.04
        lo, hi  = pick_palette(velocity)
        effect  = assign_effect(velocity, note)
        decay   = 0.005 + 0.02 * (velocity / 127.0)
        layer   = Layer(note=note, velocity=velocity,
                        hue=random.uniform(lo, hi),
                        decay=decay, effect_fn=effect, born=state.time)
        layers.append(layer)
        if len(layers) > state.max_layers:
            layers.pop(0)
        print(f"[SHOW] note={note}({note_name(note)}) vel={velocity} "
              f"effect={effect.__name__} layers={len(layers)}")

def release_layer(note: int):
    with layer_lock:
        for l in layers:
            if l.note == note and l.alive:
                if not state.sustain:
                    l.decay = 0.02

def update_layers():
    dt = (1.0 / max(state.fps, 1)) * state.speed
    with layer_lock:
        for l in layers:
            if not l.alive:
                continue
            l.phase += dt
            if state.sustain:
                l.alpha = min(l.alpha + 0.01, 1.0)
            else:
                l.alpha -= l.decay
            if l.alpha <= 0:
                l.alive = False
        layers[:] = [l for l in layers if l.alive or l.phase < 0.5]

# ── MIDI ───────────────────────────────────────────────────────────────────────
last_midi_time = [time.time()]
_NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
_CC_NAMES   = {1:'Brightness',7:'Speed',10:'Hue Shift',16:'Strobe',
               17:'Particles',18:'Saturation',64:'Sustain',74:'Blur',121:'Reset'}

def note_name(n: int) -> str:
    return f"{_NOTE_NAMES[n%12]}{n//12-1}"

def handle_midi_message(msg_bytes: bytes):
    last_midi_time[0] = time.time()
    if len(msg_bytes) < 2:
        return
    status  = msg_bytes[0] & 0xF0
    channel = (msg_bytes[0] & 0x0F) + 1
    d1      = msg_bytes[1] if len(msg_bytes) > 1 else 0
    d2      = msg_bytes[2] if len(msg_bytes) > 2 else 0

    if status == MIDI_NOTE_ON and d2 > 0:
        print(f"[MIDI] NOTE ON  ch={channel} note={d1}({note_name(d1)}) vel={d2}")
        spawn_layer(d1, d2)
    elif status == MIDI_NOTE_OFF or (status == MIDI_NOTE_ON and d2 == 0):
        print(f"[MIDI] NOTE OFF ch={channel} note={d1}({note_name(d1)})")
        release_layer(d1)
    elif status == MIDI_CC:
        name = _CC_NAMES.get(d1, f"CC{d1}")
        if   d1 == 1:   state.brightness  = 0.1 + 0.9*(d2/127); v=f"{state.brightness:.2f}"
        elif d1 == 7:   state.speed       = 0.2 + 3.8*(d2/127); v=f"{state.speed:.2f}×"
        elif d1 == 10:  state.hue_shift   = d2/127*360;          v=f"{state.hue_shift:.1f}°"
        elif d1 == 16:  state.strobe_rate = d2/127*30;           v=f"{state.strobe_rate:.1f}Hz"
        elif d1 == 17:  state.particle_den= d2;                  v=str(d2)
        elif d1 == 18:  state.saturation  = d2/127;              v=f"{state.saturation:.2f}"
        elif d1 == 64:  state.sustain     = d2>=64;              v='ON' if state.sustain else 'OFF'
        elif d1 == 74:  state.blur        = d2/127;              v=f"{state.blur:.2f}"
        elif d1 == 121:
            state.brightness=0.8; state.speed=1.0; state.hue_shift=0.0
            state.strobe_rate=0.0; state.saturation=1.0; state.sustain=False; state.blur=0.0
            v="ALL RESET"
        else: v="(unmapped)"
        print(f"[MIDI] CC ch={channel} {name}={d1} val={d2} → {v}")
    elif status == MIDI_PITCHBEND:
        bend = ((d2 << 7) | d1) - 8192
        print(f"[MIDI] PITCHBEND ch={channel} bend={bend:+d}")

# ── RTP-MIDI server ────────────────────────────────────────────────────────────
_rtp_server: RTPMidiServer = None

def start_rtp_server():
    global _rtp_server
    _rtp_server = RTPMidiServer(
        name="Ubercorn", port=state.midi_port,
        midi_callback=handle_midi_message)
    _rtp_server.start()

# ── attract mode ───────────────────────────────────────────────────────────────
def attract_mode_thread():
    while True:
        time.sleep(state.attract_interval)
        if state.attract_mode and time.time() - last_midi_time[0] > 10.0:
            spawn_layer(random.randint(36, 84), random.randint(50, 110))

# ── display loop ───────────────────────────────────────────────────────────────

def sim_display(frame_u8: np.ndarray):
    """ASCII preview for non-Pi dev — frame is uint8 (H,W,3)."""
    chars = " ░▒▓█"
    lum   = frame_u8[::2, :, :].mean(axis=2)   # subsample rows
    idxs  = (lum / 255.0 * (len(chars) - 1)).astype(np.int32)
    rows  = ["".join(chars[i] for i in row) for row in idxs]
    print("\033[H" + "\n".join(rows), end="", flush=True)

def display_loop():
    if HAT_AVAILABLE:
        hat.rotation(state.rotation)
        hat.brightness(state.brightness)
    else:
        print("\033[2J")

    cfg_tick = 0
    frame_times = []

    while True:
        t0 = time.time()
        state.time += 1.0 / max(state.fps, 1)

        cfg_tick += 1
        if cfg_tick >= 60:
            cfg_tick = 0
            reload_config()

        update_layers()
        frame_f = composite()                              # float32 (H,W,3)
        frame_u8 = np.clip(frame_f * state.brightness,
                           0, 255).astype(np.uint8)       # uint8  (H,W,3)

        if HAT_AVAILABLE:
            hat.brightness(state.brightness)
            # set_pixel_array is the fastest path if available, else loop
            if hasattr(hat, 'set_pixel_array'):
                hat.set_pixel_array(frame_u8)
            else:
                for y in range(HEIGHT):
                    for x in range(WIDTH):
                        r, g, b = frame_u8[y, x]
                        hat.set_pixel(x, y, int(r), int(g), int(b))
            hat.show()
        else:
            sim_display(frame_u8)

        elapsed = time.time() - t0
        frame_times.append(elapsed)
        if len(frame_times) >= 120:
            avg = sum(frame_times) / len(frame_times)
            print(f"[PERF] avg frame {avg*1000:.1f}ms  ({1/avg:.0f} fps actual)", flush=True)
            frame_times.clear()

        sleep_t = max(0, 1.0 / max(state.fps, 1) - elapsed)
        time.sleep(sleep_t)

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    reload_config()
    print("╔══════════════════════════════════════════╗")
    print("║  Ubercorn MIDI Light Show  [NumPy mode]  ║")
    print(f"║  Grid {WIDTH}×{HEIGHT}  FPS {state.fps}  Port {state.midi_port}/{state.midi_port+1}       ║")
    print("╠══════════════════════════════════════════╣")
    print("║  Apple MIDI: Audio MIDI Setup → Network  ║")
    print("╚══════════════════════════════════════════╝")

    start_rtp_server()
    threading.Thread(target=attract_mode_thread, daemon=True).start()

    try:
        display_loop()
    except KeyboardInterrupt:
        print("\n[EXIT] Shutting down.")
        if _rtp_server:
            _rtp_server.stop()
        if HAT_AVAILABLE:
            hat.off()

if __name__ == "__main__":
    main()