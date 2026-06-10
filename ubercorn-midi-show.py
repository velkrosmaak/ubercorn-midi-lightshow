#!/usr/bin/env python3
"""
ubercorn_midi_show.py  –  Ubercorn MIDI light show (Pi Zero optimised)
=======================================================================
All rendering is done in a single pre-allocated NumPy buffer.
Effects are intentionally simple (no sqrt/arctan per frame) so the Pi Zero
single core can keep up at 30+ fps even with several simultaneous layers.

FPS is printed to the console every second.

Install:   pip3 install numpy          (+ unicornhathd on Pi)
Run:       python3 ubercorn_midi_show.py
"""

import os, random, sys, threading, time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# ── local modules ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg_module
from rtpmidi_session import RTPMidiServer

# ── hardware ───────────────────────────────────────────────────────────────────
try:
    import unicornhathd as hat
    HAT_AVAILABLE = True
    WIDTH, HEIGHT = hat.get_shape()
except ImportError:
    HAT_AVAILABLE = False
    WIDTH, HEIGHT = 16, 16
    print("[WARN] unicornhathd not found – sim mode")

# ── coordinate grids – computed ONCE at startup ────────────────────────────────
# All float32, shape (HEIGHT, WIDTH)
_Y, _X   = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
_CX, _CY = WIDTH / 2.0 - 0.5, HEIGHT / 2.0 - 0.5
_DX      = (_X - _CX) / (WIDTH  / 2.0)   # normalised –1..1
_DY      = (_Y - _CY) / (HEIGHT / 2.0)
_DIAG    = (_X + _Y).astype(np.float32)   # 0 .. W+H-2
_DIST_SQ = (_DX**2 + _DY**2).astype(np.float32)   # 0..~2  (no sqrt needed)
_DIST    = np.sqrt(_DIST_SQ).astype(np.float32)    # used only in effects that need it
_ANGLE   = np.arctan2(_DY, _DX).astype(np.float32) # –π..π  (computed once)
_XNORM   = (_X / (WIDTH  - 1)).astype(np.float32)  # 0..1
_YNORM   = (_Y / (HEIGHT - 1)).astype(np.float32)

# reusable per-frame buffers (avoids allocation every frame)
_BUF     = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
_LAYER   = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
_H_ARR   = np.zeros((HEIGHT, WIDTH),    dtype=np.float32)
_V_ARR   = np.zeros((HEIGHT, WIDTH),    dtype=np.float32)

# ── state ──────────────────────────────────────────────────────────────────────
class State:
    brightness     = 0.8
    fps            = 30          # 30 is realistic on Pi Zero; push to 60 on faster hardware
    rotation       = 0
    speed          = 1.0
    hue_shift      = 0.0
    saturation     = 1.0
    strobe_rate    = 0.0
    particle_den   = 48
    blur           = 0.0
    sustain        = False
    blackout       = False
    attract_mode   = True
    attract_interval = 4.0
    max_layers     = 6
    midi_host      = "0.0.0.0"
    midi_port      = 5004
    time           = 0.0

state = State()
_last_update_t = time.time()
_last_cfg_mtime = [0.0]

def reload_config():
    try:
        if not os.path.exists(cfg_module.CONFIG_PATH):
            print(f"[CONFIG] creating defaults at {cfg_module.CONFIG_PATH}")
            cfg_module.save(cfg_module.DEFAULTS.copy())
            return
        mtime = os.path.getmtime(cfg_module.CONFIG_PATH)
        if mtime <= _last_cfg_mtime[0]:
            return
        _last_cfg_mtime[0] = mtime
        c = cfg_module.load()
        state.brightness       = float(c.get("brightness",       0.8))
        state.fps              = int(  c.get("fps",              30))
        state.rotation         = int(  c.get("rotation",          0))
        state.speed            = float(c.get("speed",             1.0))
        state.hue_shift        = float(c.get("hue_shift",         0.0))
        state.saturation       = float(c.get("saturation",        1.0))
        state.strobe_rate      = float(c.get("strobe_rate",       0.0))
        state.particle_den     = int(  c.get("particle_den",     48))
        state.blur             = float(c.get("blur",              0.0))
        state.sustain          = bool( c.get("sustain",          False))
        state.blackout         = bool( c.get("blackout",         False))
        state.attract_mode     = bool( c.get("attract_mode",     True))
        state.attract_interval = float(c.get("attract_interval",  4.0))
        state.max_layers       = int(  c.get("max_layers",         6))
        if HAT_AVAILABLE:
            hat.rotation(state.rotation)
        if c.get("restart", False):
            cfg_module.update({"restart": False})
            if HAT_AVAILABLE: hat.off()
            os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        print(f"[CONFIG] error: {e}")

# ── fast vectorised HSV→RGB ────────────────────────────────────────────────────
def hsv_to_rgb_array(h, s, v, out=None) -> np.ndarray:
    """
    Optimised HSV to RGB conversion.
    h: scalar or (H,W) array, s: scalar, v: (H,W) array.
    """
    h_shift = (h + state.hue_shift / 360.0) % 1.0
    s_val = float(np.clip(s * state.saturation, 0.0, 1.0))
    
    if out is None:
        out = np.empty(v.shape + (3,), dtype=np.float32)

    if np.isscalar(h_shift):
        # Fast path for uniform hue - calculate RGB scalar once
        i = int(h_shift * 6.0)
        f = h_shift * 6.0 - i
        p, q, t = (1.0 - s_val), (1.0 - f * s_val), (1.0 - (1.0 - f) * s_val)
        i %= 6
        rgb = [(1.0, t, p), (q, 1.0, p), (p, 1.0, t), (p, q, 1.0), (t, p, 1.0), (1.0, p, q)][i]
        
        out[..., 0] = v * (rgb[0] * 255.0)
        out[..., 1] = v * (rgb[1] * 255.0)
        out[..., 2] = v * (rgb[2] * 255.0)
    else:
        # Vectorised path for varying hue
        i = (h_shift * 6.0).astype(np.int32)
        f = h_shift * 6.0 - i
        v255 = v * 255.0
        p, q, t = v255 * (1.0 - s_val), v255 * (1.0 - f * s_val), v255 * (1.0 - (1.0 - f) * s_val)
        i %= 6
        
        # Boolean indexing is faster than np.select for 16x16
        for val in range(6):
            m = (i == val)
            if val == 0: out[m] = np.stack([v255[m], t[m], p[m]], axis=-1)
            elif val == 1: out[m] = np.stack([q[m], v255[m], p[m]], axis=-1)
            elif val == 2: out[m] = np.stack([p[m], v255[m], t[m]], axis=-1)
            elif val == 3: out[m] = np.stack([p[m], q[m], v255[m]], axis=-1)
            elif val == 4: out[m] = np.stack([t[m], p[m], v255[m]], axis=-1)
            elif val == 5: out[m] = np.stack([v255[m], p[m], q[m]], axis=-1)
            
    return out

# ── layer ──────────────────────────────────────────────────────────────────────
@dataclass
class Layer:
    note:      int
    velocity:  int
    hue:       float  = 0.0
    alpha:     float  = 1.0
    decay:     float  = 0.015
    alive:     bool   = True
    effect_id: int    = 0
    phase:     float  = 0.0

    @property
    def vel(self) -> float:
        return 0.3 + 0.7 * (self.velocity / 127.0)

layers: List[Layer] = []
layer_lock = threading.Lock()

# ── effects ────────────────────────────────────────────────────────────────────
# Each writes into the pre-allocated _LAYER buffer and returns it.
# NO allocations, NO Python pixel loops, NO expensive trig per frame
# (arctan/sqrt are pre-computed at startup in _DIST / _ANGLE).

NUM_EFFECTS = 8

def _effect_flash(L: Layer):
    """Solid colour fill, fades out – simplest possible effect."""
    sat = max(0.0, 1.0 - L.alpha * 0.5)
    np.copyto(_H_ARR, L.hue)
    np.multiply(L.vel * L.alpha, np.ones((HEIGHT, WIDTH), dtype=np.float32), out=_V_ARR)
    rgb = hsv_to_rgb_array(_H_ARR, sat, _V_ARR)
    np.copyto(_LAYER, rgb)
    return _LAYER

def _effect_radial(L: Layer):
    """Expanding ring using pre-computed _DIST."""
    radius    = L.phase * 1.4          # _DIST is normalised 0..~1.4
    thickness = 0.2
    dist      = np.abs(_DIST - radius)
    np.clip((1.0 - dist / max(thickness, 0.01)) * L.alpha * L.vel, 0, 1, out=_V_ARR)
    np.copyto(_H_ARR, L.hue)
    np.copyto(_LAYER, hsv_to_rgb_array(_H_ARR, 1.0, _V_ARR))
    return _LAYER

def _effect_sweep(L: Layer):
    """Diagonal band sweeping across – uses pre-computed _DIAG."""
    sweep = L.phase * (WIDTH + HEIGHT) * 1.3 - 3.0
    dist  = np.abs(_DIAG - sweep)
    np.clip((1.0 - dist / 4.0) * L.alpha * L.vel, 0, 1, out=_V_ARR)
    np.add(L.hue, dist * 0.02, out=_H_ARR)
    np.mod(_H_ARR, 1.0, out=_H_ARR)
    np.copyto(_LAYER, hsv_to_rgb_array(_H_ARR, 1.0, _V_ARR))
    return _LAYER

def _effect_plasma(L: Layer):
    """Sine-wave interference – fast because sin is vectorised."""
    t = L.phase * 4.0
    val = np.sin(_X * 0.6 + t)
    np.clip((0.5 + 0.5 * val) * L.alpha * L.vel, 0, 1, out=_V_ARR)
    np.add(L.hue, val * 0.15, out=_H_ARR)
    np.mod(_H_ARR, 1.0, out=_H_ARR)
    np.copyto(_LAYER, hsv_to_rgb_array(_H_ARR, 1.0, _V_ARR))
    return _LAYER

def _effect_columns(L: Layer):
    """Rising columns – fully vectorised."""
    x_idxs = np.arange(WIDTH, dtype=np.float32)
    heights = ((0.5 + 0.5 * np.sin(L.phase * 5 + x_idxs * 0.7)) * HEIGHT).astype(np.int32)
    hues = (L.hue + x_idxs / WIDTH * 0.25) % 1.0
    y_idxs = (HEIGHT - 1 - np.arange(HEIGHT)).reshape(-1, 1)
    mask = y_idxs < heights
    v_grad = (1.0 - np.arange(HEIGHT).reshape(-1, 1) / HEIGHT * 0.6)
    v_arr = mask * v_grad * (L.vel * L.alpha)
    h_arr = np.broadcast_to(hues, (HEIGHT, WIDTH))
    return hsv_to_rgb_array(h_arr, 1.0, v_arr, out=_LAYER)

def _effect_strobe(L: Layer):
    """Hard strobe flash."""
    if int(L.phase * 15) % 2 == 0:
        _LAYER[:] = 0
    else:
        np.copyto(_H_ARR, L.hue)
        np.multiply(L.vel * L.alpha, np.ones((HEIGHT, WIDTH), dtype=np.float32), out=_V_ARR)
        np.copyto(_LAYER, hsv_to_rgb_array(_H_ARR, 0.5, _V_ARR))
    return _LAYER

def _effect_diamond(L: Layer):
    """Expanding diamond pulse using Manhattan distance."""
    manhattan = (np.abs(_DX) + np.abs(_DY)).astype(np.float32)  # 0..~2
    size  = L.phase * 2.2
    dist  = np.abs(manhattan - size)
    np.clip((1.0 - dist / 0.25) * L.alpha * L.vel, 0, 1, out=_V_ARR)
    np.add(L.hue, dist * 0.05, out=_H_ARR)
    np.mod(_H_ARR, 1.0, out=_H_ARR)
    np.copyto(_LAYER, hsv_to_rgb_array(_H_ARR, 1.0, _V_ARR))
    return _LAYER

def _effect_spiral(L: Layer):
    """Spiral arms using pre-computed _ANGLE and _DIST."""
    t    = L.phase * 4.0
    arms = 3
    # simplified rotation, removed distance twist
    spiral = (_ANGLE - t) % (2 * np.pi / arms)
    dist   = np.minimum(spiral, 2 * np.pi / arms - spiral).astype(np.float32)
    fade   = np.clip(1.0 - _DIST / 1.5, 0, 1).astype(np.float32)
    np.clip((1.0 - dist / 0.35) * fade * L.alpha * L.vel, 0, 1, out=_V_ARR)
    np.copyto(_H_ARR, L.hue)
    np.mod(_H_ARR, 1.0, out=_H_ARR)
    np.copyto(_LAYER, hsv_to_rgb_array(_H_ARR, 1.0, _V_ARR))
    return _LAYER

_EFFECTS = [
    _effect_flash,
    _effect_radial,
    _effect_sweep,
    _effect_plasma,
    _effect_columns,
    _effect_strobe,
    _effect_diamond,
    _effect_spiral,
]

def _assign_effect(velocity: int) -> int:
    if velocity > 110:
        return random.choice([0, 5])          # flash / strobe
    if velocity > 85:
        return random.choice([1, 6, 7])       # radial / diamond / spiral
    if velocity > 60:
        return random.choice([2, 3, 4])       # sweep / plasma / columns
    return random.choice([2, 3, 4])

# ── background – simple hue-cycling gradient ───────────────────────────────────
_bg_t = [0.0]

def render_background(buf: np.ndarray):
    """Write a slow drifting gradient directly into buf."""
    t = _bg_t[0]
    v = (np.sin(_X * 0.4 + t * 0.25) * np.sin(_Y * 0.35 + t * 0.18)) * 0.10
    np.clip(v, 0, 1, out=_V_ARR)
    np.add(t * 0.03 + _XNORM * 0.12 + _YNORM * 0.08, 0, out=_H_ARR)
    np.mod(_H_ARR, 1.0, out=_H_ARR)
    np.copyto(buf, hsv_to_rgb_array(_H_ARR, 0.8, _V_ARR))
    _bg_t[0] += (1.0 / max(state.fps, 1)) * state.speed

# ── compositing ────────────────────────────────────────────────────────────────

def composite() -> np.ndarray:
    """Render everything into _BUF and return it."""
    if state.blackout:
        _BUF[:] = 0
        return _BUF

    render_background(_BUF)

    with layer_lock:
        active = [l for l in layers if l.alive]

    for layer in active:
        if layer.alpha < 0.005:
            continue
        layer_frame = _EFFECTS[layer.effect_id](layer)
        np.add(_BUF, layer_frame, out=_BUF)

    np.clip(_BUF, 0, 255, out=_BUF)

    # blur: fast 5-tap cross blur using roll
    if state.blur > 0.01:
        blurred = (np.roll(_BUF, 1, 0) + np.roll(_BUF, -1, 0) +
                   np.roll(_BUF, 1, 1) + np.roll(_BUF, -1, 1) + _BUF) / 5.0
        np.add(_BUF, (blurred - _BUF) * state.blur, out=_BUF)
        np.clip(_BUF, 0, 255, out=_BUF)

    # global strobe
    if state.strobe_rate > 0:
        period = 1.0 / state.strobe_rate
        if int(state.time / (period / 2)) % 2 == 1:
            _BUF[:] = 0

    return _BUF

# ── layer lifecycle ────────────────────────────────────────────────────────────
PALETTE_BANKS = [
    (0.0,0.08),(0.08,0.17),(0.28,0.42),(0.55,0.70),(0.70,0.85),(0.85,1.00)
]

def _pick_palette(velocity: int) -> Tuple[float, float]:
    return PALETTE_BANKS[(velocity // 22) % len(PALETTE_BANKS)]

def spawn_layer(note: int, velocity: int):
    with layer_lock:
        for l in layers:
            if l.note == note:
                l.decay = 0.05          # accelerate existing note
        lo, hi = _pick_palette(velocity)
        eid    = _assign_effect(velocity)
        layer  = Layer(
            note      = note,
            velocity  = velocity,
            hue       = random.uniform(lo, hi),
            decay     = 0.012 + 0.025 * (velocity / 127.0),
            effect_id = eid,
        )
        layers.append(layer)
        if len(layers) > state.max_layers:
            layers.pop(0)
    print(f"[SHOW] note={note}({_note_name(note)}) vel={velocity} "
          f"effect={_EFFECTS[eid].__name__} layers={len(layers)}")

def release_layer(note: int):
    with layer_lock:
        for l in layers:
            if l.note == note and l.alive and not state.sustain:
                l.decay = 0.075

def update_layers():
    global _last_update_t
    now = time.time()
    dt_real = now - _last_update_t
    _last_update_t = now
    dt = dt_real * state.speed
    with layer_lock:
        for l in layers:
            if not l.alive:
                continue
            l.phase += dt
            if state.sustain:
                l.alpha = min(l.alpha + 0.2 * dt_real, 1.0)
            else:
                l.alpha -= l.decay * (dt_real * 30.0)
            if l.alpha <= 0:
                l.alive = False
        layers[:] = [l for l in layers if l.alive or l.phase < 0.3]

# ── MIDI ───────────────────────────────────────────────────────────────────────
MIDI_NOTE_OFF  = 0x80
MIDI_NOTE_ON   = 0x90
MIDI_CC        = 0xB0
MIDI_PITCHBEND = 0xE0

last_midi_time = [time.time()]
_NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
_CC_NAMES   = {1:'Brightness',7:'Speed',10:'HueShift',16:'Strobe',
               17:'Particles',18:'Saturation',64:'Sustain',74:'Blur',121:'Reset'}

def _note_name(n: int) -> str:
    return f"{_NOTE_NAMES[n%12]}{n//12-1}"

def handle_midi_message(msg: bytes):
    last_midi_time[0] = time.time()
    if len(msg) < 2:
        return
    status  = msg[0] & 0xF0
    channel = (msg[0] & 0x0F) + 1
    d1      = msg[1] if len(msg) > 1 else 0
    d2      = msg[2] if len(msg) > 2 else 0

    if status == MIDI_NOTE_ON and d2 > 0:
        print(f"[MIDI] NOTE ON  ch={channel} {_note_name(d1)}({d1}) vel={d2}")
        spawn_layer(d1, d2)
    elif status == MIDI_NOTE_OFF or (status == MIDI_NOTE_ON and d2 == 0):
        print(f"[MIDI] NOTE OFF ch={channel} {_note_name(d1)}({d1})")
        release_layer(d1)
    elif status == MIDI_CC:
        name = _CC_NAMES.get(d1, f"CC{d1}")
        if   d1 == 1:   state.brightness  = 0.1 + 0.9*(d2/127);  v=f"{state.brightness:.2f}"
        elif d1 == 7:   state.speed       = 0.2 + 3.8*(d2/127);  v=f"{state.speed:.2f}x"
        elif d1 == 10:  state.hue_shift   = d2/127*360;           v=f"{state.hue_shift:.0f}°"
        elif d1 == 16:  state.strobe_rate = d2/127*30;            v=f"{state.strobe_rate:.1f}Hz"
        elif d1 == 17:  state.particle_den= d2;                   v=str(d2)
        elif d1 == 18:  state.saturation  = d2/127;               v=f"{state.saturation:.2f}"
        elif d1 == 64:  state.sustain     = d2>=64;               v='ON' if state.sustain else 'OFF'
        elif d1 == 74:  state.blur        = d2/127;               v=f"{state.blur:.2f}"
        elif d1 == 121:
            state.brightness=0.8; state.speed=1.0; state.hue_shift=0.0
            state.strobe_rate=0.0; state.saturation=1.0; state.sustain=False; state.blur=0.0
            v="RESET"
        else: v="(unmapped)"
        print(f"[MIDI] CC ch={channel} {name}({d1}) val={d2} → {v}")
    elif status == MIDI_PITCHBEND:
        print(f"[MIDI] PITCHBEND ch={channel} bend={((d2<<7)|d1)-8192:+d}")

# ── RTP-MIDI ───────────────────────────────────────────────────────────────────
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
            spawn_layer(random.randint(36, 84), random.randint(40, 100))

# ── display + FPS counter ──────────────────────────────────────────────────────
def display_loop():
    if HAT_AVAILABLE:
        hat.rotation(state.rotation)
        hat.brightness(1.0)   # we apply brightness via array multiply
    else:
        print("\033[2J")

    cfg_tick   = 0
    fps_frames = 0
    fps_t0     = time.time()

    while True:
        t_start = time.time()
        # calculate dt for this specific frame
        reload_dt = t_start - (state.time_last if hasattr(state, 'time_last') else t_start)
        state.time += reload_dt
        state.time_last = t_start

        # config reload every 60 frames (~2 s at 30 fps)
        cfg_tick += 1
        if cfg_tick >= 60:
            cfg_tick = 0
            reload_config()

        update_layers()
        frame = composite()    # float32 (H,W,3), already clipped 0-255

        # apply brightness and convert to uint8
        out = np.clip(frame * state.brightness, 0, 255).astype(np.uint8)

        if HAT_AVAILABLE:
            if hasattr(hat, 'set_pixel_array'):
                hat.set_pixel_array(out)
            else:
                for y in range(HEIGHT):
                    for x in range(WIDTH):
                        hat.set_pixel(x, y, int(out[y,x,0]),
                                             int(out[y,x,1]),
                                             int(out[y,x,2]))
            hat.show()
        else:
            # ascii sim
            chars = " ░▒▓█"
            lum   = out[::2].mean(axis=2)
            rows  = ["".join(chars[min(int(v/255*(len(chars)-1)),4)]
                             for v in row) for row in lum]
            print("\033[H" + "\n".join(rows), end="", flush=True)

        # FPS counter – print once per second
        fps_frames += 1
        now = time.time()
        if now - fps_t0 >= 1.0:
            fps_actual = fps_frames / (now - fps_t0)
            target     = state.fps
            bar_len    = 20
            bar_fill   = int(bar_len * min(fps_actual / target, 1.0))
            bar        = "█" * bar_fill + "░" * (bar_len - bar_fill)
            n_layers   = sum(1 for l in layers if l.alive)
            print(f"\r[FPS] {fps_actual:5.1f}/{target}  [{bar}]  "
                  f"layers={n_layers}  blur={state.blur:.2f}  "
                  f"speed={state.speed:.1f}x      ",
                  end="", flush=True)
            fps_frames = 0
            fps_t0     = now

        elapsed = time.time() - t_start
        sleep_t = max(0.0, 1.0 / max(state.fps, 1) - elapsed)
        time.sleep(sleep_t)

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    reload_config()
    print("╔══════════════════════════════════════════╗")
    print("║  Ubercorn MIDI Light Show  [optimised]   ║")
    print(f"║  {WIDTH}×{HEIGHT} grid  target {state.fps} fps            ║")
    print(f"║  Apple MIDI ports {state.midi_port}/{state.midi_port+1}              ║")
    print("╚══════════════════════════════════════════╝")

    start_rtp_server()
    threading.Thread(target=attract_mode_thread, daemon=True).start()

    try:
        display_loop()
    except KeyboardInterrupt:
        print("\n[EXIT]")
        if _rtp_server:
            _rtp_server.stop()
        if HAT_AVAILABLE:
            hat.off()

if __name__ == "__main__":
    main()