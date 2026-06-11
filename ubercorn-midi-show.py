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

# reusable per-frame buffers (avoids allocation every frame)
_BUF     = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
_LAYER   = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
_H_ARR   = np.zeros((HEIGHT, WIDTH),    dtype=np.float32)
_V_ARR   = np.zeros((HEIGHT, WIDTH),    dtype=np.float32)
_Y_GRID, _X_GRID = np.ogrid[:HEIGHT, :WIDTH]

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
    max_layers     = 4
    midi_host      = "0.0.0.0"
    midi_port      = 5004
    time           = 0.0

state = State()
_last_update_t = time.time()
_last_cfg_mtime = [0.0]
CHANNEL_EFFECTS = {}

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
        state.max_layers       = int(  c.get("max_layers",         4))
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
    
    v_np = np.asarray(v)
    if out is None:
        out = np.empty(v_np.shape + (3,), dtype=np.float32)

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
    cx:        int    = 0
    cy:        int    = 0

    @property
    def vel(self) -> float:
        return 0.3 + 0.7 * (self.velocity / 127.0)

layers: List[Layer] = []
layer_lock = threading.Lock()

# ── effects ────────────────────────────────────────────────────────────────────

def _get_rgb(L: Layer):
    """Calculate a single RGB scalar for the whole layer."""
    return hsv_to_rgb_array(L.hue, 1.0, L.vel * L.alpha)

def _effect_dot(L: Layer):
    _LAYER[:] = 0
    _LAYER[L.cy, L.cx] = _get_rgb(L)
    return _LAYER

def _effect_v_line(L: Layer):
    _LAYER[:] = 0
    _LAYER[:, L.cx] = _get_rgb(L)
    return _LAYER

def _effect_h_line(L: Layer):
    _LAYER[:] = 0
    _LAYER[L.cy, :] = _get_rgb(L)
    return _LAYER

def _effect_square(L: Layer):
    _LAYER[:] = 0
    rgb = _get_rgb(L)
    y1, y2 = max(0, L.cy-1), min(HEIGHT, L.cy+2)
    x1, x2 = max(0, L.cx-1), min(WIDTH, L.cx+2)
    _LAYER[y1:y2, x1:x2] = rgb
    return _LAYER

def _effect_cross(L: Layer):
    _LAYER[:] = 0
    rgb = _get_rgb(L)
    _LAYER[L.cy, :] = rgb
    _LAYER[:, L.cx] = rgb
    return _LAYER

def _effect_diamond(L: Layer):
    _LAYER[:] = 0
    rgb = _get_rgb(L)
    # Manhattan distance for a diamond shape
    mask = (np.abs(_X_GRID - L.cx) + np.abs(_Y_GRID - L.cy)) <= 2
    _LAYER[mask] = rgb
    return _LAYER

def _effect_x_cross(L: Layer):
    _LAYER[:] = 0
    rgb = _get_rgb(L)
    adx = np.abs(_X_GRID - L.cx)
    ady = np.abs(_Y_GRID - L.cy)
    # Diagonals: where absolute offsets are equal
    mask = (adx == ady) & (adx <= 3)
    _LAYER[mask] = rgb
    return _LAYER

def _effect_box(L: Layer):
    _LAYER[:] = 0
    rgb = _get_rgb(L)
    adx = np.abs(_X_GRID - L.cx)
    ady = np.abs(_Y_GRID - L.cy)
    # Hollow box using Chebyshev distance
    mask = (np.maximum(adx, ady) == 2)
    _LAYER[mask] = rgb
    return _LAYER

def _effect_circle(L: Layer):
    _LAYER[:] = 0
    rgb = _get_rgb(L)
    # Squared distance to avoid sqrt()
    d2 = (_X_GRID - L.cx)**2 + (_Y_GRID - L.cy)**2
    mask = (d2 >= 4) & (d2 <= 12)
    _LAYER[mask] = rgb
    return _LAYER

def _effect_plus_bold(L: Layer):
    _LAYER[:] = 0
    rgb = _get_rgb(L)
    # A thicker, limited-length cross
    mask = ((np.abs(_X_GRID - L.cx) <= 1) & (np.abs(_Y_GRID - L.cy) <= 3)) | \
           ((np.abs(_Y_GRID - L.cy) <= 1) & (np.abs(_X_GRID - L.cx) <= 3))
    _LAYER[mask] = rgb
    return _LAYER

_EFFECTS = [
    _effect_dot, _effect_v_line, _effect_h_line, _effect_square, _effect_cross,
    _effect_diamond, _effect_x_cross, _effect_box, _effect_circle, _effect_plus_bold
]

def init_channel_map():
    """Assign a random consistent effect to each of the 16 MIDI channels."""
    global CHANNEL_EFFECTS
    for ch in range(16):
        CHANNEL_EFFECTS[ch] = random.randint(0, len(_EFFECTS) - 1)

# ── background – simple hue-cycling gradient ───────────────────────────────────
_bg_t = [0.0]

def render_background(buf: np.ndarray):
    """Static gradient to save CPU."""
    t = _bg_t[0]
    buf[:] = [10, 0, 20] # Very dim purple wash
    _bg_t[0] += 0.01

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

def spawn_layer(note: int, velocity: int, channel: int = 0):
    with layer_lock:
        for l in layers:
            if l.note == note:
                l.decay = 0.05          # accelerate existing note
        lo, hi = _pick_palette(velocity)
        eid    = CHANNEL_EFFECTS.get(channel, 0)
        # Map note to grid position: low notes bottom/left, high notes top/right
        rel_note = np.clip(note - 36, 0, 72)
        cx = int(rel_note % WIDTH)
        cy = int((rel_note // 8) % HEIGHT)
        layer  = Layer(
            note      = note,
            velocity  = velocity,
            hue       = random.uniform(lo, hi),
            decay     = 0.1 + 0.1 * (velocity / 127.0),
            effect_id = eid,
            cx        = cx,
            cy        = cy
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
                l.decay = 0.15

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
    channel = (msg[0] & 0x0F)
    d1      = msg[1] if len(msg) > 1 else 0
    d2      = msg[2] if len(msg) > 2 else 0

    if status == MIDI_NOTE_ON and d2 > 0:
        print(f"[MIDI] NOTE ON  ch={channel+1} {_note_name(d1)}({d1}) vel={d2}")
        spawn_layer(d1, d2, channel=channel)
    elif status == MIDI_NOTE_OFF or (status == MIDI_NOTE_ON and d2 == 0):
        print(f"[MIDI] NOTE OFF ch={channel+1} {_note_name(d1)}({d1})")
        release_layer(d1)
    elif status == MIDI_CC:
        name = _CC_NAMES.get(d1, f"CC{d1}"); ch_num = channel + 1
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
        print(f"[MIDI] CC ch={ch_num} {name}({d1}) val={d2} → {v}")
    elif status == MIDI_PITCHBEND:
        print(f"[MIDI] PITCHBEND ch={channel+1} bend={((d2<<7)|d1)-8192:+d}")

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
            spawn_layer(random.randint(36, 84), random.randint(40, 100), channel=random.randint(0, 15))

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
    init_channel_map()
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