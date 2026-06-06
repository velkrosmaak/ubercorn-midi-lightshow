#!/usr/bin/env python3
"""
ubercorn_midi_show.py
=====================
MIDI-over-WiFi light show for Raspberry Pi Zero + Pimoroni Ubercorn (Unicorn HAT HD 16×16).

Now config-file aware: reads ubercorn_config.json every second and hot-reloads
all parameters without restarting. The web UI (webapp.py) writes to that file.

Run standalone:
    python3 ubercorn_midi_show.py

Or managed by the web UI via systemd / subprocess.
"""

import colorsys
import math
import os
import random
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import List, Tuple

# ── local config module ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg_module

# ── optional hardware import ───────────────────────────────────────────────────
try:
    import unicornhathd as hat
    HAT_AVAILABLE = True
    WIDTH, HEIGHT = hat.get_shape()
except ImportError:
    HAT_AVAILABLE = False
    WIDTH, HEIGHT = 16, 16
    print("[WARN] unicornhathd not found – running in sim mode.")

# ── runtime state (populated from config) ─────────────────────────────────────
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
        # existence check FIRST, before getmtime
        if not os.path.exists(cfg_module.CONFIG_PATH):
            cfg_module.save(cfg_module.DEFAULTS.copy())
            return

        mtime = os.path.getmtime(cfg_module.CONFIG_PATH)
        if mtime <= _last_cfg_mtime[0]:
            return
        _last_cfg_mtime[0] = mtime
        c = cfg_module.load()
        state.brightness       = float(c.get("brightness", 0.8))
        state.fps              = int(c.get("fps", 60))
        state.rotation         = int(c.get("rotation", 0))
        state.speed            = float(c.get("speed", 1.0))
        state.hue_shift        = float(c.get("hue_shift", 0.0))
        state.strobe_rate      = float(c.get("strobe_rate", 0.0))
        state.particle_den     = int(c.get("particle_den", 64))
        state.saturation       = float(c.get("saturation", 1.0))
        state.sustain          = bool(c.get("sustain", False))
        state.blur             = float(c.get("blur", 0.0))
        state.blackout         = bool(c.get("blackout", False))
        state.attract_mode     = bool(c.get("attract_mode", True))
        state.attract_interval = float(c.get("attract_interval", 4.0))
        state.max_layers       = int(c.get("max_layers", 12))

        if c.get("restart", False):
            cfg_module.update({"restart": False})
            print("[CONFIG] Restart requested – re-execing...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        print(f"[CONFIG] reload error: {e}")

# ── constants ──────────────────────────────────────────────────────────────────
MIDI_NOTE_OFF  = 0x80
MIDI_NOTE_ON   = 0x90
MIDI_CC        = 0xB0
MIDI_PITCHBEND = 0xE0
UDP_BUFSIZE    = 1024

PALETTE_BANKS = [
    (0.0,  0.08),
    (0.08, 0.17),
    (0.28, 0.42),
    (0.55, 0.70),
    (0.70, 0.85),
    (0.85, 1.00),
]

# ── helpers ────────────────────────────────────────────────────────────────────

def hsv_to_rgb(h, s, v) -> Tuple[int, int, int]:
    h = (h + state.hue_shift / 360.0) % 1.0
    s = s * state.saturation
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)

def lerp(a, b, t):
    return a + (b - a) * t

def clamp(v, lo=0, hi=255):
    return max(lo, min(hi, int(v)))

def velocity_to_value(velocity: int) -> float:
    return 0.3 + 0.7 * (velocity / 127.0)

def pick_palette(velocity: int) -> Tuple[float, float]:
    return PALETTE_BANKS[(velocity // 22) % len(PALETTE_BANKS)]

# ── layer system ───────────────────────────────────────────────────────────────

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

layers: List[Layer] = []
layer_lock = threading.Lock()

# ── effects ────────────────────────────────────────────────────────────────────

def effect_radial_burst(layer: Layer):
    pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    t = layer.phase
    cx, cy = WIDTH/2 - 0.5, HEIGHT/2 - 0.5
    radius = t * (WIDTH * 0.8)
    thickness = 2.5 + 1.5 * math.sin(t * 3)
    v = velocity_to_value(layer.velocity)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            d = math.sqrt((x-cx)**2 + (y-cy)**2)
            dist = abs(d - radius)
            if dist < thickness:
                br = (1.0 - dist/thickness) * layer.alpha
                r,g,b = hsv_to_rgb(layer.hue, 1.0, v * br)
                pix[y][x] = (r, g, b)
    return pix

def effect_note_flash(layer: Layer):
    pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    v = velocity_to_value(layer.velocity)
    sat = max(0, 1.0 - layer.alpha * 0.4)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            r,g,b = hsv_to_rgb(layer.hue, sat, v * layer.alpha)
            pix[y][x] = (r, g, b)
    return pix

def effect_diagonal_sweep(layer: Layer):
    pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    sweep = (layer.phase * (WIDTH + HEIGHT) * 1.4) - 4
    v = velocity_to_value(layer.velocity)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            dist = abs((x + y) - sweep)
            if dist < 4:
                br = (1.0 - dist/4) * layer.alpha
                h2 = (layer.hue + dist * 0.03) % 1.0
                r,g,b = hsv_to_rgb(h2, 1.0, v * br)
                pix[y][x] = (r, g, b)
    return pix

def effect_sparkle(layer: Layer):
    pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    rng = random.Random(layer.note + int(layer.phase * 10))
    count = int(state.particle_den / 5) + 5
    v = velocity_to_value(layer.velocity)
    for _ in range(count):
        x = rng.randint(0, WIDTH-1)
        y = rng.randint(0, HEIGHT-1)
        h = (layer.hue + rng.uniform(-0.1, 0.1)) % 1.0
        br = rng.uniform(0.5, 1.0) * layer.alpha * v
        r,g,b = hsv_to_rgb(h, rng.uniform(0.7, 1.0), br)
        pix[y][x] = (r, g, b)
    return pix

def effect_spiral(layer: Layer):
    pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    cx, cy = WIDTH/2 - 0.5, HEIGHT/2 - 0.5
    t = layer.phase * math.pi * 4
    v = velocity_to_value(layer.velocity)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            dx, dy = x - cx, y - cy
            r = math.sqrt(dx*dx + dy*dy) + 0.001
            angle = math.atan2(dy, dx)
            sp = (angle + r * 0.6 - t) % (2*math.pi / 3)
            dist = min(sp, 2*math.pi/3 - sp)
            if dist < 0.4:
                br = (1.0 - dist/0.4) * layer.alpha * (1.0 - r/(WIDTH*0.85))
                h = (layer.hue + r/WIDTH * 0.3) % 1.0
                rr,g,b = hsv_to_rgb(h, 1.0, v * max(0, br))
                pix[y][x] = (rr, g, b)
    return pix

def effect_columns(layer: Layer):
    pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    v = velocity_to_value(layer.velocity)
    for x in range(WIDTH):
        h_frac = 0.5 + 0.5 * math.sin(layer.phase * 6 + x * 0.8)
        col_h = int(h_frac * HEIGHT)
        h = (layer.hue + x/WIDTH * 0.3) % 1.0
        for y in range(col_h):
            br = (1.0 - y/HEIGHT * 0.5) * layer.alpha * v
            rr,g,b = hsv_to_rgb(h, 1.0, br)
            pix[HEIGHT-1-y][x] = (rr, g, b)
    return pix

def effect_plasma(layer: Layer):
    pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    t = layer.phase * 4
    v = velocity_to_value(layer.velocity)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            val = (math.sin(x*0.5+t) + math.sin(y*0.5+t*1.3) +
                   math.sin((x+y)*0.35+t*0.7) +
                   math.sin(math.sqrt(x*x+y*y)*0.4+t)) / 4
            h = (layer.hue + val * 0.2) % 1.0
            br = (0.5 + 0.5*val) * layer.alpha * v
            rr,g,b = hsv_to_rgb(h, 1.0, max(0, br))
            pix[y][x] = (rr, g, b)
    return pix

def effect_strobe_burst(layer: Layer):
    pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    if int(layer.phase * 20) % 2 == 0:
        return pix
    v = velocity_to_value(layer.velocity)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            r,g,b = hsv_to_rgb(layer.hue, 0.6, v * layer.alpha)
            pix[y][x] = (r, g, b)
    return pix

def effect_diamond(layer: Layer):
    pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    cx, cy = WIDTH//2, HEIGHT//2
    size = layer.phase * max(WIDTH, HEIGHT) * 1.2
    v = velocity_to_value(layer.velocity)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            d = abs(x - cx) + abs(y - cy)
            dist = abs(d - size)
            if dist < 3:
                br = (1.0 - dist/3) * layer.alpha
                h = (layer.hue + dist * 0.04) % 1.0
                rr,g,b = hsv_to_rgb(h, 1.0, v * br)
                pix[y][x] = (rr, g, b)
    return pix

def effect_meteor(layer: Layer):
    """Diagonal streaks raining across the grid."""
    pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    v = velocity_to_value(layer.velocity)
    rng = random.Random(layer.note)
    for _ in range(4):
        sx = rng.randint(0, WIDTH-1)
        speed_m = rng.uniform(0.8, 1.5)
        tail = 6
        for i in range(tail):
            t_off = layer.phase * speed_m * HEIGHT
            px = (sx + int(t_off) - i) % WIDTH
            py = (int(t_off) - i) % HEIGHT
            br = ((tail-i)/tail) * layer.alpha * v
            h = (layer.hue + i * 0.02) % 1.0
            rr,g,b = hsv_to_rgb(h, 1.0, br)
            pix[py][px] = (rr, g, b)
    return pix

def effect_ripple(layer: Layer):
    """Multiple concentric rings from random points."""
    pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    rng = random.Random(layer.note)
    v = velocity_to_value(layer.velocity)
    for _ in range(3):
        cx = rng.uniform(2, WIDTH-3)
        cy = rng.uniform(2, HEIGHT-3)
        radius = layer.phase * WIDTH * 0.7
        for y in range(HEIGHT):
            for x in range(WIDTH):
                d = math.sqrt((x-cx)**2 + (y-cy)**2)
                dist = abs(d - radius)
                if dist < 2:
                    br = (1.0 - dist/2) * layer.alpha * 0.6 * v
                    h = (layer.hue + radius/WIDTH * 0.2) % 1.0
                    rr,g,b = hsv_to_rgb(h, 1.0, br)
                    pix[y][x] = (clamp(pix[y][x][0]+rr),
                                 clamp(pix[y][x][1]+g),
                                 clamp(pix[y][x][2]+b))
    return pix

EFFECTS = [
    effect_radial_burst, effect_note_flash, effect_diagonal_sweep,
    effect_sparkle, effect_spiral, effect_columns, effect_plasma,
    effect_strobe_burst, effect_diamond, effect_meteor, effect_ripple,
]

def assign_effect(velocity: int, note: int):
    if velocity > 110:
        return random.choice([effect_strobe_burst, effect_note_flash, effect_radial_burst])
    if velocity > 85:
        return random.choice([effect_plasma, effect_spiral, effect_diamond, effect_ripple])
    if velocity > 60:
        return random.choice([effect_radial_burst, effect_diagonal_sweep, effect_columns, effect_meteor])
    return random.choice([effect_sparkle, effect_columns, effect_diagonal_sweep, effect_ripple])

# ── background ─────────────────────────────────────────────────────────────────

class Background:
    def __init__(self):
        self.t = 0.0

    def render(self):
        pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
        t = self.t
        for y in range(HEIGHT):
            for x in range(WIDTH):
                v = (math.sin(x*0.4+t*0.3) * math.sin(y*0.3+t*0.2)) * 0.12
                if v > 0:
                    h = (t*0.04 + x/WIDTH*0.15 + y/HEIGHT*0.1) % 1.0
                    r,g,b = hsv_to_rgb(h, 0.8, v)
                    pix[y][x] = (r, g, b)
        self.t += (1.0/state.fps) * state.speed
        return pix

background = Background()

# ── compositing ────────────────────────────────────────────────────────────────

def blur_pass(pix, amount):
    if amount < 0.01:
        return pix
    out = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    for y in range(HEIGHT):
        for x in range(WIDTH):
            rs = gs = bs = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx, ny = (x+dx) % WIDTH, (y+dy) % HEIGHT
                    r,g,b = pix[ny][nx]
                    rs+=r; gs+=g; bs+=b
            ar, ag, ab = rs//9, gs//9, bs//9
            or_,og,ob = pix[y][x]
            out[y][x] = (int(lerp(or_,ar,amount)),
                         int(lerp(og,ag,amount)),
                         int(lerp(ob,ab,amount)))
    return out

def composite():
    if state.blackout:
        return [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]

    frame = background.render()
    with layer_lock:
        active = [l for l in layers if l.alive]

    for layer in active:
        lpix = layer.effect_fn(layer)
        for y in range(HEIGHT):
            for x in range(WIDTH):
                lr,lg,lb = lpix[y][x]
                fr,fg,fb = frame[y][x]
                frame[y][x] = (clamp(fr+lr), clamp(fg+lg), clamp(fb+lb))

    frame = blur_pass(frame, state.blur)

    if state.strobe_rate > 0:
        period = 1.0 / state.strobe_rate
        if int(state.time / (period/2)) % 2 == 1:
            return [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]

    return frame

# ── layer lifecycle ────────────────────────────────────────────────────────────

def spawn_layer(note: int, velocity: int):
    with layer_lock:
        for l in layers:
            if l.note == note:
                l.decay = 0.04
        lo, hi = pick_palette(velocity)
        layer = Layer(
            note      = note,
            velocity  = velocity,
            hue       = random.uniform(lo, hi),
            decay     = lerp(0.005, 0.025, velocity/127.0),
            effect_fn = assign_effect(velocity, note),
            born      = state.time,
        )
        layers.append(layer)
        if len(layers) > state.max_layers:
            layers.pop(0)

def release_layer(note: int):
    with layer_lock:
        for l in layers:
            if l.note == note and l.alive:
                if not state.sustain:
                    l.decay = 0.02

def update_layers():
    dt = (1.0/state.fps) * state.speed
    with layer_lock:
        for l in layers:
            if not l.alive:
                continue
            l.phase += dt
            if not state.sustain:
                l.alpha -= l.decay
            else:
                l.alpha = min(l.alpha + 0.01, 1.0)
            if l.alpha <= 0:
                l.alive = False
        layers[:] = [l for l in layers if l.alive or l.phase < 0.5]

# ── MIDI parsing ───────────────────────────────────────────────────────────────

last_midi_time = [time.time()]

def handle_midi_message(msg_bytes: bytes):
    last_midi_time[0] = time.time()
    if len(msg_bytes) < 2:
        return
    status = msg_bytes[0] & 0xF0
    data1  = msg_bytes[1] if len(msg_bytes) > 1 else 0
    data2  = msg_bytes[2] if len(msg_bytes) > 2 else 0

    if status == MIDI_NOTE_ON and data2 > 0:
        spawn_layer(data1, data2)
    elif status == MIDI_NOTE_OFF or (status == MIDI_NOTE_ON and data2 == 0):
        release_layer(data1)
    elif status == MIDI_CC:
        cc, val = data1, data2
        if   cc == 1:   state.brightness  = 0.1 + 0.9*(val/127)
        elif cc == 7:   state.speed       = 0.2 + 3.8*(val/127)
        elif cc == 10:  state.hue_shift   = val/127*360
        elif cc == 16:  state.strobe_rate = val/127*30
        elif cc == 17:  state.particle_den = val
        elif cc == 18:  state.saturation  = val/127
        elif cc == 64:  state.sustain     = (val >= 64)
        elif cc == 74:  state.blur        = val/127
        elif cc == 121:
            state.brightness=0.8; state.speed=1.0; state.hue_shift=0.0
            state.strobe_rate=0.0; state.saturation=1.0
            state.sustain=False; state.blur=0.0

def is_rtpmidi(data):
    return len(data) >= 12 and ((data[0] >> 6) & 0x3) == 2

def parse_rtpmidi(data):
    if len(data) < 13:
        return b''
    payload = data[12:]
    if not payload:
        return b''
    long_hdr = (payload[0] & 0x80) != 0
    return payload[2:] if long_hdr and len(payload) >= 2 else payload[1:]

def midi_server_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((state.midi_host, state.midi_port))
        print(f"[MIDI] Listening on UDP {state.midi_host}:{state.midi_port}")
    except OSError as e:
        print(f"[MIDI] Bind failed: {e}")
        return
    sock.settimeout(2.0)
    while True:
        try:
            data, _ = sock.recvfrom(UDP_BUFSIZE)
            midi_bytes = parse_rtpmidi(data) if is_rtpmidi(data) else data
            i = 0
            while i < len(midi_bytes):
                b = midi_bytes[i]
                if b & 0x80:
                    status = b & 0xF0
                    if status in (MIDI_NOTE_ON, MIDI_NOTE_OFF, MIDI_CC, MIDI_PITCHBEND):
                        handle_midi_message(midi_bytes[i:i+3])
                        i += 3
                    else:
                        i += 1
                else:
                    i += 1
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[MIDI] Error: {e}")

# ── attract mode ───────────────────────────────────────────────────────────────

def attract_mode_thread():
    while True:
        time.sleep(state.attract_interval)
        if state.attract_mode and time.time() - last_midi_time[0] > 10.0:
            spawn_layer(random.randint(36, 84), random.randint(50, 110))

# ── display loop ───────────────────────────────────────────────────────────────

def sim_display(frame):
    chars = " ░▒▓█"
    rows = []
    for y in range(0, HEIGHT, 2):
        row = ""
        for x in range(WIDTH):
            r,g,b = frame[y][x]
            lum = (r+g+b)//3
            row += chars[lum*(len(chars)-1)//255]
        rows.append(row)
    print("\033[H" + "\n".join(rows), end="", flush=True)

def display_loop():
    if HAT_AVAILABLE:
        hat.rotation(state.rotation)
        hat.brightness(state.brightness)
    else:
        print("\033[2J")

    cfg_tick = [0]

    while True:
        t0 = time.time()
        state.time += 1.0/state.fps

        # reload config every ~60 frames
        cfg_tick[0] += 1
        if cfg_tick[0] >= 60:
            cfg_tick[0] = 0
            reload_config()

        update_layers()
        frame = composite()

        if HAT_AVAILABLE:
            hat.brightness(state.brightness)
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    r,g,b = frame[y][x]
                    hat.set_pixel(x, y, r, g, b)
            hat.show()
        else:
            sim_display(frame)

        elapsed = time.time() - t0
        frame_time = 1.0 / max(state.fps, 1)
        sleep_t = max(0, frame_time - elapsed)
        time.sleep(sleep_t)

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    # load initial config
    reload_config()

    print("╔══════════════════════════════════════════╗")
    print("║  Ubercorn MIDI Light Show                ║")
    print(f"║  Grid: {WIDTH}×{HEIGHT}  FPS: {state.fps}  Port: {state.midi_port}        ║")
    print("╚══════════════════════════════════════════╝")

    threads = [
        threading.Thread(target=midi_server_thread,  daemon=True),
        threading.Thread(target=attract_mode_thread, daemon=True),
    ]
    for t in threads:
        t.start()

    try:
        display_loop()   # blocking – runs on main thread for HAT safety
    except KeyboardInterrupt:
        print("\n[EXIT] Shutting down.")
        if HAT_AVAILABLE:
            hat.off()

if __name__ == "__main__":
    main()
