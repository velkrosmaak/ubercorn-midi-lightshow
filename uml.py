#!/usr/bin/env python3
"""
ubercorn_midi_show.py
=====================
MIDI-over-WiFi light show for Raspberry Pi Zero + Pimoroni Ubercorn (Unicorn HAT HD 16x16).

Listens for MIDI messages on a UDP port (RTP-MIDI / raw MIDI bytes) and drives
a dynamic, colourful light show that responds to:
  - Note On  → triggers shape explosions, strobes, sweeps
  - Note Off → fades / dissolves the corresponding layer
  - CC msgs  → controls global parameters (brightness, speed, hue shift, strobe rate…)

Dependencies (install with pip3):
    python-rtmidi  → pip3 install python-rtmidi
    unicornhathd   → pip3 install unicornhathd   (or install from Pimoroni)
    mido           → pip3 install mido

Usage:
    python3 ubercorn_midi_show.py [--port 5004] [--brightness 0.8]

MIDI routing options:
  • Use a DAW / MIDI app that supports RTP-MIDI / network MIDI and point it at
    the Pi's IP on the chosen UDP port.
  • Or use `rtpmidi` / `TouchOSC Bridge` / `Network MIDI` (macOS built-in) and
    forward to this port.
  • A helper RTP-MIDI wrapper is included below so raw UDP byte streams also work.

CC Map (channel-agnostic):
  CC 1  → Master brightness  (0-127 → 0.1-1.0)
  CC 7  → Animation speed    (0-127 → 0.2x-4x)
  CC 10 → Global hue shift   (0-127 → 0-360°)
  CC 16 → Strobe rate        (0-127 → 0 off → 30 Hz max)
  CC 17 → Particle density   (0-127)
  CC 18 → Colour saturation  (0-127 → grey-full)
  CC 64 → Sustain pedal      (≥64 = freeze decay, let colours bloom)
  CC 74 → Blur / glow amount (0-127)
  CC 121→ Reset all controllers
"""

import argparse
import colorsys
import math
import random
import socket
import struct
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ── optional hardware import (graceful fallback for dev on non-Pi) ─────────────
try:
    import unicornhathd as hat
    HAT_AVAILABLE = True
    WIDTH, HEIGHT = hat.get_shape()
except ImportError:
    HAT_AVAILABLE = False
    WIDTH, HEIGHT = 16, 16
    print("[WARN] unicornhathd not found – running in console/sim mode.")

# ── constants ──────────────────────────────────────────────────────────────────
FPS            = 60
FRAME_TIME     = 1.0 / FPS
UDP_BUFSIZE    = 1024
MIDI_NOTE_OFF  = 0x80
MIDI_NOTE_ON   = 0x90
MIDI_CC        = 0xB0
MIDI_PITCHBEND = 0xE0
MIDI_CLOCK     = 0xF8

# colour palette banks (hue ranges per bank, chosen by note velocity zone)
PALETTE_BANKS = [
    (0.0,   0.08),   # reds / oranges
    (0.08,  0.17),   # yellows / ambers
    (0.28,  0.42),   # greens / teals
    (0.55,  0.70),   # blues / indigos
    (0.70,  0.85),   # purples / magentas
    (0.85,  1.00),   # pinks / crimsons
]

# ── global state ───────────────────────────────────────────────────────────────
canvas     = [[(0, 0, 0)] * WIDTH for _ in range(HEIGHT)]
lock       = threading.Lock()

class State:
    brightness    = 0.8
    speed         = 1.0
    hue_shift     = 0.0
    strobe_rate   = 0.0     # Hz, 0 = off
    particle_den  = 64
    saturation    = 1.0
    sustain       = False
    blur          = 0.0
    time          = 0.0

state = State()

# ── layer system ───────────────────────────────────────────────────────────────
@dataclass
class Layer:
    note:      int
    velocity:  int
    pixels:    list = field(default_factory=lambda: [[(0,0,0)]*WIDTH for _ in range(HEIGHT)])
    alpha:     float = 1.0
    decay:     float = 0.015   # alpha units per frame
    alive:     bool  = True
    effect_fn: object = None
    phase:     float = 0.0
    hue:       float = 0.0
    born:      float = 0.0

layers: List[Layer] = []
layer_lock = threading.Lock()

# ── utility ────────────────────────────────────────────────────────────────────

def hsv_to_rgb(h, s, v) -> Tuple[int,int,int]:
    h = (h + state.hue_shift / 360.0) % 1.0
    s = s * state.saturation
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r*255), int(g*255), int(b*255)

def lerp(a, b, t):
    return a + (b - a) * t

def clamp(v, lo=0, hi=255):
    return max(lo, min(hi, v))

def add_pixel(pix, r, g, b):
    pr, pg, pb = pix
    return clamp(pr+r), clamp(pg+g), clamp(pb+b)

def note_to_hue(note: int) -> float:
    """Map MIDI note 0-127 to a hue, cycling through spectrum."""
    return (note / 127.0) % 1.0

def velocity_to_value(velocity: int) -> float:
    return 0.3 + 0.7 * (velocity / 127.0)

def pick_palette(velocity: int) -> Tuple[float, float]:
    bank_idx = (velocity // 22) % len(PALETTE_BANKS)
    return PALETTE_BANKS[bank_idx]

# ── effect functions ───────────────────────────────────────────────────────────
# Each returns a rendered pixel grid (list of list of (r,g,b)) for the layer.

def effect_radial_burst(layer: Layer):
    """Expanding ring from centre, coloured by note hue."""
    pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    t   = layer.phase
    cx, cy = WIDTH/2 - 0.5, HEIGHT/2 - 0.5
    radius   = t * (WIDTH * 0.8)
    thickness = 2.5 + 1.5 * math.sin(t * 3)
    v = velocity_to_value(layer.velocity)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            d = math.sqrt((x-cx)**2 + (y-cy)**2)
            dist = abs(d - radius)
            if dist < thickness:
                brightness = (1.0 - dist/thickness) * layer.alpha
                r,g,b = hsv_to_rgb(layer.hue, 1.0, v * brightness)
                pix[y][x] = (r,g,b)
    return pix

def effect_note_flash(layer: Layer):
    """Full-screen flash that fades quickly – high velocity = white core."""
    pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    v   = velocity_to_value(layer.velocity)
    sat = max(0, 1.0 - layer.alpha * 0.4)   # desaturate at peak
    for y in range(HEIGHT):
        for x in range(WIDTH):
            r,g,b = hsv_to_rgb(layer.hue, sat, v * layer.alpha)
            pix[y][x] = (r,g,b)
    return pix

def effect_diagonal_sweep(layer: Layer):
    """A diagonal colour band sweeping across the grid."""
    pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    t   = layer.phase
    sweep = (t * (WIDTH + HEIGHT) * 1.4) - 4
    v = velocity_to_value(layer.velocity)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            pos = x + y
            dist = abs(pos - sweep)
            if dist < 4:
                brightness = (1.0 - dist/4) * layer.alpha
                hue2 = (layer.hue + dist * 0.03) % 1.0
                r,g,b = hsv_to_rgb(hue2, 1.0, v * brightness)
                pix[y][x] = (r,g,b)
    return pix

def effect_sparkle(layer: Layer):
    """Random sparkling particles seeded by note."""
    pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    rng = random.Random(layer.note + int(layer.phase * 10))
    count = int(state.particle_den / 5) + 5
    v = velocity_to_value(layer.velocity)
    for _ in range(count):
        x = rng.randint(0, WIDTH-1)
        y = rng.randint(0, HEIGHT-1)
        h = (layer.hue + rng.uniform(-0.1, 0.1)) % 1.0
        br = rng.uniform(0.5, 1.0) * layer.alpha * v
        r,g,b = hsv_to_rgb(h, rng.uniform(0.7,1.0), br)
        pix[y][x] = (r,g,b)
    return pix

def effect_spiral(layer: Layer):
    """Rotating spiral arms."""
    pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    cx, cy = WIDTH/2 - 0.5, HEIGHT/2 - 0.5
    t   = layer.phase * math.pi * 4
    arms = 3
    v = velocity_to_value(layer.velocity)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            dx, dy = x - cx, y - cy
            r = math.sqrt(dx*dx + dy*dy) + 0.001
            angle = math.atan2(dy, dx)
            spiral_phase = (angle + r * 0.6 - t) % (2*math.pi / arms)
            dist = min(spiral_phase, 2*math.pi/arms - spiral_phase)
            if dist < 0.4:
                brightness = (1.0 - dist/0.4) * layer.alpha * (1.0 - r/(WIDTH*0.85))
                h = (layer.hue + r/WIDTH * 0.3) % 1.0
                rr,g,b = hsv_to_rgb(h, 1.0, v * max(0, brightness))
                pix[y][x] = (rr,g,b)
    return pix

def effect_columns(layer: Layer):
    """Vertical colour columns that rise and fade."""
    pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    t = layer.phase
    v = velocity_to_value(layer.velocity)
    for x in range(WIDTH):
        height_frac = 0.5 + 0.5 * math.sin(t * 6 + x * 0.8)
        col_height = int(height_frac * HEIGHT)
        h = (layer.hue + x/WIDTH * 0.3) % 1.0
        for y in range(col_height):
            brightness = (1.0 - y/HEIGHT * 0.5) * layer.alpha * v
            rr,g,b = hsv_to_rgb(h, 1.0, brightness)
            pix[HEIGHT-1-y][x] = (rr,g,b)
    return pix

def effect_plasma(layer: Layer):
    """Classic plasma / sine wave interference."""
    pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    t = layer.phase * 4
    v = velocity_to_value(layer.velocity)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            val = (math.sin(x * 0.5 + t) +
                   math.sin(y * 0.5 + t * 1.3) +
                   math.sin((x+y) * 0.35 + t * 0.7) +
                   math.sin(math.sqrt(x*x+y*y) * 0.4 + t)) / 4
            h = (layer.hue + val * 0.2) % 1.0
            brightness = (0.5 + 0.5*val) * layer.alpha * v
            rr,g,b = hsv_to_rgb(h, 1.0, max(0, brightness))
            pix[y][x] = (rr,g,b)
    return pix

def effect_strobe_burst(layer: Layer):
    """Rapid stroboscopic flash – good for high-velocity notes."""
    pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    # strobe at ~10Hz independent of global strobe
    strobe = int(layer.phase * 20) % 2
    if strobe == 0:
        return pix
    v = velocity_to_value(layer.velocity)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            r,g,b = hsv_to_rgb(layer.hue, 0.6, v * layer.alpha)
            pix[y][x] = (r,g,b)
    return pix

def effect_diamond(layer: Layer):
    """Expanding diamond / rhombus pulse."""
    pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    cx, cy = WIDTH//2, HEIGHT//2
    t = layer.phase
    size = t * max(WIDTH, HEIGHT) * 1.2
    v = velocity_to_value(layer.velocity)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            d = abs(x - cx) + abs(y - cy)
            dist = abs(d - size)
            if dist < 3:
                brightness = (1.0 - dist/3) * layer.alpha
                h = (layer.hue + dist * 0.04) % 1.0
                rr,g,b = hsv_to_rgb(h, 1.0, v * brightness)
                pix[y][x] = (rr,g,b)
    return pix

EFFECTS = [
    effect_radial_burst,
    effect_note_flash,
    effect_diagonal_sweep,
    effect_sparkle,
    effect_spiral,
    effect_columns,
    effect_plasma,
    effect_strobe_burst,
    effect_diamond,
]

# ── background ambient animation ───────────────────────────────────────────────

class Background:
    def __init__(self):
        self.t = 0.0

    def render(self) -> list:
        pix = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
        t = self.t
        for y in range(HEIGHT):
            for x in range(WIDTH):
                # slow drifting aurora
                v = (math.sin(x*0.4 + t*0.3) * math.sin(y*0.3 + t*0.2)) * 0.12
                if v > 0:
                    h = (t * 0.04 + x/WIDTH * 0.15 + y/HEIGHT * 0.1) % 1.0
                    r,g,b = hsv_to_rgb(h, 0.8, v)
                    pix[y][x] = (r,g,b)
        self.t += FRAME_TIME * state.speed
        return pix

background = Background()

# ── compositing ────────────────────────────────────────────────────────────────

def blur_pass(pix, amount):
    """Simple 3×3 box blur, weighted by amount (0-1)."""
    if amount < 0.01:
        return pix
    out = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]
    for y in range(HEIGHT):
        for x in range(WIDTH):
            rs = gs = bs = 0
            cnt = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx, ny = (x+dx) % WIDTH, (y+dy) % HEIGHT
                    r,g,b = pix[ny][nx]
                    rs+=r; gs+=g; bs+=b; cnt+=1
            ar, ag, ab = rs//cnt, gs//cnt, bs//cnt
            or_, og, ob = pix[y][x]
            out[y][x] = (
                int(lerp(or_, ar, amount)),
                int(lerp(og, ag, amount)),
                int(lerp(ob, ab, amount)),
            )
    return out

def composite():
    """Merge background + all layers → canvas."""
    frame = background.render()

    with layer_lock:
        active = [l for l in layers if l.alive]

    for layer in active:
        lpix = layer.effect_fn(layer)
        for y in range(HEIGHT):
            for x in range(WIDTH):
                lr, lg, lb = lpix[y][x]
                fr, fg, fb = frame[y][x]
                frame[y][x] = (
                    clamp(fr + lr),
                    clamp(fg + lg),
                    clamp(fb + lb),
                )

    # global blur / glow
    frame = blur_pass(frame, state.blur)

    # global strobe
    if state.strobe_rate > 0:
        period = 1.0 / state.strobe_rate
        if int(state.time / (period/2)) % 2 == 1:
            frame = [[(0,0,0)]*WIDTH for _ in range(HEIGHT)]

    return frame

# ── layer lifecycle ────────────────────────────────────────────────────────────

def assign_effect(velocity: int, note: int):
    """Choose effect function based on velocity and note characteristics."""
    if velocity > 110:
        return random.choice([effect_strobe_burst, effect_note_flash, effect_radial_burst])
    if velocity > 85:
        return random.choice([effect_plasma, effect_spiral, effect_diamond])
    if velocity > 60:
        return random.choice([effect_radial_burst, effect_diagonal_sweep, effect_columns])
    return random.choice([effect_sparkle, effect_columns, effect_diagonal_sweep])

def spawn_layer(note: int, velocity: int):
    with layer_lock:
        # remove any existing layer for this note
        for l in layers:
            if l.note == note:
                l.decay = 0.04
        lo, hi = pick_palette(velocity)
        hue = random.uniform(lo, hi)
        decay_rate = lerp(0.005, 0.025, velocity / 127.0)
        layer = Layer(
            note      = note,
            velocity  = velocity,
            hue       = hue,
            decay     = decay_rate,
            alive     = True,
            effect_fn = assign_effect(velocity, note),
            born      = state.time,
        )
        layers.append(layer)
        # cap total layers
        if len(layers) > 12:
            layers.pop(0)

def release_layer(note: int):
    with layer_lock:
        for l in layers:
            if l.note == note and l.alive:
                if not state.sustain:
                    l.decay = 0.02   # accelerate fade on note-off

def update_layers():
    dt = FRAME_TIME * state.speed
    with layer_lock:
        for l in layers:
            if not l.alive:
                continue
            l.phase += dt
            if not state.sustain:
                l.alpha -= l.decay
            else:
                l.alpha = min(l.alpha + 0.01, 1.0)   # sustain = bloom
            if l.alpha <= 0:
                l.alive = False
        # prune dead layers periodically
        layers[:] = [l for l in layers if l.alive or l.phase < 0.5]

# ── MIDI handling ──────────────────────────────────────────────────────────────

def handle_midi_message(msg_bytes: bytes):
    if len(msg_bytes) < 2:
        return
    status   = msg_bytes[0] & 0xF0
    # channel = msg_bytes[0] & 0x0F  # channel-agnostic for now
    data1    = msg_bytes[1] if len(msg_bytes) > 1 else 0
    data2    = msg_bytes[2] if len(msg_bytes) > 2 else 0

    if status == MIDI_NOTE_ON and data2 > 0:
        spawn_layer(data1, data2)

    elif status == MIDI_NOTE_OFF or (status == MIDI_NOTE_ON and data2 == 0):
        release_layer(data1)

    elif status == MIDI_CC:
        cc, val = data1, data2
        if cc == 1:   state.brightness   = 0.1 + 0.9 * (val / 127.0)
        elif cc == 7: state.speed        = 0.2 + 3.8 * (val / 127.0)
        elif cc == 10:state.hue_shift    = val / 127.0 * 360.0
        elif cc == 16:state.strobe_rate  = (val / 127.0) * 30.0
        elif cc == 17:state.particle_den = val
        elif cc == 18:state.saturation   = val / 127.0
        elif cc == 64:state.sustain      = (val >= 64)
        elif cc == 74:state.blur         = val / 127.0
        elif cc == 121:
            # reset all controllers
            state.brightness  = 0.8
            state.speed       = 1.0
            state.hue_shift   = 0.0
            state.strobe_rate = 0.0
            state.saturation  = 1.0
            state.sustain     = False
            state.blur        = 0.0

# ── UDP MIDI server ────────────────────────────────────────────────────────────
# Accepts both raw MIDI byte streams and RTP-MIDI packets (strips 12-byte RTP header).

def is_rtpmidi(data: bytes) -> bool:
    """Heuristic: RTP-MIDI has specific header bits."""
    if len(data) < 12:
        return False
    version = (data[0] >> 6) & 0x3
    return version == 2

def parse_rtpmidi(data: bytes) -> bytes:
    """Strip 12-byte RTP header + RTP-MIDI command section header."""
    if len(data) < 13:
        return b''
    payload = data[12:]
    # RTP-MIDI command section: first byte is length flags
    if not payload:
        return b''
    b0 = payload[0]
    long_header = (b0 & 0x80) != 0
    if long_header and len(payload) >= 2:
        midi_data = payload[2:]
    else:
        midi_data = payload[1:]
    return midi_data

def midi_server_thread(host: str, port: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    print(f"[MIDI] Listening on UDP {host}:{port}")
    while True:
        try:
            data, addr = sock.recvfrom(UDP_BUFSIZE)
            if is_rtpmidi(data):
                midi_bytes = parse_rtpmidi(data)
            else:
                midi_bytes = data

            # parse one or more MIDI messages from the byte stream
            i = 0
            while i < len(midi_bytes):
                b = midi_bytes[i]
                if b & 0x80:   # status byte
                    status = b & 0xF0
                    if status in (MIDI_NOTE_ON, MIDI_NOTE_OFF, MIDI_CC, MIDI_PITCHBEND):
                        msg = midi_bytes[i:i+3]
                        handle_midi_message(msg)
                        i += 3
                    elif b == MIDI_CLOCK:
                        handle_midi_message(bytes([b]))
                        i += 1
                    else:
                        i += 1
                else:
                    i += 1
        except Exception as e:
            print(f"[MIDI] Error: {e}")

# ── display thread ─────────────────────────────────────────────────────────────

def sim_display(frame):
    """Fallback: print a rough ASCII colour map to terminal."""
    chars = " ░▒▓█"
    rows = []
    for y in range(0, HEIGHT, 2):
        row = ""
        for x in range(WIDTH):
            r,g,b = frame[y][x]
            lum = (r+g+b) // 3
            row += chars[lum * (len(chars)-1) // 255]
        rows.append(row)
    print("\033[H" + "\n".join(rows), end="", flush=True)

def display_thread_fn():
    if HAT_AVAILABLE:
        hat.rotation(0)
        hat.brightness(state.brightness)
    else:
        print("\033[2J")  # clear screen for sim

    while True:
        t0 = time.time()
        state.time += FRAME_TIME

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
        sleep_t = max(0, FRAME_TIME - elapsed)
        time.sleep(sleep_t)

# ── idle animation: attract mode ───────────────────────────────────────────────
# Fires synthetic notes periodically when no real MIDI arrives,
# so the display stays alive even when idle.

last_midi_time = [time.time()]

def attract_mode_thread():
    """Periodically spawn random layers when MIDI has been silent for >10s."""
    while True:
        time.sleep(4.0)
        if time.time() - last_midi_time[0] > 10.0:
            note = random.randint(36, 84)
            vel  = random.randint(50, 110)
            spawn_layer(note, vel)

# wrap handle_midi_message to track last MIDI time
_orig_handle = handle_midi_message
def handle_midi_message(msg_bytes: bytes):
    last_midi_time[0] = time.time()
    if not msg_bytes:
        return
    if msg_bytes[0] == MIDI_CLOCK:
        # Pass clock through to the original handler if it supports it
        pass 
    _orig_handle(msg_bytes)

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ubercorn MIDI light show")
    parser.add_argument("--host",       default="0.0.0.0",  help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port",       default=5004, type=int, help="UDP MIDI port (default: 5004)")
    parser.add_argument("--brightness", default=0.8,  type=float, help="Initial brightness 0.0-1.0")
    args = parser.parse_args()

    state.brightness = args.brightness

    print("╔══════════════════════════════════════════╗")
    print("║  Ubercorn MIDI Light Show                ║")
    print(f"║  Grid: {WIDTH}×{HEIGHT}  FPS: {FPS}  Port: {args.port}          ║")
    print("╠══════════════════════════════════════════╣")
    print("║  CC Map:                                 ║")
    print("║   1  → Brightness    7  → Speed          ║")
    print("║   10 → Hue Shift    16  → Strobe Rate    ║")
    print("║   17 → Particles    18  → Saturation     ║")
    print("║   64 → Sustain      74  → Blur/Glow      ║")
    print("║   121→ Reset All                         ║")
    print("╚══════════════════════════════════════════╝")

    # start threads
    threads = [
        threading.Thread(target=midi_server_thread, args=(args.host, args.port), daemon=True),
        threading.Thread(target=display_thread_fn,  daemon=True),
        threading.Thread(target=attract_mode_thread, daemon=True),
    ]
    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[EXIT] Shutting down.")
        if HAT_AVAILABLE:
            hat.off()

if __name__ == "__main__":
    main()