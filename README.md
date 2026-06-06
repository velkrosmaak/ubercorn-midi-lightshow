# Ubercorn MIDI Light Show
## Requirements

```
flask>=2.3
```

Install everything:
```bash
pip3 install flask unicornhathd
```

---

## File Layout

```
ubercorn/
├── config.py               ← shared config contract (both scripts import this)
├── ubercorn_config.json    ← auto-created on first run
├── ubercorn_midi_show.py   ← light show engine
├── webapp.py               ← Flask web control UI
├── static/
│   └── index.html          ← single-page UI
├── ubercorn-show.service   ← systemd unit for light show
└── ubercorn-web.service    ← systemd unit for web UI
```

---

## Quick Start

### Terminal A – light show
```bash
python3 ubercorn_midi_show.py
```

### Terminal B – web UI
```bash
python3 webapp.py --port 8080
```

Then open `http://<pi-ip>:8080` in a browser.

---

## Run as systemd services (auto-start on boot)

```bash
# copy service files
sudo cp ubercorn-show.service /etc/systemd/system/
sudo cp ubercorn-web.service  /etc/systemd/system/

# enable & start
sudo systemctl daemon-reload
sudo systemctl enable ubercorn-show ubercorn-web
sudo systemctl start  ubercorn-show ubercorn-web

# check status
sudo systemctl status ubercorn-show
sudo systemctl status ubercorn-web
```

---

## How it works

```
ubercorn_config.json
       ▲  writes          reads every ~1 s
       │                        │
  webapp.py  ◄── HTTP ──  Browser
  (Flask)                 (port 8080)
                                
  ubercorn_midi_show.py   ← UDP MIDI (port 5004)
  (HAT render loop)
```

- The web UI writes to `ubercorn_config.json`.
- The main script polls the config file every ~60 frames and hot-reloads all parameters.
- The **Restart** button writes a `"restart": true` flag; the main script detects it and calls `os.execv()` to re-exec itself cleanly.
- The web UI can also **start/stop the show process directly** via subprocess (useful when running both under the same user account).

---

## MIDI Routing

| Platform | Method |
|---|---|
| macOS | Audio MIDI Setup → Network Session → connect to Pi IP:5004 |
| iOS/Android | Any RTP-MIDI app (MIDI Network, MIDI Tools, etc.) |
| Linux | `sendmidi udp:<pi-ip>:5004 ...` or `rtpmidi` daemon |
| DAW | UDP MIDI bridge plugin (e.g. loopMIDI + rtpMIDI on Windows) |

Both raw UDP MIDI byte streams **and** RTP-MIDI packets are accepted.

---

## CC Map

| CC | Parameter | Range |
|---|---|---|
| 1 | Brightness | 0.1–1.0 |
| 7 | Speed | 0.2×–4.0× |
| 10 | Hue Shift | 0–360° |
| 16 | Strobe Rate | 0–30 Hz |
| 17 | Particle Density | 0–127 |
| 18 | Saturation | 0–1.0 |
| 64 | Sustain / Bloom | ≥64 = on |
| 74 | Blur / Glow | 0–1.0 |
| 121 | Reset All | any |
