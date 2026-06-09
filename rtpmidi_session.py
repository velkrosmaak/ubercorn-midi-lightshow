"""
rtpmidi_session.py
==================
Full Apple MIDI (RTP-MIDI) session handler.

Implements the Apple MIDI session protocol so that macOS sees the Pi as a
Network MIDI device in Audio MIDI Setup, and any DAW / app can connect to it
natively without any bridging software on the Mac.

Protocol overview
-----------------
Apple MIDI uses two adjacent UDP ports:
  - Control port  (base_port,     default 5004)  – session negotiation
  - Data port     (base_port + 1, default 5005)  – RTP-MIDI packets

Session handshake (initiator = Mac, responder = Pi):
  Mac  → Pi   IN packet  (control port)  "I want to connect"
  Pi   → Mac  OK packet  (control port)  "Accepted"
  Mac  → Pi   IN packet  (data port)     "Data channel ready"
  Pi   → Mac  OK packet  (data port)     "Data channel accepted"

Clock sync (every ~10 s):
  Mac  → Pi   CK0  (timestamp = 0)
  Pi   → Mac  CK1  (echo ts1, add our ts2)
  Mac  → Pi   CK2  (echo ts1+ts2, add ts3)

Termination:
  Either side → BY packet on control port

MIDI data arrives in RTP packets on the data port.  We strip the 12-byte
RTP header + the RTP-MIDI command section header, then parse raw MIDI bytes.

Usage
-----
  from rtpmidi_session import RTPMidiServer

  def on_midi(msg_bytes):
      print(msg_bytes.hex())

  server = RTPMidiServer(name="Ubercorn", port=5004, midi_callback=on_midi)
  server.start()   # launches threads, returns immediately
"""

import socket
import struct
import threading
import time
import random

# ── Apple MIDI magic bytes ─────────────────────────────────────────────────────
APPLEMIDI_SIGNATURE = 0xFFFF
CMD_IN  = 0x494E   # 'IN'
CMD_OK  = 0x4F4B   # 'OK'
CMD_NO  = 0x4E4F   # 'NO'
CMD_BY  = 0x4259   # 'BY'
CMD_CK  = 0x434B   # 'CK'
CMD_RS  = 0x5253   # 'RS'

APPLEMIDI_VERSION = 2

# ── helpers ────────────────────────────────────────────────────────────────────

def _ts100ns() -> int:
    """Current time as 100-microsecond units (Apple MIDI clock)."""
    return int(time.time() * 10000)

def _pack_session(cmd: int, ssrc: int, token: int, version: int = APPLEMIDI_VERSION) -> bytes:
    """Pack an Apple MIDI session command (IN / OK / BY etc.) without name."""
    return struct.pack("!HHIIII",
                       APPLEMIDI_SIGNATURE, cmd,
                       version, token, 0, ssrc)

def _pack_session_named(cmd: int, ssrc: int, token: int, name: str) -> bytes:
    """Pack an Apple MIDI session command with a name field (used in IN/OK)."""
    name_bytes = name.encode("utf-8") + b"\x00"
    return struct.pack("!HHIII",
                       APPLEMIDI_SIGNATURE, cmd,
                       APPLEMIDI_VERSION, token, ssrc) + name_bytes

def _pack_ck(ssrc: int, count: int, ts1: int, ts2: int, ts3: int) -> bytes:
    return struct.pack("!HHIBxxxQQQ",
                       APPLEMIDI_SIGNATURE, CMD_CK,
                       ssrc, count, ts1, ts2, ts3)


class Peer:
    """Represents a connected Apple MIDI peer (e.g. a Mac running a DAW)."""
    def __init__(self, addr, ssrc, name, token):
        self.addr    = addr    # (ip, port) of their CONTROL port
        self.ssrc    = ssrc
        self.name    = name
        self.token   = token
        self.data_addr = None  # filled once data-port IN arrives
        self.connected = False
        self.last_ck   = 0.0


class RTPMidiServer:
    """
    Apple MIDI / RTP-MIDI server.

    Parameters
    ----------
    name          : str   Name shown in macOS Audio MIDI Setup
    port          : int   Control port (data port = port + 1)
    midi_callback : callable(bytes)  Called with raw MIDI message bytes
    """

    def __init__(self, name: str = "Ubercorn", port: int = 5004,
                 midi_callback=None):
        self.name          = name
        self.ctrl_port     = port
        self.data_port     = port + 1
        self.midi_callback = midi_callback or (lambda b: None)
        self.ssrc          = random.randint(1, 0xFFFFFFFF)
        self.peers: dict   = {}   # ssrc → Peer
        self._seq          = random.randint(0, 0xFFFF)
        self._lock         = threading.Lock()

        self._ctrl_sock = None
        self._data_sock = None

    # ── public ─────────────────────────────────────────────────────────────────

    def start(self):
        """Bind sockets and launch listener + clock threads."""
        self._ctrl_sock = self._make_sock(self.ctrl_port)
        self._data_sock = self._make_sock(self.data_port)

        print(f"[RTP-MIDI] '{self.name}'  control={self.ctrl_port}  data={self.data_port}  ssrc=0x{self.ssrc:08X}")

        threading.Thread(target=self._ctrl_loop, daemon=True).start()
        threading.Thread(target=self._data_loop, daemon=True).start()
        threading.Thread(target=self._clock_loop, daemon=True).start()

    def stop(self):
        """Send BY to all peers and close sockets."""
        with self._lock:
            for peer in self.peers.values():
                if peer.connected:
                    self._send_by(peer)
        if self._ctrl_sock:
            self._ctrl_sock.close()
        if self._data_sock:
            self._data_sock.close()

    # ── socket factory ─────────────────────────────────────────────────────────

    @staticmethod
    def _make_sock(port: int) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", port))
        s.settimeout(1.0)
        return s

    # ── control port loop ──────────────────────────────────────────────────────

    def _ctrl_loop(self):
        while True:
            try:
                data, addr = self._ctrl_sock.recvfrom(4096)
                self._handle_ctrl(data, addr)
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle_ctrl(self, data: bytes, addr):
        if len(data) < 4:
            return
        sig, cmd = struct.unpack_from("!HH", data, 0)
        if sig != APPLEMIDI_SIGNATURE:
            return

        if cmd == CMD_IN:
            self._handle_in(data, addr, self._ctrl_sock, is_ctrl=True)
        elif cmd == CMD_BY:
            self._handle_by(data, addr)
        elif cmd == CMD_CK:
            self._handle_ck(data, addr, self._ctrl_sock)

    # ── data port loop ─────────────────────────────────────────────────────────

    def _data_loop(self):
        while True:
            try:
                data, addr = self._data_sock.recvfrom(4096)
                self._handle_data(data, addr)
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle_data(self, data: bytes, addr):
        if len(data) < 4:
            return
        sig, cmd = struct.unpack_from("!HH", data, 0)

        if sig == APPLEMIDI_SIGNATURE:
            if cmd == CMD_IN:
                self._handle_in(data, addr, self._data_sock, is_ctrl=False)
            elif cmd == CMD_CK:
                self._handle_ck(data, addr, self._data_sock)
            elif cmd == CMD_BY:
                self._handle_by(data, addr)
            return

        # RTP packet
        if len(data) < 12:
            return
        self._handle_rtp(data, addr)

    # ── session negotiation ────────────────────────────────────────────────────

    def _handle_in(self, data: bytes, addr, sock: socket.socket, is_ctrl: bool):
        """Handle an IN (invitation) packet."""
        if len(data) < 16:
            return
        try:
            _, _, version, token, ssrc = struct.unpack_from("!HHIII", data, 0)
        except struct.error:
            return

        # extract optional name
        name_start = 16
        name = data[name_start:].rstrip(b"\x00").decode("utf-8", errors="replace") \
               if len(data) > name_start else "Unknown"

        with self._lock:
            if is_ctrl:
                peer = Peer(addr=addr, ssrc=ssrc, name=name, token=token)
                self.peers[ssrc] = peer
                print(f"[RTP-MIDI] IN from '{name}' ({addr[0]})  ssrc=0x{ssrc:08X}")
            else:
                peer = self.peers.get(ssrc)
                if peer is None:
                    # unsolicited data-port IN – create peer anyway
                    peer = Peer(addr=addr, ssrc=ssrc, name=name, token=token)
                    self.peers[ssrc] = peer
                peer.data_addr = addr
                peer.connected = True
                print(f"[RTP-MIDI] '{peer.name}' data channel ready  {addr}")

        # send OK
        ok = _pack_session_named(CMD_OK, self.ssrc, token, self.name)
        sock.sendto(ok, addr)

    def _handle_by(self, data: bytes, addr):
        if len(data) < 16:
            return
        try:
            _, _, _, _, ssrc = struct.unpack_from("!HHIII", data, 0)
        except struct.error:
            return
        with self._lock:
            peer = self.peers.pop(ssrc, None)
        if peer:
            print(f"[RTP-MIDI] '{peer.name}' disconnected (BY)")

    # ── clock sync ─────────────────────────────────────────────────────────────

    def _handle_ck(self, data: bytes, addr, sock: socket.socket):
        """Handle a CK (clock) packet – respond to CK0 with CK1."""
        if len(data) < 36:
            return
        try:
            _, _, ssrc, count, ts1, ts2, ts3 = struct.unpack_from("!HHIBxxxQQQ", data, 0)
        except struct.error:
            return

        if count == 0:
            # CK0 → send CK1
            now = _ts100ns()
            pkt = _pack_ck(self.ssrc, 1, ts1, now, 0)
            sock.sendto(pkt, addr)
        # CK2 is the final step – no response needed

    def _clock_loop(self):
        """Periodically send CK0 to all connected peers to maintain sync."""
        while True:
            time.sleep(10)
            with self._lock:
                peers = list(self.peers.values())
            for peer in peers:
                if peer.connected and peer.data_addr:
                    try:
                        now = _ts100ns()
                        pkt = _pack_ck(self.ssrc, 0, now, 0, 0)
                        self._data_sock.sendto(pkt, peer.data_addr)
                        peer.last_ck = time.time()
                    except Exception:
                        pass

    # ── send BY ────────────────────────────────────────────────────────────────

    def _send_by(self, peer: Peer):
        pkt = struct.pack("!HHIII",
                          APPLEMIDI_SIGNATURE, CMD_BY,
                          APPLEMIDI_VERSION, peer.token, self.ssrc)
        self._ctrl_sock.sendto(pkt, peer.addr)

    # ── RTP-MIDI parsing ───────────────────────────────────────────────────────

    def _handle_rtp(self, data: bytes, addr):
        """Parse an RTP packet and extract MIDI messages."""
        # RTP header: V(2) P(1) X(1) CC(4) M(1) PT(7) seq(16) ts(32) ssrc(32)
        if len(data) < 12:
            return
        pt = data[1] & 0x7F
        if pt != 97:   # RTP-MIDI payload type
            return

        payload = data[12:]
        if not payload:
            return

        # RTP-MIDI command section header
        b0 = payload[0]
        long_header = (b0 & 0x80) != 0
        if long_header:
            if len(payload) < 2:
                return
            length = ((b0 & 0x0F) << 8) | payload[1]
            midi_data = payload[2:2+length]
        else:
            length = b0 & 0x0F
            midi_data = payload[1:1+length]

        # parse individual MIDI messages (running status not implemented – most
        # RTP-MIDI senders include full status bytes)
        i = 0
        while i < len(midi_data):
            b = midi_data[i]
            if b & 0x80:
                status = b & 0xF0
                if status in (0x80, 0x90, 0xA0, 0xB0, 0xE0):   # 3-byte
                    msg = midi_data[i:i+3]
                    if len(msg) == 3:
                        self.midi_callback(bytes(msg))
                    i += 3
                elif status in (0xC0, 0xD0):                      # 2-byte
                    msg = midi_data[i:i+2]
                    if len(msg) == 2:
                        self.midi_callback(bytes(msg))
                    i += 2
                elif b == 0xF0:                                    # sysex
                    end = midi_data.find(0xF7, i)
                    if end == -1:
                        break
                    i = end + 1
                else:
                    i += 1
            else:
                i += 1