#!/usr/bin/env python3
"""xq_turrell_room_2d_v7_turrell.py — ALMA OS Authentic Turrell Room

Immersive light field visuals inspired by James Turrell exhibits.
Pure perceptual depth through soft gradients, infinite horizons, and boundless color washes.
No hard shapes or circles—vast, uniform Ganzfeld-like spaces with subtle shifts.

Fully compatible with TurrellRunner arguments.
Restores HUD with hotkeys and HCE-driven transcendent immersion.

Key Features:
- Turrell authenticity: Full-screen soft-edged color fields, horizon illusions, infinite depth.
- HCE-driven magnificence: Higher HCE → deeper saturation, color harmony, perceptual expansion (e.g., golden light bleed on transcendent peaks).
- Modes: OFF (subtle ambient), MAX_FOCUS (crisp clarity), MAX_Q (rich saturation), STABILIZE (balanced calm).
- Full HUD with hotkeys (toggle H, default ON if --hud).
- Reliability LIVE/STALE dot.

Hotkeys:
  0 = OFF          1 = MAX_FOCUS
  2 = MAX_Q        3 = STABILIZE
  F = Fullscreen   H = Toggle HUD
  V = Cycle Q source   M = Cycle Q metric
  ESC = Quit
"""

import argparse
import json
import os
import time
import math
import pygame
from pathlib import Path

try:
    import pygame.freetype as ft  # type: ignore
except Exception:
    ft = None

# ------------------------------
# Argument Parsing (Runner Compatibility)
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--ndjson", default="sessions/current/state_stream.ndjson")
parser.add_argument("--mode-file", default="sessions/current/adaptive_mode.json")
parser.add_argument("--no-freeze-on-invalid", action="store_true")
parser.add_argument("--hud", action="store_true", help="Start with HUD visible")
parser.add_argument("--quality", default="standard", choices=["standard", "4k"])
parser.add_argument("--display", type=int, default=0)
parser.add_argument("--q-metric", default="vibe_focus")
parser.add_argument("--fullscreen", action="store_true")
args = parser.parse_args()

# Resolution
BASE_SW, BASE_SH = 1920, 1080
SCALE = 2 if args.quality == "4k" else 1
SW, SH = BASE_SW * SCALE, BASE_SH * SCALE

# Turrell-inspired colors (soft, immersive)
COLORS = {
    "OFF": ((20, 10, 30), (40, 20, 60)),           # Deep indigo twilight
    "MAX_FOCUS": ((80, 100, 140), (120, 140, 180)),  # Crisp blue clarity
    "MAX_Q": ((140, 60, 180), (200, 100, 240)),     # Rich magenta saturation
    "STABILIZE": ((60, 80, 60), (100, 120, 100)),   # Calm green balance
}
GOLD_TRANSCEND = (255, 200, 100)

# ------------------------------
# NDJSON Follower
# ------------------------------
class StateFollower:
    def __init__(self, path):
        self.path = Path(path)
        self.file = None
        self.pos = 0
        self.state = None
        self.last_ts = 0

    def _open(self):
        if self.file:
            return
        if not self.path.exists():
            return
        try:
            self.file = open(self.path, "r", encoding="utf-8", errors="ignore")
            self.file.seek(self.pos if self.pos else 0, os.SEEK_SET)
        except:
            self.file = None

    def update(self):
        self._open()
        if not self.file:
            return False
        updated = False
        try:
            cur_size = os.path.getsize(self.path)
            if cur_size < self.pos:
                # truncated/rotated
                self.pos = 0
                self.file.seek(0, os.SEEK_SET)
            self.file.seek(self.pos, os.SEEK_SET)
            for line in self.file:
                line = line.strip()
                if not line:
                    continue
                try:
                    packet = json.loads(line)
                    self.state = packet
                    self.last_ts = packet.get("ts_unix", time.time())
                    updated = True
                except Exception:
                    continue
            self.pos = self.file.tell() if self.file else 0
        except Exception:
            try:
                if self.file:
                    self.file.close()
            finally:
                self.file = None
                self.pos = 0
        return updated

# ------------------------------
# Text helpers (robust fallback)
# ------------------------------
CHAR_PIXELS = {
    "A": ["0110", "1001", "1111", "1001", "1001"],
    "C": ["0111", "1000", "1000", "1000", "0111"],
    "E": ["1111", "1110", "1110", "1000", "1111"],
    "F": ["1111", "1110", "1110", "1000", "1000"],
    "H": ["1001", "1001", "1111", "1001", "1001"],
    "I": ["111", "010", "010", "010", "111"],
    "L": ["1000", "1000", "1000", "1000", "1111"],
    "O": ["0110", "1001", "1001", "1001", "0110"],
    "R": ["1110", "1001", "1110", "1010", "1001"],
    "S": ["0111", "1000", "0110", "0001", "1110"],
    "T": ["111", "010", "010", "010", "010"],
    "U": ["1001", "1001", "1001", "1001", "0110"],
    "V": ["10001", "10001", "10001", "01010", "00100"],
    "X": ["1001", "1001", "0110", "1001", "1001"],
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "010", "010", "010"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
    ":": ["0", "1", "0", "1", "0"],
    ".": ["0", "0", "0", "1", "0"],
    "-": ["0", "0", "111", "0", "0"],
    " ": ["0", "0", "0", "0", "0"],
}


def _pixel_text_surface(text: str, color=(230, 230, 255)) -> pygame.Surface:
    text = text.upper()
    char_w, char_h = 5, 5
    spacing = 2
    width = (char_w + spacing) * len(text) + spacing
    height = 7 + 2 * spacing
    surf = pygame.Surface((max(width, 8), max(height, 8)), pygame.SRCALPHA)
    for idx, ch in enumerate(text):
        glyph = CHAR_PIXELS.get(ch, CHAR_PIXELS[" "])
        x0 = idx * (char_w + spacing) + spacing
        y0 = spacing
        for y, row in enumerate(glyph):
            for x, bit in enumerate(row):
                if bit == "1":
                    surf.set_at((x0 + x, y0 + y), color)
    return surf


def make_font():
    # Primary: pygame.font.SysFont
    try:
        pygame.font.init()
        f = pygame.font.SysFont("consolas, monaco, monospace", 22)
        if f:
            return f, lambda txt, col: f.render(txt, True, col)
    except Exception as exc:
        print(f"[turrell_v7] SysFont unavailable: {exc}", flush=True)

    # Freetype fallback
    if ft:
        try:
            ft.init()
            ff = ft.SysFont("Menlo,Consolas,Monaco,monospace", 22)
            if ff:
                return ff, lambda txt, col: ff.render(txt, col)[0]
        except Exception as exc:
            print(f"[turrell_v7] freetype unavailable: {exc}", flush=True)

    # Pixel fallback
    return None, lambda txt, col: _pixel_text_surface(txt, col)


def draw_text(screen, render_fn, text, color, pos):
    surf = render_fn(text, color)
    screen.blit(surf, pos)

# ------------------------------
# Turrell Visual Core (Ganzfeld Light Fields)
# ------------------------------
def draw_turrell_field(screen, t, mode, hce_norm):
    w, h = screen.get_size()
    base1, base2 = COLORS.get(mode, COLORS["OFF"])

    # Soft vertical gradient (infinite depth)
    for y in range(h):
        blend = y / h
        color = (
            int(base1[0] + (base2[0] - base1[0]) * blend),
            int(base1[1] + (base2[1] - base1[1]) * blend),
            int(base1[2] + (base2[2] - base1[2]) * blend),
        )
        pygame.draw.line(screen, color, (0, y), (w, y))

    # Subtle horizon glow
    horizon_y = h // 2
    glow_height = int(300 + 600 * hce_norm)
    for i in range(glow_height):
        alpha = int(80 * (1 - abs(i) / glow_height))
        dy = horizon_y + i
        if 0 <= dy < h:
            color = (*GOLD_TRANSCEND, alpha) if hce_norm > 0.5 else (base2[0] + 50, base2[1] + 50, base2[2] + 50, alpha)
            pygame.draw.line(screen, color, (0, dy), (w, dy))

    # Transcendent bleed (HCE peaks)
    if hce_norm > 0.6:
        bleed = int(100 * (hce_norm - 0.6) / 0.4)
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((*GOLD_TRANSCEND, bleed))
        screen.blit(overlay, (0, 0))

    # Gentle breathing pulse
    pulse = 0.98 + 0.02 * math.sin(t * 0.5)
    screen.blit(screen, (0, 0), special_flags=pygame.BLEND_MULT)
    # Note: Simplified; actual pulse via surface scaling or alpha modulation

# ------------------------------
# HUD and Helpers
# ------------------------------
def draw_live_dot(screen, alive, valid, qualia, x, y):
    color = (50, 255, 50) if (alive and valid and qualia) else (255, 50, 50)
    pygame.draw.circle(screen, color, (x, y), 14)
    if alive:
        pulse = int(10 + 6 * math.sin(time.time() * 10))
        pygame.draw.circle(screen, (255, 255, 255), (x, y), pulse, width=4)

def draw_hud(screen, render_fn, mode, follower):
    if not follower.state:
        return
    state = follower.state
    rel = state.get("reliability", {})
    raw = state.get("raw", {})
    lines = [
        f"Turrell Room v7 — Mode: {mode.get('name', 'OFF')}",
        f"X: {state.get('X', 0):.3f} | Q: {state.get('Q_vibe_focus', 0):.3f}",
        f"HCE: {raw.get('HCE_raw', 0):.1f}",
        f"Valid: {rel.get('valid', False)} | Qualia: {rel.get('qualia_valid', False)}",
        "",
        "Hotkeys: 0-3 Modes | F Fullscreen | H HUD | ESC Quit",
        "HUD ACTIVE",
    ]
    y = 30
    for line in lines:
        draw_text(screen, render_fn, line, (230, 230, 255), (30, y))
        y += 22

# ------------------------------
# Main
# ------------------------------
def main():
    pygame.init()
    flags = pygame.RESIZABLE
    if args.fullscreen:
        flags |= pygame.FULLSCREEN
    try:
        screen = pygame.display.set_mode((SW, SH), flags, display=args.display)
    except Exception as exc:
        print(f"[turrell_v7] display set_mode failed ({exc}), retrying without display hint", flush=True)
        screen = pygame.display.set_mode((SW, SH), flags)
    pygame.display.set_caption("ALMA OS Turrell Room v7 — Authentic Light Fields")
    clock = pygame.time.Clock()
    font_obj, render_fn = make_font()
    if font_obj is None:
        print("[turrell_v7] HUD using pixel fallback", flush=True)

    follower = StateFollower(args.ndjson)
    mode_path = Path(args.mode_file)
    mode_info = {"mode": "OFF", "name": "OFF"}
    if mode_path.exists():
        try:
            mode_info = json.load(open(mode_path))
        except:
            pass

    show_hud = True
    print("[turrell_v7] HUD forced ON", flush=True)
    toast = "Turrell Room v7 — Authentic James Turrell Light Fields"
    toast_until = time.time() + 5
    esc_confirm_until = 0.0

    running = True
    t = 0
    while running:
        dt = clock.tick(60) / 1000
        t += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                print(f"[turrell_v7] Key pressed: {event.key}", flush=True)
                if event.key == pygame.K_ESCAPE:
                    now = time.time()
                    if now < esc_confirm_until:
                        running = False
                    else:
                        toast = "Press ESC again to quit"
                        toast_until = now + 2.0
                        esc_confirm_until = now + 2.0
                elif event.key == pygame.K_h:
                    show_hud = not show_hud
                elif event.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()
                elif event.key in (pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3):
                    modes = {pygame.K_0: "OFF", pygame.K_1: "MAX_FOCUS", pygame.K_2: "MAX_Q", pygame.K_3: "STABILIZE"}
                    mode_info["mode"] = modes[event.key]
                    mode_info["name"] = modes[event.key]
                    try:
                        with open(mode_path, "w") as f:
                            json.dump(mode_info, f)
                    except Exception as exc:
                        print(f"[turrell_v7] Failed to write mode file: {exc}", flush=True)

        alive = follower.update()
        state = follower.state
        hce_norm = state["raw"].get("HCE_raw", 0) / 60.0 if state and state.get("raw") else 0
        hce_norm = min(hce_norm, 1.0)

        mode = mode_info["mode"]

        draw_turrell_field(screen, t, mode, hce_norm)

        draw_live_dot(screen, alive, state.get("reliability", {}).get("valid", False) if state else False,
                      state.get("reliability", {}).get("qualia_valid", False) if state else False,
                      screen.get_width() - 40, 40)

        if show_hud:
            draw_hud(screen, render_fn, mode_info, follower)

        if toast and time.time() < toast_until:
            draw_text(screen, render_fn, toast, (235, 235, 255), (30, screen.get_height() - 50))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    raise SystemExit(main())