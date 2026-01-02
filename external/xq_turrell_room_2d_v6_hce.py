#!/usr/bin/env python3
"""
xq_turrell_room_2d_v6_hce.py — ALMA OS Turrell Room (Style-matched to v5.3)

Goal: Make v6 visuals use the SAME palette + SAME tonemapped/bloomed look as:
  xq_turrell_room_2d_v5_3_style_modes.py

We keep:
- Runner-compatible args
- Mode-file + hotkeys
- HUD + live/stale dot + ESC double-press
We replace:
- The old line-by-line pygame gradient renderer
With:
- v5.3 palette + numpy tonemap/bloom/dither renderer
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

try:
    import numpy as np  # type: ignore
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False


# ------------------------------
# Args (Runner compatible + optional render knobs)
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--ndjson", default="sessions/current/state_stream.ndjson")
parser.add_argument("--mode-file", default="sessions/current/adaptive_mode.json")
parser.add_argument("--no-freeze-on-invalid", action="store_true")
parser.add_argument("--hud", action="store_true", help="Start with HUD visible")
parser.add_argument("--quality", default="standard", choices=["standard", "4k"])
parser.add_argument("--display", type=int, default=0)
parser.add_argument("--q-metric", default="vibe_focus")  # vibe_focus|vibe|focus|abs|auto
parser.add_argument("--fullscreen", action="store_true")

# Optional style knobs (safe defaults mirror v5.3 “balanced”)
parser.add_argument("--render-scale", type=float, default=None, help="Internal render scale (0.35-0.80). Default: standard=0.55, 4k=0.80")
parser.add_argument("--bloom-down", type=int, default=4)
parser.add_argument("--bloom-blur-passes", type=int, default=3)
parser.add_argument("--exposure", type=float, default=1.70)
parser.add_argument("--bloom-strength", type=float, default=1.35)
parser.add_argument("--bloom-threshold", type=float, default=0.36)
parser.add_argument("--drift-speed", type=float, default=0.002)

# Normalization ranges (match v5.3 defaults; tweak if needed)
parser.add_argument("--x-min", type=float, default=0.85)
parser.add_argument("--x-max", type=float, default=2.10)
parser.add_argument("--q-min", type=float, default=0.00)
parser.add_argument("--q-max", type=float, default=0.18)

args = parser.parse_args()

BASE_SW, BASE_SH = 1920, 1080
SCALE = 2 if args.quality == "4k" else 1
SW, SH = BASE_SW * SCALE, BASE_SH * SCALE

if args.render_scale is None:
    args.render_scale = 0.80 if args.quality == "4k" else 0.55

args.bloom_down = max(2, int(args.bloom_down))
args.bloom_blur_pases = max(1, int(args.bloom_blur_pases)) if hasattr(args, "bloom_blur_pases") else None
args.bloom_blur_passes = max(1, int(args.bloom_blur_passes))


# ------------------------------
# v5.3 Palette (exact)
# ------------------------------
TURRELL_5_PALETTE = [
    "#330E21",  # ink (depth)
    "#3A08D2",  # indigo
    "#6D00E4",  # violet
    "#A632FD",  # electric purple
    "#F22BCB",  # fuchsia
    "#FC67C0",  # neon pink
    "#FD540A",  # blaze orange
    "#FF8E24",  # amber
    "#C4B1DB",  # lavender-white bloom
]


# ------------------------------
# NDJSON follower (same as your v6)
# ------------------------------
class StateFollower:
    def __init__(self, path):
        self.path = Path(path)
        self.file = None
        self.pos = 0
        self.state = None
        self.last_ts = 0.0

    def _open(self):
        if self.file:
            return
        if not self.path.exists():
            return
        try:
            self.file = open(self.path, "r", encoding="utf-8", errors="ignore")
            self.file.seek(self.pos if self.pos else 0, os.SEEK_SET)
        except Exception:
            self.file = None

    def update(self):
        self._open()
        if not self.file:
            return False
        updated = False
        try:
            cur_size = os.path.getsize(self.path)
            if cur_size < self.pos:
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
                    self.last_ts = float(packet.get("ts_unix", time.time()))
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
# HUD text helpers (your robust fallback)
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
    try:
        pygame.font.init()
        f = pygame.font.SysFont("consolas, monaco, monospace", 22)
        if f:
            return f, lambda txt, col: f.render(txt, True, col)
    except Exception as exc:
        print(f"[turrell] SysFont unavailable: {exc}", flush=True)

    if ft:
        try:
            ft.init()
            ff = ft.SysFont("Menlo,Consolas,Monaco,monospace", 22)
            if ff:
                return ff, lambda txt, col: ff.render(txt, col)[0]
        except Exception as exc:
            print(f"[turrell] freetype unavailable: {exc}", flush=True)

    return None, lambda txt, col: _pixel_text_surface(txt, col)

def draw_text(screen, render_fn, text, color, pos):
    surf = render_fn(text, color)
    screen.blit(surf, pos)


# ------------------------------
# v5.3-style renderer (minimal port)
# ------------------------------
def clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else 1.0 if x >= 1.0 else x

def clampf(x: float, lo: float, hi: float) -> float:
    try:
        xf = float(x)
    except Exception:
        return lo
    if xf < lo:
        return lo
    if xf > hi:
        return hi
    return xf

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def smoothstep(a: float, b: float, x):
    if HAVE_NUMPY and isinstance(x, np.ndarray):
        t = np.clip((x - a) / (b - a), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    else:
        t = clamp01((x - a) / (b - a))
        return t * t * (3.0 - 2.0 * t)

def ema(prev: float, x: float, alpha: float) -> float:
    return (1.0 - alpha) * prev + alpha * x

def alpha_from_dt(dt_s: float, alpha_per_sec: float) -> float:
    if dt_s <= 0:
        return 0.0
    k = max(0.0, float(alpha_per_sec))
    return 1.0 - math.exp(-k * dt_s)

if HAVE_NUMPY:
    def hex_to_srgb01(h: str) -> np.ndarray:
        h = h.strip().lstrip("#")
        return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0

    def srgb_to_linear(c: np.ndarray) -> np.ndarray:
        c = np.clip(c, 0.0, 1.0)
        a = 0.055
        return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1.0 + a)) ** 2.4)

    def linear_to_srgb(c: np.ndarray) -> np.ndarray:
        c = np.clip(c, 0.0, None)
        a = 0.055
        return np.where(c <= 0.0031308, 12.92 * c, (1.0 + a) * (c ** (1.0 / 2.4)) - a)

    def make_uv(w: int, h: int):
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        u = (x / (w - 1.0)) * 2.0 - 1.0
        v = (y / (h - 1.0)) * 2.0 - 1.0
        u *= (w / h)
        return u, v

    def sdf_round_rect(u, v, cx, cy, hx, hy, r):
        qx = np.abs(u - cx) - (hx - r)
        qy = np.abs(v - cy) - (hy - r)
        qx2 = np.maximum(qx, 0.0)
        qy2 = np.maximum(qy, 0.0)
        outside = np.sqrt(qx2*qx2 + qy2*qy2) - r
        inside = np.minimum(np.maximum(qx, qy), 0.0)
        return outside + inside

    def blur_box(img: np.ndarray, passes: int = 3) -> np.ndarray:
        out = img
        for _ in range(passes):
            out = (
                out
                + np.roll(out, 1, 0) + np.roll(out, -1, 0)
                + np.roll(out, 1, 1) + np.roll(out, -1, 1)
                + np.roll(np.roll(out, 1, 0), 1, 1)
                + np.roll(np.roll(out, 1, 0), -1, 1)
                + np.roll(np.roll(out, -1, 0), 1, 1)
                + np.roll(np.roll(out, -1, 0), -1, 1)
            ) / 9.0
        return out

    def add_bloom(lin_rgb: np.ndarray, strength: float, threshold: float, down: int, blur_passes: int) -> np.ndarray:
        small = lin_rgb[::down, ::down]
        bright = np.clip(small - threshold, 0.0, None)
        blur = blur_box(bright, passes=blur_passes)
        up = np.repeat(np.repeat(blur, down, axis=0), down, axis=1)
        up = up[:lin_rgb.shape[0], :lin_rgb.shape[1]]
        return lin_rgb + strength * up

    def tonemap_exp(lin_rgb: np.ndarray, exposure: float) -> np.ndarray:
        return 1.0 - np.exp(-lin_rgb * exposure)

    def dither(lin_rgb: np.ndarray, amp: float) -> np.ndarray:
        noise = (np.random.rand(*lin_rgb.shape).astype(np.float32) - 0.5) * amp
        return np.clip(lin_rgb + noise, 0.0, None)

    class TurrellishRenderer:
        def __init__(self, w: int, h: int, bloom_down: int = 4, bloom_blur_passes: int = 3):
            self.bloom_down = int(bloom_down)
            self.bloom_blur_passes = int(bloom_blur_passes)
            self.resize(w, h)

            self.turrell_lin = [srgb_to_linear(hex_to_srgb01(hx)) for hx in TURRELL_5_PALETTE]
            self.deep = self.turrell_lin[0]
            self.white = self.turrell_lin[-1]

        def resize(self, w: int, h: int):
            self.w = int(w)
            self.h = int(h)
            self.u, self.v = make_uv(self.w, self.h)

        def palette_at(self, t: float) -> np.ndarray:
            t = clamp01(t) * (len(self.turrell_lin) - 1)
            i = int(t)
            f = t - i
            c0 = self.turrell_lin[i]
            c1 = self.turrell_lin[min(i + 1, len(self.turrell_lin) - 1)]
            return c0 * (1.0 - f) + c1 * f

        def palette_q_triplet(self, q: float):
            q0 = clamp01(q)
            portal_q = clamp01(q0 + 0.12)
            rim_q = clamp01(q0 + 0.22)
            wall = self.palette_at(q0)
            portal = self.palette_at(portal_q)
            rim = self.palette_at(rim_q)
            return wall, portal, rim

        def _room_base(self, wall_col: np.ndarray, t: float) -> np.ndarray:
            u, v = self.u, self.v
            r = np.sqrt((u * 0.95) ** 2 + (v * 0.90) ** 2)
            vign = 1.0 - 0.28 * smoothstep(0.22, 1.60, r)
            vertical = smoothstep(-1.15, 0.25, v)

            room = wall_col[None, None, :] * (0.84 + 0.16 * vertical)[..., None]
            room = room * vign[..., None] + self.deep[None, None, :] * (1.0 - vign[..., None]) * 0.85

            air = 0.015 * math.sin(t * 0.55)
            room *= (1.0 + air)
            return room

        def _apply_post(self, lin: np.ndarray, exposure: float, bloom_strength: float, bloom_thr: float, micro_texture_gain: float, warmth_bias: float) -> np.ndarray:
            lin = add_bloom(lin, strength=bloom_strength, threshold=bloom_thr, down=self.bloom_down, blur_passes=self.bloom_blur_passes)
            lin = tonemap_exp(lin, exposure=exposure * (1.0 + float(warmth_bias)))
            lin = dither(lin, amp=(0.9 / 255.0) * float(micro_texture_gain))
            srgb = linear_to_srgb(lin)
            return (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

        def render(self, mode: str, x: float, q: float, hce: float, valid_ema: float, qc_ema: float, t: float,
                   exposure: float, bloom_strength: float, bloom_thr: float, drift_speed: float,
                   micro_texture_gain: float, warmth_bias: float) -> np.ndarray:
            valid_factor = lerp(0.18, 1.0, clamp01(valid_ema))
            qc_factor = clamp01((qc_ema - 0.35) / 0.55)
            gate = lerp(0.35, 1.0, valid_factor * qc_factor)

            h_norm = clampf(hce / 50.0, 0.0, 1.0)
            peak = smoothstep(0.25, 1.0, h_norm)
            gold_boost = 1.0 + 1.8 * peak
            cosmic_depth = 1.0 + 1.2 * peak

            style = {"exp": 1.00, "bloom": 1.00, "bloom_thr": 1.00, "drift": 1.00, "micro": 1.00, "warmth": 0.0}
            if mode == "MAX_FOCUS":
                style.update({"exp": 0.95, "bloom": 0.90, "bloom_thr": 0.95, "drift": 1.15, "micro": 0.95, "warmth": -0.015})
            elif mode == "MAX_Q":
                style.update({"exp": 1.05, "bloom": 1.05, "bloom_thr": 0.90, "drift": 1.10, "micro": 1.00, "warmth": 0.01})
            elif mode == "STABILIZE":
                style.update({"exp": 1.00, "bloom": 1.00, "bloom_thr": 1.05, "drift": 0.90, "micro": 1.05, "warmth": -0.005})

            exposure_eff = clampf(exposure * style["exp"] * (1.0 + 0.5 * peak), 0.60, 2.20)
            bloom_strength_eff = clampf(bloom_strength * style["bloom"] * (1.0 + 0.8 * peak), 0.18, 2.0)
            bloom_thr_eff = clampf(bloom_thr * style["bloom_thr"], 0.12, 0.85)
            # Relaxed clamp to allow tiny drift speeds (e.g., 0.007)
            drift_eff = clampf(drift_speed * style["drift"], 0.001, 2.50)
            micro_eff = clampf(micro_texture_gain * style["micro"], 0.65, 1.30)
            warmth_eff = clampf(warmth_bias + style.get("warmth", 0.0), -0.08, 0.08)

            # Core look: v5.3 “OFF-room” style (used for all modes; mode alters style params)
            sp = 0.05 * (0.30 + q) * drift_eff
            motion = 0.0
            wobble = 0.0
            phase = math.tau * t * drift_eff
            pal_wobble = 0.060 * math.sin(phase) + 0.030 * math.sin(phase * 0.37 + 1.7)
            q_for_palette = clamp01(0.15 + 0.55 * q + pal_wobble)
            wall, portal, rim = self.palette_q_triplet(q_for_palette)

            cx = 0.0
            cy = -0.06
            aspect = self.w / self.h
            hx = 0.74 * aspect
            hy = 0.54

            lin = self._room_base(wall, t)
            d = sdf_round_rect(self.u, self.v, cx=cx, cy=cy, hx=hx, hy=hy, r=0.09)

            portal_mask = 1.0 - smoothstep(-0.02, 0.065, d)
            halo = np.exp(-(np.maximum(d, 0.0) / (0.22 + 0.08 * q_for_palette)) ** 2)
            rim_mask = np.exp(-(np.abs(d) / 0.010) ** 2)

            base_intensity = (0.35 + 1.10 * q_for_palette) * gate * cosmic_depth
            intensity = clampf(base_intensity, 0.10, 1.65)

            lin += portal_mask[..., None] * portal[None, None, :] * (1.85 * intensity * gold_boost)
            lin += halo[..., None] * portal[None, None, :] * (0.50 * intensity * gold_boost)
            lin += rim_mask[..., None] * rim[None, None, :] * (2.05 * intensity * gold_boost)

            floor = smoothstep(0.10, 1.00, self.v)
            lin *= (1.0 - 0.08 * floor)[..., None]

            exposure_eff = clampf(exposure_eff * 1.02, 0.65, 1.85)
            bloom_strength_eff = clampf(bloom_strength_eff * 1.05, 0.18, 1.65)
            bloom_thr_eff = clampf(bloom_thr_eff, 0.18, 0.85)
            micro_eff = clampf(micro_eff, 0.65, 1.25)

            return self._apply_post(
                lin,
                exposure=exposure_eff,
                bloom_strength=bloom_strength_eff,
                bloom_thr=bloom_thr_eff,
                micro_texture_gain=micro_eff,
                warmth_bias=warmth_eff,
            )


# ------------------------------
# HUD + live dot (your v6 behavior)
# ------------------------------
def draw_live_dot(screen, alive, valid, qualia, x, y):
    color = (50, 255, 50) if (alive and valid and qualia) else (255, 50, 50)
    pygame.draw.circle(screen, color, (x, y), 14)
    if alive:
        pulse = int(10 + 6 * math.sin(time.time() * 10))
        pygame.draw.circle(screen, (255, 255, 255), (x, y), pulse, width=4)

def draw_hud(screen, render_fn, mode_info, follower):
    if not follower.state:
        return
    state = follower.state
    rel = state.get("reliability", {}) if isinstance(state.get("reliability", {}), dict) else {}
    raw = state.get("raw", {}) if isinstance(state.get("raw", {}), dict) else {}
    hce_raw = float(raw.get("HCE_raw", 0.0) or 0.0)

    lines = [
        f"Turrell Room — Mode: {mode_info.get('name', 'OFF')}",
        f"X: {float(state.get('X', 0.0) or 0.0):.3f} | Q(vibe_focus): {float(state.get('Q_vibe_focus', 0.0) or 0.0):.3f}",
        f"HCE_raw: {hce_raw:.4f}   (HCE display may be scaled in stream)",
        f"Valid: {bool(rel.get('valid', False))} | Qualia: {bool(rel.get('qualia_valid', False))} | QC: {float(rel.get('quality_conf', rel.get('qc', 0.0)) or 0.0):.2f}",
        "",
        "Hotkeys: 0-3 Modes | F Fullscreen | H HUD | ESC Quit",
        "HUD ACTIVE",
    ]
    y = 30
    for line in lines:
        draw_text(screen, render_fn, line, (230, 230, 255), (30, y))
        y += 22


# ------------------------------
# helpers: extract X/Q/HCE + normalize (match v5.3 feel)
# ------------------------------
def pick_q(state: dict, q_metric: str) -> float:
    qm = (q_metric or "auto").strip().lower()
    # prefer masked/top-level if present
    if qm in ("vibe_focus", "vibefocus", "q_vibe_focus"):
        return float(state.get("Q_vibe_focus", state.get("q_vibe_focus", 0.0)) or 0.0)
    if qm in ("vibe", "q_vibe"):
        return float(state.get("Q_vibe", state.get("q_vibe", 0.0)) or 0.0)
    if qm in ("focus", "q_focus"):
        return float(state.get("Q_focus", state.get("q_focus", 0.0)) or 0.0)
    if qm in ("abs", "q_abs"):
        return float(state.get("Q_abs", state.get("q_abs", 0.0)) or 0.0)

    # auto fallback
    for k in ("Q_vibe_focus", "Q_abs", "Q_focus", "Q_vibe", "q_vibe_focus", "q_abs", "q_focus", "q_vibe"):
        if k in state and state[k] is not None:
            try:
                return float(state[k])
            except Exception:
                pass
    return 0.0

def normalize_x(x_raw: float) -> float:
    if args.x_max <= args.x_min:
        return 0.0
    x_n = (x_raw - args.x_min) / (args.x_max - args.x_min)
    x_n = clamp01(x_n)
    return clamp01(x_n ** 0.6)

def normalize_q(q_raw: float) -> float:
    if args.q_max <= args.q_min:
        return 0.0
    q_n = (q_raw - args.q_min) / (args.q_max - args.q_min)
    q_n = clamp01(q_n)
    return clamp01(q_n ** 0.7)

def get_hce_scaled(state: dict) -> float:
    # Prefer scaled HCE if present
    if "HCE" in state and state["HCE"] is not None:
        try:
            return float(state["HCE"])
        except Exception:
            pass
    raw = state.get("raw", {}) if isinstance(state.get("raw", {}), dict) else {}
    # Fallback: raw * 10k (matches your scaling era; still “looks right”)
    if "HCE_raw" in raw and raw["HCE_raw"] is not None:
        try:
            return float(raw["HCE_raw"]) * 10000.0
        except Exception:
            pass
    return 0.0


# ------------------------------
# Main
# ------------------------------
def main():
    if not HAVE_NUMPY:
        print("ERROR: Style-matched Turrell renderer needs numpy. Install: pip install numpy", flush=True)
        return 2

    pygame.init()
    flags = pygame.RESIZABLE
    if args.fullscreen:
        flags |= pygame.FULLSCREEN
    try:
        screen = pygame.display.set_mode((SW, SH), flags, display=args.display)
    except Exception as exc:
        print(f"[turrell] display set_mode failed ({exc}), retrying without display hint", flush=True)
        screen = pygame.display.set_mode((SW, SH), flags)

    pygame.display.set_caption("ALMA OS Turrell Room — v6 (v5.3 style)")
    clock = pygame.time.Clock()
    font_obj, render_fn = make_font()
    if font_obj is None:
        print("[turrell] HUD using pixel fallback", flush=True)

    follower = StateFollower(args.ndjson)

    mode_path = Path(args.mode_file)
    mode_info = {"mode": "OFF", "name": "OFF"}
    if mode_path.exists():
        try:
            mode_info = json.load(open(mode_path))
        except Exception:
            pass

    show_hud = True
    toast = "Turrell Room — v5.3 palette/tonemap/bloom style"
    toast_until = time.time() + 5
    esc_confirm_until = 0.0

    # Drift presets (4/5/6/7) and live drift value
    drift_presets = {
        pygame.K_4: 0.002,
        pygame.K_5: 0.007,
        pygame.K_6: 0.18,
        pygame.K_7: 0.7,
    }
    drift_speed_live = float(args.drift_speed if args.drift_speed is not None else 0.007)

    # Smoothing for gate + HCE (visual butter)
    valid_ema = 0.0
    qc_ema = 0.8
    hce_ema = 0.0
    last_wall = time.time()

    # Visual-only smoothing (sensual butter)
    q_vis = 0.0
    x_vis = 0.0
    vis_init = False

    renderer = None
    render_w = render_h = 0

    running = True
    t0 = time.time()
    while running:
        now = time.time()
        t = now - t0
        dt = now - last_wall
        last_wall = now
        dt = max(0.0, min(dt, 0.25))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                print(f"[turrell] Key pressed: {event.key}", flush=True)
                if event.key == pygame.K_ESCAPE:
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
                elif event.key in drift_presets:
                    drift_speed_live = float(drift_presets[event.key])
                    args.drift_speed = drift_speed_live
                    toast = f"Drift speed set to {drift_speed_live:g}"
                    toast_until = now + 2.0
                elif event.key in (pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3):
                    modes = {pygame.K_0: "OFF", pygame.K_1: "MAX_FOCUS", pygame.K_2: "MAX_Q", pygame.K_3: "STABILIZE"}
                    mode_info["mode"] = modes[event.key]
                    mode_info["name"] = modes[event.key]
                    try:
                        with open(mode_path, "w") as f:
                            json.dump(mode_info, f)
                    except Exception as exc:
                        print(f"[turrell] Failed to write mode file: {exc}", flush=True)

        follower.update()
        state = follower.state or {}
        rel = state.get("reliability", {}) if isinstance(state.get("reliability", {}), dict) else {}

        # Heartbeat
        alive = (follower.last_ts > 0.0) and ((time.time() - follower.last_ts) < 1.5)

        # Validity + QC (for gating like v5.3)
        valid = bool(rel.get("valid", False))
        qualia_valid = bool(rel.get("qualia_valid", True))
        qc = float(rel.get("quality_conf", rel.get("qc", 0.8)) or 0.8)

        av = alpha_from_dt(dt, 3.0)
        aqc = alpha_from_dt(dt, 3.0)
        valid_ema = ema(valid_ema, 1.0 if valid else 0.0, av if dt > 0 else 1.0)
        qc_ema = ema(qc_ema, qc, aqc if dt > 0 else 1.0)

        # Metrics
        x_raw = float(state.get("X", 0.0) or 0.0)
        q_raw = pick_q(state, args.q_metric)
        hce_scaled = get_hce_scaled(state)
        hce_ema = ema(hce_ema, max(0.0, float(hce_scaled)), alpha_from_dt(dt, 2.0) if dt > 0 else 1.0)

        x_n = normalize_x(x_raw)
        q_n = normalize_q(q_raw)

        # Visual-only smoothing (frame-based; dt-aware alpha)
        a_vis = alpha_from_dt(dt, 2.1)
        if not vis_init:
            q_vis = float(q_n)
            x_vis = float(x_n)
            vis_init = True
        else:
            q_vis = ema(q_vis, float(q_n), a_vis)
            x_vis = ema(x_vis, float(x_n), a_vis)

        # Renderer resize (internal render)
        sw, sh = screen.get_size()
        target_w = max(240, int(sw * float(args.render_scale)))
        target_h = max(160, int(sh * float(args.render_scale)))

        if renderer is None or target_w != render_w or target_h != render_h:
            renderer = TurrellishRenderer(target_w, target_h, bloom_down=int(args.bloom_down), bloom_blur_passes=int(args.bloom_blur_passes))
            render_w, render_h = target_w, target_h

        mode = (mode_info.get("mode", "OFF") or "OFF").strip().upper()

        img = renderer.render(
            mode=mode,
            x=float(x_vis),
            q=float(q_vis),
            hce=float(hce_ema),
            valid_ema=float(valid_ema),
            qc_ema=float(qc_ema),
            t=float(t),
            exposure=float(args.exposure),
            bloom_strength=float(args.bloom_strength),
            bloom_thr=float(args.bloom_threshold),
            drift_speed=float(drift_speed_live),
            micro_texture_gain=1.0,
            warmth_bias=0.0,
        )

        surf = pygame.surfarray.make_surface(np.swapaxes(img, 0, 1))
        surf = pygame.transform.smoothscale(surf, (sw, sh))
        screen.blit(surf, (0, 0))

        draw_live_dot(
            screen,
            alive=alive,
            valid=valid,
            qualia=qualia_valid,
            x=screen.get_width() - 40,
            y=40,
        )

        if show_hud:
            draw_hud(screen, render_fn, mode_info, follower)

        if toast and time.time() < toast_until:
            draw_text(screen, render_fn, toast, (235, 235, 255), (30, screen.get_height() - 50))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
