#!/usr/bin/env python3
"""
Raspberry Pi HQ Camera Script — HIGHLY OPTIMIZED FOR PI ZERO
- Sensor:  Official Raspberry Pi HQ Camera (IMX477)
- Display: 3.5" GPIO touchscreen, 480x320
- Shutter: GPIO26 momentary button (hold 2s = shutter-set mode, short = capture/cycle)
- Film profiles: tap the on-screen FILM button to cycle through profiles
"""

import os
import time
import threading
import traceback
from datetime import datetime

import cv2
import numpy as np
from gpiozero import Button
from picamera2 import Picamera2

# ------------------------------------------------------------
#  Configuration
# ------------------------------------------------------------
PICTURES_DIR = "/home/pi/Pictures"

SCREEN_W, SCREEN_H = 480, 320
PREVIEW_W, PREVIEW_H = 426, 320  # Active 4:3 preview area scaled to fit 480x320
BAR_H      = 40
FILM_BTN_W = 180
FILM_BTN_H = 42

PEAK_EVERY     = 4    # recompute focus peaking every N frames
PEAK_THRESHOLD = 28   # raise = fewer highlights, lower = more

EV_OPTIONS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
current_ev_idx = 4

AWB_MODES = [("Auto", 0), ("Daylight", 5), ("Cloudy", 6), ("Tungsten", 2), ("Fluorescent", 3)]
current_awb_idx = 0

zoom_levels      = [1.0, 2.0, 4.0]
current_zoom_idx = 0
zoom_center      = (0.5, 0.5)

sleep_mode       = False
press_start_time = 0.0
image_count      = 0
focus_peaking_enabled = True

os.system("unclutter &")
os.makedirs(PICTURES_DIR, exist_ok=True)

btn_bounds       = {"bx1": 0, "bx2": FILM_BTN_W, "by1": 0, "by2": FILM_BTN_H}
btn_bounds_ev    = {"bx1": 0, "bx2": 0, "by1": 0, "by2": 0}
btn_bounds_awb   = {"bx1": 0, "bx2": 0, "by1": 0, "by2": 0}
btn_bounds_pm    = {"bx1": 0, "bx2": 0, "by1": 0, "by2": 0}
btn_bounds_meter = {"bx1": 0, "bx2": 0, "by1": 0, "by2": 0}

# ============================================================
#  PRECOMPUTED ASSETS (Built once at startup)
# ============================================================

def _lut_from_curve(pts):
    xs = np.array([p[0] for p in pts], dtype=np.float32)
    xs = np.array([p[0] for p in pts], dtype=np.float32)
    ys = np.array([p[1] for p in pts], dtype=np.float32)
    return np.clip(np.interp(np.arange(256), xs, ys), 0, 255).astype(np.uint8)

# --- Channel LUTs ---
_LUT_CC_B = _lut_from_curve([(0,25),(64,80),(128,132),(192,188),(255,242)])
_LUT_CC_G = _lut_from_curve([(0,18),(64,72),(128,126),(192,184),(255,238)])
_LUT_CC_R = _lut_from_curve([(0,20),(64,75),(128,128),(192,185),(255,240)])

_LUT_KP_B = _lut_from_curve([(0,12),(64,68),(128,118),(192,172),(255,220)])
_LUT_KP_G = _lut_from_curve([(0, 8),(64,74),(128,130),(192,192),(255,248)])
_LUT_KP_R = _lut_from_curve([(0,10),(64,80),(128,138),(192,200),(255,255)])

_LUT_FV_B = _lut_from_curve([(0,0),(64,66),(128,140),(192,208),(255,255)])
_LUT_FV_G = _lut_from_curve([(0,0),(64,62),(128,135),(192,205),(255,255)])
_LUT_FV_R = _lut_from_curve([(0,0),(64,60),(128,130),(192,200),(255,255)])

_LUT_FA_B = _lut_from_curve([(0,8),(64,70),(128,126),(192,188),(255,245)])
_LUT_FA_G = _lut_from_curve([(0,4),(64,70),(128,130),(192,194),(255,250)])
_LUT_FA_R = _lut_from_curve([(0,5),(64,72),(128,132),(192,196),(255,252)])

_LUT_IB   = _lut_from_curve([(0,0),(60,50),(128,128),(190,210),(255,255)])

_LUT_KG_B = _lut_from_curve([(0,20),(64,60),(128,110),(192,162),(255,210)])
_LUT_KG_G = _lut_from_curve([(0,10),(64,76),(128,132),(192,194),(255,248)])
_LUT_KG_R = _lut_from_curve([(0,15),(64,85),(128,142),(192,205),(255,255)])

# --- Vignette masks (Precomputed as 3-channel uint8 arrays to exploit cv2.multiply) ---
def _make_vignette_mask(w, h, strength):
    Y, X  = np.ogrid[:h, :w]
    dist  = np.sqrt(((X - w/2)/(w/2))**2 + ((Y - h/2)/(h/2))**2)
    mask  = (1.0 - np.clip(dist * strength, 0, 1))
    mask_3ch = np.stack([mask, mask, mask], axis=-1)
    return (mask_3ch * 255).astype(np.uint8)

_VIG_MASK_PREVIEW_KP = _make_vignette_mask(PREVIEW_W, PREVIEW_H, 0.25)
_VIG_MASK_PREVIEW_KG = _make_vignette_mask(PREVIEW_W, PREVIEW_H, 0.30)

# --- Grain Buffers (Separated into Add and Subtract arrays to remain fully uint8) ---
_noise_kg = np.random.normal(0, 4, (PREVIEW_H, PREVIEW_W, 3))
_GRAIN_KG_ADD = np.clip(_noise_kg, 0, 255).astype(np.uint8)
_GRAIN_KG_SUB = np.clip(-_noise_kg, 0, 255).astype(np.uint8)

_noise_ib = np.random.normal(0, 5, (PREVIEW_H, PREVIEW_W, 3))
_GRAIN_IB_ADD = np.clip(_noise_ib, 0, 255).astype(np.uint8)
_GRAIN_IB_SUB = np.clip(-_noise_ib, 0, 255).astype(np.uint8)

# --- Static Canvas Buffer (Reused every frame) ---
_canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

# --- Text block cache ---
_tb_cache = {}

# ============================================================
#  FAST HELPERS
# ============================================================

def _apply_channel_luts(img, lb, lg, lr):
    b, g, r = cv2.split(img)
    return cv2.merge([cv2.LUT(b, lb), cv2.LUT(g, lg), cv2.LUT(r, lr)])

def _sat_fast(img, scale):
    """Changes saturation by blending with grayscale version. Avoids heavy HSV math."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, scale, gray_3ch, 1.0 - scale, 0)

def _vignette_fast(img, mask_u8):
    """Apply a precomputed uint8 vignette mask using optimized C-level scaling."""
    return cv2.multiply(img, mask_u8, scale=1.0/255.0)

def _vignette_still(img, strength):
    """Vignette for full-res stills — mask computed on demand."""
    h, w  = img.shape[:2]
    Y, X  = np.ogrid[:h, :w]
    dist  = np.sqrt(((X-w/2)/(w/2))**2 + ((Y-h/2)/(h/2))**2)
    mask  = (1.0 - np.clip(dist*strength, 0, 1)).astype(np.float32)[:, :, np.newaxis]
    return np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)

def _grain_fast(img, g_add, g_sub):
    """Add precomputed grain buffer without converting arrays to float32."""
    return cv2.subtract(cv2.add(img, g_add), g_sub)

def _grain_still(img, amount):
    """Grain for full-res stills — fresh noise each shot."""
    noise = np.random.normal(0, amount, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def apply_pro_mist(img, threshold=190, glow_spread=15, blend=0.25):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    highlights = cv2.bitwise_and(img, img, mask=mask)
    h, w = img.shape[:2]
    small = cv2.resize(highlights, (w//4, h//4), interpolation=cv2.INTER_NEAREST)
    if glow_spread % 2 == 0: glow_spread += 1
    blurred = cv2.GaussianBlur(small, (glow_spread, glow_spread), 0)
    blurred = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(img, 1.0, blurred, blend, 0)

# ============================================================
#  FILM PROFILES
# ============================================================

def profile_standard(img, preview=True):
    return img

def profile_classic_chrome(img, preview=True):
    out = _apply_channel_luts(img, _LUT_CC_B, _LUT_CC_G, _LUT_CC_R)
    if preview:
        return _sat_fast(out, 0.72)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.LUT(hsv[:, :, 1], np.clip(np.arange(256)*0.72, 0, 255).astype(np.uint8))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def profile_kodak_portra(img, preview=True):
    out = _apply_channel_luts(img, _LUT_KP_B, _LUT_KP_G, _LUT_KP_R)
    if preview:
        return _vignette_fast(_sat_fast(out, 0.85), _VIG_MASK_PREVIEW_KP)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.LUT(hsv[:, :, 1], np.clip(np.arange(256)*0.85, 0, 255).astype(np.uint8))
    return _vignette_still(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), 0.25)

def profile_fuji_velvia(img, preview=True):
    out = _apply_channel_luts(img, _LUT_FV_B, _LUT_FV_G, _LUT_FV_R)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 1] = np.clip((hsv[:, :, 1].astype(np.float32) * 1.45), 0, 255).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] - 3) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def profile_fuji_astia(img, preview=True):
    out = _apply_channel_luts(img, _LUT_FA_B, _LUT_FA_G, _LUT_FA_R)
    if preview:
        return _sat_fast(out, 0.95)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.LUT(hsv[:, :, 1], np.clip(np.arange(256)*0.95, 0, 255).astype(np.uint8))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def profile_ilford_bw(img, preview=True):
    b, g, r = cv2.split(img)
    pan = np.clip(0.21*r.astype(np.float32) + 0.72*g.astype(np.float32) + 0.07*b.astype(np.float32), 0, 255).astype(np.uint8)
    pan = cv2.LUT(pan, _LUT_IB)
    bgr = cv2.cvtColor(pan, cv2.COLOR_GRAY2BGR)
    if preview:
        return _grain_fast(bgr, _GRAIN_IB_ADD, _GRAIN_IB_SUB)
    return _grain_still(bgr, 5)

def profile_kodak_gold(img, preview=True):
    out = _apply_channel_luts(img, _LUT_KG_B, _LUT_KG_G, _LUT_KG_R)
    if preview:
        out = _sat_fast(out, 0.90)
        out = _vignette_fast(out, _VIG_MASK_PREVIEW_KG)
        return _grain_fast(out, _GRAIN_KG_ADD, _GRAIN_KG_SUB)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.LUT(hsv[:, :, 1], np.clip(np.arange(256)*0.90, 0, 255).astype(np.uint8))
    out = _vignette_still(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), 0.30)
    return _grain_still(out, 4)

FILM_PROFILES = [
    ("Standard",       profile_standard,       (180, 180, 180)),
    ("Classic Chrome", profile_classic_chrome, ( 80, 180, 160)),
    ("Kodak Portra",   profile_kodak_portra,   ( 40, 140, 230)),
    ("Fuji Velvia",    profile_fuji_velvia,    ( 30, 200,  90)),
    ("Fuji Astia",     profile_fuji_astia,     (200, 160,  80)),
    ("Ilford B&W",     profile_ilford_bw,      (210, 210, 210)),
    ("Kodak Gold",     profile_kodak_gold,     (  0, 190, 230)),
]
current_profile_idx = 0

def apply_current_profile(img, preview=True):
    return FILM_PROFILES[current_profile_idx][1](img, preview=preview)

def cycle_film_profile():
    global current_profile_idx
    current_profile_idx = (current_profile_idx + 1) % len(FILM_PROFILES)
    print(f"[Film Profile] {FILM_PROFILES[current_profile_idx][0]}")

# ============================================================
#  CAMERA / CONTROLS
# ============================================================
METERING_MODES    = [("Matrix", 2), ("Center", 0), ("Spot", 1)]
current_meter_idx = 1

def cycle_metering():
    global current_meter_idx
    current_meter_idx = (current_meter_idx + 1) % len(METERING_MODES)
    name, val = METERING_MODES[current_meter_idx]
    picam2.set_controls({"AeMeteringMode": val})
    print(f"[Metering] {name}")

def cycle_ev():
    global current_ev_idx
    current_ev_idx = (current_ev_idx + 1) % len(EV_OPTIONS)
    val = EV_OPTIONS[current_ev_idx]
    picam2.set_controls({"ExposureValue": val})
    print(f"[EV] {val:+}")

def cycle_awb():
    global current_awb_idx
    current_awb_idx = (current_awb_idx + 1) % len(AWB_MODES)
    name, val = AWB_MODES[current_awb_idx]
    picam2.set_controls({"AwbMode": val})
    print(f"[AWB] {name}")

pro_mist_enabled = False
def toggle_pro_mist():
    global pro_mist_enabled
    pro_mist_enabled = not pro_mist_enabled
    print(f"[Pro-Mist] {'ON' if pro_mist_enabled else 'OFF'}")

def handle_focus_tap(x, y):
    global current_zoom_idx, zoom_center
    zoom_center      = (x / SCREEN_W, y / SCREEN_H)
    current_zoom_idx = (current_zoom_idx + 1) % len(zoom_levels)
    print(f"[Zoom] {zoom_levels[current_zoom_idx]}x @ {zoom_center}")

# ============================================================
#  SLEEP / WAKE
# ============================================================
_just_woke = False

def enter_sleep_mode():
    global sleep_mode
    sleep_mode = True

def wake_display():
    global sleep_mode
    if sleep_mode:
        print("[Sleep] Wake")
        sleep_mode = False

def _on_pressed():
    global press_start_time, _just_woke
    press_start_time = time.time()
    if sleep_mode:
        _just_woke = True
    wake_display()

def _on_released():
    global _hold_fired, press_start_time, _just_woke
    if _just_woke:
        _just_woke = False
        return
    press_duration = time.time() - press_start_time
    if press_duration >= 10.0:
        return
    if press_duration > 5:
        global shutter_set_mode
        shutter_set_mode = False
        _hold_fired = False
        enter_sleep_mode()
        return
    if _hold_fired:
        _hold_fired = False
        return
    if shutter_set_mode:
        _cycle_shutter()
    else:
        shoot_event.set()

# ============================================================
#  TOUCHSCREEN
# ============================================================
_touch_lock    = threading.Lock()
_last_tap_time = 0.0

def _on_mouse(event, x, y, flags, param):
    global _last_tap_time
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    now = time.time()
    with _touch_lock:
        if now - _last_tap_time < 0.35:
            return
        _last_tap_time = now
    if   btn_bounds["bx1"]       <= x <= btn_bounds["bx2"]       and btn_bounds["by1"]       <= y <= btn_bounds["by2"]:       cycle_film_profile()
    elif btn_bounds_pm["bx1"]    <= x <= btn_bounds_pm["bx2"]    and btn_bounds_pm["by1"]    <= y <= btn_bounds_pm["by2"]:    toggle_pro_mist()
    elif btn_bounds_meter["bx1"] <= x <= btn_bounds_meter["bx2"] and btn_bounds_meter["by1"] <= y <= btn_bounds_meter["by2"]: cycle_metering()
    elif btn_bounds_ev["bx1"]    <= x <= btn_bounds_ev["bx2"]    and btn_bounds_ev["by1"]    <= y <= btn_bounds_ev["by2"]:    cycle_ev()
    elif btn_bounds_awb["bx1"]   <= x <= btn_bounds_awb["bx2"]   and btn_bounds_awb["by1"]   <= y <= btn_bounds_awb["by2"]:   cycle_awb()
    else: handle_focus_tap(x, y)

# ============================================================
#  DRAWING HELPERS
# ============================================================
def ensure_channels(img, ch):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR if ch == 3 else cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == ch: return img
    if img.shape[2] == 3 and ch == 4: return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    if img.shape[2] == 4 and ch == 3: return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def blit_add(dst, src, x, y):
    dh, dw = dst.shape[:2]; sh, sw = src.shape[:2]
    if sw <= 0 or sh <= 0: return
    x1=max(0,x); y1=max(0,y); x2=min(dw,x+sw); y2=min(dh,y+sh)
    if x1>=x2 or y1>=y2: return
    roi   = dst[y1:y2, x1:x2]
    src_c = src[y1-y:y2-y, x1-x:x2-x]
    ch    = roi.shape[2] if roi.ndim == 3 else 3
    cv2.add(ensure_channels(roi, ch), ensure_channels(src_c, ch), dst=roi)
    dst[y1:y2, x1:x2] = roi

def format_shutter(us):
    return f"1/{int(round(1e6/us))}s" if us and us > 0 else "Auto"

# ============================================================
#  FOCUS PEAKING — Half-res integer Sobel
# ============================================================
def apply_focus_peaking(frame_bgr):
    h, w    = frame_bgr.shape[:2]
    half    = cv2.resize(frame_bgr, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
    gray_s  = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
    blur    = cv2.GaussianBlur(gray_s, (3, 3), 0)
    gx      = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3)
    gy      = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3)
    mag     = cv2.addWeighted(cv2.convertScaleAbs(gx), 0.5, cv2.convertScaleAbs(gy), 0.5, 0)
    _, mask_s = cv2.threshold(mag, PEAK_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask      = cv2.resize(mask_s, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay   = frame_bgr.copy()
    overlay[mask > 0] = (0, 255, 0)
    return cv2.addWeighted(frame_bgr, 0.7, overlay, 0.3, 0)

# ============================================================
#  HISTOGRAM — Fully Vectorized (No Loops, No Percentiles)
# ============================================================
_HIST_W = 128

def draw_histogram(gray, height=BAR_H, width=_HIST_W):
    hist  = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    bsz   = 256 // width
    comp  = hist[:bsz*width].reshape(width, bsz).mean(axis=1)
    
    dmax = comp.max()
    if dmax < 50.0: dmax = 50.0
    scale = (height - 4) / dmax
    bars  = np.minimum((comp * scale).astype(np.int32), height - 4)
    
    # Vectorized canvas drawing entirely in NumPy C-implementation
    img   = np.zeros((height, width, 3), dtype=np.uint8)
    Y = np.arange(height).reshape(height, 1)
    X_threshold = (height - 1 - bars).reshape(1, width)
    mask = Y >= X_threshold
    img[mask] = [200, 200, 200]
    return img

# ============================================================
#  TEXT BLOCK — Cached string layouts
# ============================================================
def make_text_block(lines, font_scale=0.42, thickness=1, max_h=BAR_H-6):
    key = "|".join(lines)
    if key in _tb_cache:
        return _tb_cache[key]
    font   = cv2.FONT_HERSHEY_SIMPLEX
    sizes  = [cv2.getTextSize(l, font, font_scale, thickness)[0] for l in lines]
    w      = max(s[0] for s in sizes) + 8
    line_h = max(s[1] for s in sizes) + 5
    h      = line_h * len(lines) + 6
    img    = np.zeros((h, w, 3), dtype=np.uint8)
    y      = 3 + sizes[0][1]
    for l in lines:
        cv2.putText(img, l, (4, y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
        y += line_h
    if img.shape[0] > max_h:
        sc  = max_h / img.shape[0]
        img = cv2.resize(img, (max(1, int(img.shape[1]*sc)), max_h), cv2.INTER_NEAREST)
    if len(_tb_cache) < 64:
        _tb_cache[key] = img
    return img

# ============================================================
#  BUTTON DRAWING — Optimized Alpha Blends
# ============================================================

def draw_film_button(canvas, name, accent_bgr, x, y):
    w, h = FILM_BTN_W, FILM_BTN_H
    ch_h, cw = canvas.shape[:2]
    if x+w > cw: w = cw-x
    if y+h > ch_h: h = ch_h-y
    if w <= 0 or h <= 0: return (x, y, x, y)
    
    # In-place darkening pass using fast integer scale logic (No addWeighted array allocs)
    canvas[y:y+h, x:x+w] = cv2.convertScaleAbs(canvas[y:y+h, x:x+w], alpha=0.40)
    
    cv2.rectangle(canvas, (x,y), (x+w-1,y+h-1), accent_bgr, 2)
    perf_w, perf_h = 6, 5
    n_perfs = max(2, h//12); spacing = h//(n_perfs+1)
    for i in range(n_perfs):
        fy = y + spacing*(i+1) - perf_h//2
        cv2.rectangle(canvas, (x+4,fy), (x+4+perf_w,fy+perf_h), accent_bgr, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX; fscale = 0.50
    (tw, th), _ = cv2.getTextSize(name[:16], font, fscale, 1)
    cv2.putText(canvas, name[:16], (x+4+perf_w+8, y+(h+th)//2), font, fscale, (255,255,255), 1, cv2.LINE_AA)
    return (x, y, x+w, y+h)

def draw_toggle_button(canvas, label, is_active, x, y):
    w, h = FILM_BTN_W, FILM_BTN_H
    ch_h, cw = canvas.shape[:2]
    if x+w > cw: w = cw-x
    if y+h > ch_h: h = ch_h-y
    if w <= 0 or h <= 0: return (x, y, x, y)
    
    canvas[y:y+h, x:x+w] = cv2.convertScaleAbs(canvas[y:y+h, x:x+w], alpha=0.40)
    
    accent = (0,200,0) if is_active else (100,100,100)
    cv2.rectangle(canvas, (x,y), (x+w-1,y+h-1), accent, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX; fscale = 0.50
    (tw, th), _ = cv2.getTextSize(label, font, fscale, 1)
    color = (255,255,255) if is_active else (150,150,150)
    cv2.putText(canvas, label, (x+(w-tw)//2, y+(h+th)//2), font, fscale, color, 1, cv2.LINE_AA)
    return (x, y, x+w, y+h)

# ============================================================
#  CAMERA SETUP
# ============================================================
picam2 = Picamera2()

FULL_W, FULL_H       = 4056, 3040
DEFAULT_FRAME_LIMITS = (125, 16667)

preview_config = picam2.create_preview_configuration(
    main={"size": (SCREEN_W, SCREEN_H), "format": "RGB888"},
    lores=None, display=None,
    controls={
        "AeMeteringMode":      2,
        "NoiseReductionMode":  0,
        "FrameDurationLimits": DEFAULT_FRAME_LIMITS,
    }
)

still_config = picam2.create_still_configuration(
    main={"size": (FULL_W, FULL_H)},
    controls={
        "AeMeteringMode":      2,
        "NoiseReductionMode":  0,
        "FrameDurationLimits": DEFAULT_FRAME_LIMITS,
    }
)

picam2.configure(preview_config)
picam2.start()
picam2.set_controls({"FrameDurationLimits": DEFAULT_FRAME_LIMITS})
time.sleep(1)

# ============================================================
#  SHUTTER CONTROL (GPIO26)
# ============================================================
SHUTTER_OPTIONS_US = [None, 33333, 16667, 8000, 4000, 2000, 1000, 500, 250]
SHUTTER_LABELS     = ["Auto","1/30","1/60","1/125","1/250","1/500","1/1000","1/2000","1/4000"]

current_shutter_idx = 0
shutter_set_mode    = False
_hold_fired         = False
shoot_event         = threading.Event()

def _apply_shutter():
    us = SHUTTER_OPTIONS_US[current_shutter_idx]
    if us is None:
        picam2.set_controls({"AeEnable": True, "FrameDurationLimits": DEFAULT_FRAME_LIMITS})
    else:
        picam2.set_controls({"AeEnable": False, "ExposureTime": int(us), "FrameDurationLimits": (int(us), int(us))})

def _toggle_shutter_set():
    global shutter_set_mode
    shutter_set_mode = not shutter_set_mode
    print(f"[Shutter Set] {'ON' if shutter_set_mode else 'OFF'} – {SHUTTER_LABELS[current_shutter_idx]}")
    _apply_shutter()

def _cycle_shutter():
    global current_shutter_idx
    current_shutter_idx = (current_shutter_idx + 1) % len(SHUTTER_OPTIONS_US)
    print(f"[Shutter] {SHUTTER_LABELS[current_shutter_idx]}")
    _apply_shutter()

button = Button(26, pull_up=True, bounce_time=0.05, hold_time=2.0)

def _on_held():
    global _hold_fired
    _hold_fired = True
    _toggle_shutter_set()

button.when_held     = _on_held
button.when_released = _on_released
button.when_pressed  = _on_pressed

# ============================================================
#  DISPLAY + TOUCH
# ============================================================
cv2.namedWindow("Camera", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
cv2.moveWindow("Camera", 0, 0)
cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("Camera", _on_mouse)

# ============================================================
#  MAIN LOOP
# ============================================================
_peak_cache     = None
_peak_frame_idx = 0
_last_tb_state  = None
_cached_tb_img  = None

while True:
    # Power-off hold (10 s)
    if button.is_pressed and press_start_time > 0 and (time.time() - press_start_time) >= 10.0:
        print("[System] Powering off...")
        _canvas.fill(0)
        cv2.putText(_canvas, "Shutting down...", (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Camera", _canvas)
        cv2.waitKey(100)
        os.system("sudo poweroff")
        break

    # Sleep mode
    if sleep_mode:
        _canvas.fill(0)
        cv2.imshow("Camera", _canvas)
        cv2.waitKey(1)
        time.sleep(0.2)
        continue

    # ---- Capture still ----
    if shoot_event.is_set() and not shutter_set_mode:
        shoot_event.clear()
        dt  = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = FILM_PROFILES[current_profile_idx][0].replace(" ", "_")
        try:
            meta    = picam2.capture_metadata()
            iso     = int(meta.get("AnalogueGain", 1) * 100)
            shutter = format_shutter(meta.get("ExposureTime", 0)).replace("/", "_")
            raw     = picam2.switch_mode_and_capture_array(still_config)
            if raw.ndim == 3 and raw.shape[2] == 4:
                raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            elif raw.ndim == 3 and raw.shape[2] == 3:
                pass
            processed = apply_current_profile(raw, preview=False)
            if pro_mist_enabled:
                processed = apply_pro_mist(processed)
            png_path = f"{PICTURES_DIR}/{dt}_{tag}_ISO{iso}_{shutter}.png"
            cv2.imwrite(png_path, processed)
            image_count += 1
            print(f"Captured {png_path}  (#{image_count})")
        except Exception as e:
            print("Capture error:", e); traceback.print_exc()
        finally:
            picam2.switch_mode(preview_config)
            picam2.set_controls({
                "AeMeteringMode": METERING_MODES[current_meter_idx][1],
                "ExposureValue":  EV_OPTIONS[current_ev_idx],
                "AwbMode":        AWB_MODES[current_awb_idx][1]
            })
            _apply_shutter()
            current_zoom_idx = 0

    # ---- Preview frame ----
    frame = picam2.capture_array()
    meta  = picam2.capture_metadata()

    # ---- Digital zoom crop ----
    zoom = zoom_levels[current_zoom_idx]
    if zoom > 1.0:
        fh, fw = frame.shape[:2]
        cx = int(zoom_center[0] * fw); cy = int(zoom_center[1] * fh)
        cw = int(fw / zoom);           ch = int(fh / zoom)
        x1 = max(0, cx - cw//2);       y1 = max(0, cy - ch//2)
        x2 = min(fw, x1 + cw);         y2 = min(fh, y1 + ch)
        frame = frame[y1:y2, x1:x2]

    # ---- Scale to screen ----
    fh, fw = frame.shape[:2]
    s      = min(SCREEN_W / fw, SCREEN_H / fh)
    new_w  = max(1, int(fw * s))
    new_h  = max(1, int(fh * s))
    scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # ---- Film profile (preview=True = fast path) ----
    profiled = apply_current_profile(scaled, preview=True)

    # ---- Focus peaking — update every PEAK_EVERY frames ----
    if focus_peaking_enabled:
        _peak_frame_idx += 1
        if _peak_frame_idx % PEAK_EVERY == 0 or _peak_cache is None:
            _peak_cache = apply_focus_peaking(profiled)
        disp = _peak_cache
    else:
        disp = profiled

    # Gray for histogram
    gray_hist = cv2.cvtColor(profiled, cv2.COLOR_BGR2GRAY)

    # ---- Compose into reused canvas buffer ----
    x_off = (SCREEN_W - new_w) // 2
    y_off = (SCREEN_H - new_h) // 2
    _canvas[y_off:y_off+new_h, x_off:x_off+new_w] = disp

    # Info bar: Fast in-place darkening pass
    bar_y = max(y_off, y_off + new_h - BAR_H)
    _canvas[bar_y:bar_y+BAR_H, x_off:x_off+new_w] = cv2.convertScaleAbs(_canvas[bar_y:bar_y+BAR_H, x_off:x_off+new_w], alpha=0.75)

    hist_w = min(_HIST_W, new_w // 3)
    blit_add(_canvas, draw_histogram(gray_hist, height=BAR_H, width=hist_w), x_off+6, bar_y)

    shutter_us = meta.get("ExposureTime", 0)
    iso_val    = meta.get("AnalogueGain", 0) * 100
    status     = "SET" if shutter_set_mode else "RDY"
    
    # Stateful UI check: Only render new text assets when structural data changes
    current_tb_state = (status, current_shutter_idx, shutter_us, iso_val, image_count)
    if current_tb_state != _last_tb_state:
        _last_tb_state = current_tb_state
        _cached_tb_img = make_text_block([
            f"{status} {SHUTTER_LABELS[current_shutter_idx]}",
            f"{format_shutter(shutter_us)} ISO{int(iso_val)} #{image_count}",
        ], max_h=BAR_H-6)
    
    blit_add(_canvas, _cached_tb_img, x_off+6+hist_w+8, bar_y+(BAR_H-_cached_tb_img.shape[0])//2)

    # ---- Buttons ----
    name, _, accent = FILM_PROFILES[current_profile_idx]
    bx1,by1,bx2,by2 = draw_film_button(_canvas, name, accent, x=x_off+4, y=y_off+4)
    btn_bounds.update({"bx1":bx1,"bx2":bx2,"by1":by1,"by2":by2})

    pm_y = y_off+4+FILM_BTN_H+6
    pm_label = "Pro-Mist: ON" if pro_mist_enabled else "Pro-Mist: OFF"
    pbx1,pby1,pbx2,pby2 = draw_toggle_button(_canvas, pm_label, pro_mist_enabled, x=x_off+4, y=pm_y)
    btn_bounds_pm.update({"bx1":pbx1,"bx2":pbx2,"by1":pby1,"by2":pby2})

    meter_y = pm_y+FILM_BTN_H+6
    meter_name, _ = METERING_MODES[current_meter_idx]
    mbx1,mby1,mbx2,mby2 = draw_toggle_button(_canvas, f"Meter: {meter_name}", True, x=x_off+4, y=meter_y)
    btn_bounds_meter.update({"bx1":mbx1,"bx2":mbx2,"by1":mby1,"by2":mby2})

    ev_y  = meter_y+FILM_BTN_H+6
    ev_val = EV_OPTIONS[current_ev_idx]
    ebx1,eby1,ebx2,eby2 = draw_toggle_button(_canvas, f"EV: {ev_val:+}", True, x=x_off+4, y=ev_y)
    btn_bounds_ev.update({"bx1":ebx1,"bx2":ebx2,"by1":eby1,"by2":eby2})

    awb_y = ev_y+FILM_BTN_H+6
    awb_name, _ = AWB_MODES[current_awb_idx]
    abx1,aby1,abx2,aby2 = draw_toggle_button(_canvas, f"WB: {awb_name}", True, x=x_off+4, y=awb_y)
    btn_bounds_awb.update({"bx1":abx1,"bx2":abx2,"by1":aby1,"by2":aby2})

    cv2.imshow("Camera", _canvas)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
picam2.stop()
