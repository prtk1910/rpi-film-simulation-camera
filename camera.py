#!/usr/bin/env python3
"""
Raspberry Pi HQ Camera Script
- Sensor:  Official Raspberry Pi HQ Camera (IMX477)
- Display: 3.5" GPIO touchscreen, 480x320
- Power:   USB-C (no battery monitoring)
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
#  Configuration and state
# ------------------------------------------------------------
PICTURES_DIR = "/home/pi/Pictures"

SCREEN_W, SCREEN_H = 480, 320
BAR_H = 40
FILM_BTN_W = 180
FILM_BTN_H = 42

EV_OPTIONS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
current_ev_idx = 4

AWB_MODES = [("Auto", 0), ("Daylight", 5), ("Cloudy", 6), ("Tungsten", 2), ("Fluorescent", 3)]
current_awb_idx = 0

zoom_levels = [1.0, 2.0, 4.0]
current_zoom_idx = 0
zoom_center = (0.5, 0.5)

sleep_mode = False
press_start_time = 0.0
image_count = 0
focus_peaking_enabled = True

grain_cache = None

os.system("unclutter &")
os.makedirs(PICTURES_DIR, exist_ok=True)

btn_bounds_ev = {"bx1": 0, "bx2": 0, "by1": 0, "by2": 0}
btn_bounds_awb = {"bx1": 0, "bx2": 0, "by1": 0, "by2": 0}

def enter_sleep_mode():
    global sleep_mode
    sleep_mode = True

def wake_display():
    global sleep_mode
    if sleep_mode:
        print("[Sleep] Wake")
        sleep_mode = False

_just_woke = False

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
#  FILM PROFILES  (all operate on BGR uint8 arrays)
# ============================================================

def _lut_from_curve(pts):
    xs = np.array([p[0] for p in pts], dtype=np.float32)
    ys = np.array([p[1] for p in pts], dtype=np.float32)
    return np.clip(np.interp(np.arange(256), xs, ys), 0, 255).astype(np.uint8)

def _apply_channel_luts(img, lb, lg, lr):
    b, g, r = cv2.split(img)
    return cv2.merge([cv2.LUT(b, lb), cv2.LUT(g, lg), cv2.LUT(r, lr)])

def _sat(img, s):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * s, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def _hue(img, shift):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:,:,0] = (hsv[:,:,0] + int(shift)) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def _vignette(img, strength=0.35):
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((X - w/2)/(w/2))**2 + ((Y - h/2)/(h/2))**2)
    mask = 1.0 - np.clip(dist * strength, 0, 1)
    return np.clip(img.astype(np.float32) * mask[:,:,np.newaxis], 0, 255).astype(np.uint8)

def _grain(img, amount=6):
    global grain_cache

    h, w = img.shape[:2]

    # Generate once or resize if needed
    if grain_cache is None or grain_cache.shape[:2] != (h, w):
        grain_cache = np.random.normal(0, amount, (h, w, 3)).astype(np.float32)

    return np.clip(img.astype(np.float32) + grain_cache, 0, 255).astype(np.uint8)

def profile_standard(img):
    return img.copy()

def profile_classic_chrome(img):
    img = _apply_channel_luts(img,
        _lut_from_curve([(0,25),(64,80),(128,132),(192,188),(255,242)]),
        _lut_from_curve([(0,18),(64,72),(128,126),(192,184),(255,238)]),
        _lut_from_curve([(0,20),(64,75),(128,128),(192,185),(255,240)]))
    return _sat(img, 0.72)

def profile_kodak_portra(img):
    img = _apply_channel_luts(img,
        _lut_from_curve([(0,12),(64,68),(128,118),(192,172),(255,220)]),
        _lut_from_curve([(0, 8),(64,74),(128,130),(192,192),(255,248)]),
        _lut_from_curve([(0,10),(64,80),(128,138),(192,200),(255,255)]))
    return _vignette(_sat(img, 0.85), 0.25)

def profile_fuji_velvia(img):
    img = _apply_channel_luts(img,
        _lut_from_curve([(0,0),(64,66),(128,140),(192,208),(255,255)]),
        _lut_from_curve([(0,0),(64,62),(128,135),(192,205),(255,255)]),
        _lut_from_curve([(0,0),(64,60),(128,130),(192,200),(255,255)]))
    return _hue(_sat(img, 1.45), -3)

def profile_fuji_astia(img):
    img = _apply_channel_luts(img,
        _lut_from_curve([(0,8),(64,70),(128,126),(192,188),(255,245)]),
        _lut_from_curve([(0,4),(64,70),(128,130),(192,194),(255,250)]),
        _lut_from_curve([(0,5),(64,72),(128,132),(192,196),(255,252)]))
    return _sat(img, 0.95)

def profile_ilford_bw(img):
    b, g, r = cv2.split(img)
    pan = np.clip(0.21*r.astype(np.float32)
                + 0.72*g.astype(np.float32)
                + 0.07*b.astype(np.float32), 0, 255).astype(np.uint8)
    pan = cv2.LUT(pan, _lut_from_curve([(0,0),(60,50),(128,128),(190,210),(255,255)]))
    return _grain(cv2.cvtColor(pan, cv2.COLOR_GRAY2BGR), amount=5)

def profile_kodak_gold(img):
    img = _apply_channel_luts(img,
        _lut_from_curve([(0,20),(64,60),(128,110),(192,162),(255,210)]),
        _lut_from_curve([(0,10),(64,76),(128,132),(192,194),(255,248)]),
        _lut_from_curve([(0,15),(64,85),(128,142),(192,205),(255,255)]))
    return _grain(_vignette(_sat(img, 0.90), 0.30), amount=4)

def profile_cinestill_800t(img):
    img = _apply_channel_luts(img,
        _lut_from_curve([(0,30),(64,88),(128,142),(192,198),(255,248)]),
        _lut_from_curve([(0,10),(64,68),(128,128),(192,188),(255,240)]),
        _lut_from_curve([(0,10),(64,72),(128,135),(192,200),(255,250)]))

    img = _grain(_vignette(_hue(_sat(img, 1.05), 4), 0.40), amount=8)

    # 🔥 Add halation HERE
    img = apply_halation(img)

    return img

def apply_pro_mist(img, threshold=190, glow_spread=15, blend=0.25):
    """
    Simulates a Black Pro-Mist filter: blooms highlights and lowers digital sharpness.
    Optimized for Raspberry Pi by blurring at a 1/4th resolution scale.
    """
    # 1. Isolate the brightest parts of the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    highlights = cv2.bitwise_and(img, img, mask=mask)
    
    # 2. Performance Hack: Downscale -> Blur -> Upscale
    h, w = img.shape[:2]
    scale_factor = 4 # Shrink by 4x for speed
    small_highlights = cv2.resize(highlights, (w // scale_factor, h // scale_factor), interpolation=cv2.INTER_LINEAR)
    
    # Ensure the blur kernel size is an odd number (required by OpenCV)
    if glow_spread % 2 == 0: 
        glow_spread += 1
        
    blurred_small = cv2.GaussianBlur(small_highlights, (glow_spread, glow_spread), 0)
    blurred_highlights = cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 3. Blend the glowing highlights back over the original image
    out = cv2.addWeighted(img, 1.0, blurred_highlights, blend, 0)
    
    return out
    
def apply_halation(img, threshold=200, blur_size=21, strength=0.4):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    r = img[:,:,2]
    glow = cv2.GaussianBlur(r, (blur_size, blur_size), 0)

    halated = r.copy()
    halated[mask > 0] = cv2.addWeighted(r, 1.0, glow, strength, 0)[mask > 0]

    out = img.copy()
    out[:,:,2] = halated
    return out

# Pro-Mist State
pro_mist_enabled = False
btn_bounds_pm = {"bx1": 0, "bx2": 0, "by1": 0, "by2": 0}

def toggle_pro_mist():
    global pro_mist_enabled
    pro_mist_enabled = not pro_mist_enabled
    print(f"[Pro-Mist] {'ON' if pro_mist_enabled else 'OFF'}")


# Metering Modes: (Display Name, libcamera Enum Value)
METERING_MODES = [("Matrix", 2), ("Center", 0), ("Spot", 1)]
current_meter_idx = 1
btn_bounds_meter = {"bx1": 0, "bx2": 0, "by1": 0, "by2": 0}

def cycle_metering():
    global current_meter_idx
    current_meter_idx = (current_meter_idx + 1) % len(METERING_MODES)
    name, val = METERING_MODES[current_meter_idx]
    
    # Immediately update the camera sensor's exposure algorithm
    picam2.set_controls({"AeMeteringMode": val})
    print(f"[Metering] {name}")

# (name, function, accent_BGR)
FILM_PROFILES = [
    ("Standard",       profile_standard,      (180, 180, 180)),
    ("Classic Chrome", profile_classic_chrome, ( 80, 180, 160)),
    ("Kodak Portra",   profile_kodak_portra,   ( 40, 140, 230)),
    ("Fuji Velvia",    profile_fuji_velvia,    ( 30, 200,  90)),
    ("Fuji Astia",     profile_fuji_astia,     (200, 160,  80)),
    ("Ilford B&W",     profile_ilford_bw,      (210, 210, 210)),
    ("Kodak Gold",     profile_kodak_gold,     (  0, 190, 230)),
    ("CineStill 800T", profile_cinestill_800t, (  0, 100, 220)),
]
current_profile_idx = 0

def apply_current_profile(img):
    return FILM_PROFILES[current_profile_idx][1](img)

def cycle_film_profile():
    global current_profile_idx
    current_profile_idx = (current_profile_idx + 1) % len(FILM_PROFILES)
    print(f"[Film Profile] {FILM_PROFILES[current_profile_idx][0]}")

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

def handle_focus_tap(x, y):
    global current_zoom_idx, zoom_center

    # Convert to normalized coordinates
    zoom_center = (x / SCREEN_W, y / SCREEN_H)

    # Cycle zoom
    current_zoom_idx = (current_zoom_idx + 1) % len(zoom_levels)

    print(f"[Zoom] {zoom_levels[current_zoom_idx]}x @ {zoom_center}")

# ============================================================
#  TOUCHSCREEN
# ============================================================
_touch_lock    = threading.Lock()
_last_tap_time = 0.0
btn_bounds     = {"bx1": 0, "bx2": FILM_BTN_W, "by1": 0, "by2": FILM_BTN_H}

def _on_mouse(event, x, y, flags, param):
    global _last_tap_time
    if event == cv2.EVENT_LBUTTONDOWN:
        now = time.time()
        with _touch_lock:
            if now - _last_tap_time < 0.35:
                return
            _last_tap_time = now
            
        # 1. Check if Film Button was tapped
        if btn_bounds["bx1"] <= x <= btn_bounds["bx2"] and btn_bounds["by1"] <= y <= btn_bounds["by2"]:
            cycle_film_profile()
            
        elif btn_bounds_pm["bx1"] <= x <= btn_bounds_pm["bx2"] and btn_bounds_pm["by1"] <= y <= btn_bounds_pm["by2"]:
            toggle_pro_mist()

        # 3. Check if Metering Button was tapped
        elif btn_bounds_meter["bx1"] <= x <= btn_bounds_meter["bx2"] and btn_bounds_meter["by1"] <= y <= btn_bounds_meter["by2"]:
            cycle_metering()
        
        elif btn_bounds_ev["bx1"] <= x <= btn_bounds_ev["bx2"] and btn_bounds_ev["by1"] <= y <= btn_bounds_ev["by2"]:
            cycle_ev()

        elif btn_bounds_awb["bx1"] <= x <= btn_bounds_awb["bx2"] and btn_bounds_awb["by1"] <= y <= btn_bounds_awb["by2"]:
            cycle_awb()
        
        else:
            handle_focus_tap(x, y)

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
    sx1=x1-x; sy1=y1-y
    roi = dst[y1:y2, x1:x2]
    src_c = src[sy1:sy1+(y2-y1), sx1:sx1+(x2-x1)]
    ch = roi.shape[2] if roi.ndim==3 else 3
    cv2.add(ensure_channels(roi, ch), ensure_channels(src_c, ch), dst=roi)
    dst[y1:y2, x1:x2] = roi

def overlay_translucent(dst, overlay, x, y, alpha=0.15):
    dh, dw = dst.shape[:2]; oh, ow = overlay.shape[:2]
    if x<0 or y<0 or x+ow>dw or y+oh>dh: return dst
    ch = dst.shape[2] if dst.ndim==3 else 3
    roi = ensure_channels(dst[y:y+oh, x:x+ow], ch)
    dst[y:y+oh, x:x+ow] = cv2.addWeighted(roi, 1-alpha,
                                            ensure_channels(overlay, ch), alpha, 0)
    return dst

def format_shutter(us):
    return f"1/{int(round(1e6/us))}s" if us and us > 0 else "Auto"

def apply_focus_peaking_prominent(frame_bgr):
    """
    Highlight sharp edges in green. Moderate prominence:
    - threshold at mean + 1.5*std (catches real edges, skips noise)
    - 3x3 open to clean speckles
    - 65/35 blend so peaking is clearly visible but not overwhelming
    """
    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    lap     = cv2.Laplacian(blurred, cv2.CV_64F)
    lap     = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thr     = max(30, int(np.mean(lap) + 1.5 * np.std(lap)))
    _, mask = cv2.threshold(lap, thr, 255, cv2.THRESH_BINARY)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    overlay = frame_bgr.copy()
    overlay[mask > 0] = (0, 255, 0)   # bright green on sharp edges
    out     = cv2.addWeighted(frame_bgr, 0.65, overlay, 0.35, 0)
    return out, gray

def apply_focus_peaking_medium(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Bilateral filter: removes noise but strictly preserves hard edges
    blurred = cv2.bilateralFilter(gray, 5, 50, 50)
    
    # 2. Sobel operators: better directional edge detection than Laplacian
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    
    # Normalize to 0-255
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 3. Percentile thresholding: strictly only peak the top 3% of sharpest edges
    # (Adjust 97 to a lower number like 90 if you want more peaking)
    thr = np.percentile(magnitude, 97) 
    _, mask = cv2.threshold(magnitude, thr, 255, cv2.THRESH_BINARY)
    
    # Clean up isolated speckles
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    overlay = frame_bgr.copy()
    overlay[mask > 0] = (0, 255, 0)
    
    out = cv2.addWeighted(frame_bgr, 0.65, overlay, 0.35, 0)
    return out, gray

def apply_focus_peaking(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # FAST blur instead of bilateral
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Sobel gradients
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Slightly more permissive threshold (faster + clearer)
    thr = np.percentile(magnitude, 95)
    _, mask = cv2.threshold(magnitude, thr, 255, cv2.THRESH_BINARY)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    overlay = frame_bgr.copy()
    overlay[mask > 0] = (0, 255, 0)
    
    out = cv2.addWeighted(frame_bgr, 0.7, overlay, 0.3, 0)
    return out, gray

def fix_chromatic_aberration(img, r_scale=0.998, b_scale=1.002):
    """Slightly shrinks the red channel and expands the blue channel."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    b, g, r = cv2.split(img)
    
    # Transformation matrices for scaling from the center
    M_r = cv2.getRotationMatrix2D((cx, cy), 0, r_scale)
    M_b = cv2.getRotationMatrix2D((cx, cy), 0, b_scale)
    
    # Warp the R and B channels
    r_corrected = cv2.warpAffine(r, M_r, (w, h), flags=cv2.INTER_LINEAR)
    b_corrected = cv2.warpAffine(b, M_b, (w, h), flags=cv2.INTER_LINEAR)
    
    return cv2.merge([b_corrected, g, r_corrected])

def draw_histogram(gray, height=BAR_H, width=128):
    """Bars grow upward from the bottom of the bar area."""
    hist    = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    bin_sz  = max(1, 256 // width)
    comp    = np.array([float(np.mean(hist[i*bin_sz:(i+1)*bin_sz])) for i in range(width)])
    disp_max = max(50.0, min(float(np.percentile(comp, 99)), 5000.0))
    img     = np.zeros((height, width, 3), dtype=np.uint8)
    scale   = (height - 4) / disp_max
    bars    = np.minimum((comp * scale).astype(np.int32), height - 4)
    for x, bar in enumerate(bars):
        if bar:
            # draw from the BOTTOM of the image upward — no flip needed
            cv2.line(img, (x, height - 1), (x, height - 1 - bar), (200, 200, 200), 1)
    return img

def make_text_block(lines, font_scale=0.42, thickness=1, max_h=BAR_H-6):
    """Render text lines into a small BGR image (right-side up, no rotation)."""
    font   = cv2.FONT_HERSHEY_SIMPLEX
    sizes  = [cv2.getTextSize(l, font, font_scale, thickness)[0] for l in lines]
    w      = max(s[0] for s in sizes) + 8
    line_h = max(s[1] for s in sizes) + 5
    h      = line_h * len(lines) + 6
    img    = np.zeros((h, w, 3), dtype=np.uint8)
    y      = 3 + sizes[0][1]
    for l in lines:
        cv2.putText(img, l, (4, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_h
    if img.shape[0] > max_h:
        sc  = max_h / img.shape[0]
        img = cv2.resize(img, (max(1, int(img.shape[1]*sc)), max_h), cv2.INTER_LINEAR)
    return img

def draw_film_button(canvas, name, accent_bgr, x, y):
    """Draw the film profile button. Returns its bounding box."""
    w, h = FILM_BTN_W, FILM_BTN_H
    # Clamp to canvas
    ch, cw = canvas.shape[:2]
    if x + w > cw: w = cw - x
    if y + h > ch: h = ch - y
    if w <= 0 or h <= 0: return (x, y, x, y)

    # Semi-transparent dark background
    roi  = canvas[y:y+h, x:x+w]
    dark = np.full_like(roi, 15)
    canvas[y:y+h, x:x+w] = cv2.addWeighted(roi, 0.40, dark, 0.60, 0)

    # Coloured border (2px for visibility)
    cv2.rectangle(canvas, (x, y), (x+w-1, y+h-1), accent_bgr, 2)

    # Film-strip perforations along left edge
    perf_w, perf_h = 6, 5
    n_perfs = max(2, h // 12)
    spacing = h // (n_perfs + 1)
    for i in range(n_perfs):
        fy = y + spacing * (i + 1) - perf_h // 2
        cv2.rectangle(canvas, (x+4, fy), (x+4+perf_w, fy+perf_h), accent_bgr, -1)

    # Profile name — larger font to fill the button
    font  = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.50
    label = name[:16]
    (tw, th), _ = cv2.getTextSize(label, font, fscale, 1)
    tx = x + 4 + perf_w + 8
    ty = y + (h + th) // 2
    cv2.putText(canvas, label, (tx, ty), font, fscale, (255, 255, 255), 1, cv2.LINE_AA)

    return (x, y, x+w, y+h)

def draw_toggle_button(canvas, label, is_active, x, y):
    """Draws a standard toggle button."""
    w, h = FILM_BTN_W, FILM_BTN_H
    ch, cw = canvas.shape[:2]
    if x + w > cw: w = cw - x
    if y + h > ch: h = ch - y
    if w <= 0 or h <= 0: return (x, y, x, y)

    roi = canvas[y:y+h, x:x+w]
    dark = np.full_like(roi, 15)
    canvas[y:y+h, x:x+w] = cv2.addWeighted(roi, 0.40, dark, 0.60, 0)

    # Green border if ON, Gray border if OFF
    accent_bgr = (0, 200, 0) if is_active else (100, 100, 100)
    cv2.rectangle(canvas, (x, y), (x+w-1, y+h-1), accent_bgr, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.50
    (tw, th), _ = cv2.getTextSize(label, font, fscale, 1)
    tx = x + (w - tw) // 2  # Center text
    ty = y + (h + th) // 2
    
    # White text if ON, Gray text if OFF
    text_color = (255, 255, 255) if is_active else (150, 150, 150)
    cv2.putText(canvas, label, (tx, ty), font, fscale, text_color, 1, cv2.LINE_AA)

    return (x, y, x+w, y+h)

# ============================================================
#  CAMERA SETUP  (RPi HQ Camera — IMX477)
# ============================================================
picam2 = Picamera2()

FULL_W, FULL_H        = 4056, 3040
preview_size          = (1920, 1280)
DEFAULT_FRAME_LIMITS  = (125, 16667)   # ~1/8000s – ~1/60s

preview_config = picam2.create_preview_configuration(
    main={"size": preview_size, "format": "RGB888"},
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
#  SHUTTER CONTROL  (GPIO26)
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
        # FULL AUTO
        picam2.set_controls({
            "AeEnable": True,
            "FrameDurationLimits": DEFAULT_FRAME_LIMITS
        })
    else:
        # MANUAL SHUTTER
        picam2.set_controls({
            "AeEnable": False,
            "ExposureTime": int(us),
            "FrameDurationLimits": (int(us), int(us))
        })

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
button.when_pressed = _on_pressed

# ============================================================
#  DISPLAY + TOUCH
#  Headless framebuffer: export SDL_VIDEODRIVER=fbcon SDL_FBDEV=/dev/fb1
# ============================================================
cv2.namedWindow("Camera", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
cv2.moveWindow("Camera", 0, 0)
cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("Camera", _on_mouse, btn_bounds)

# ============================================================
#  MAIN LOOP
# ============================================================
canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
while True:

    if sleep_mode:
        canvas[:] = 0
        cv2.imshow("Camera", canvas)
        cv2.waitKey(1)
        time.sleep(0.2)
        continue

    # --- Capture ---
    if shoot_event.is_set() and not shutter_set_mode:
        shoot_event.clear()

        dt  = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = FILM_PROFILES[current_profile_idx][0].replace(" ", "_")

        try:
            # Metadata
            meta = picam2.capture_metadata()
            iso = int(meta.get("AnalogueGain", 1) * 100)
            shutter = format_shutter(meta.get("ExposureTime", 0)).replace("/", "_")

            raw = picam2.switch_mode_and_capture_array(still_config)

            # Convert to BGR
            if raw.ndim == 3 and raw.shape[2] == 4:
                raw = cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR)
            elif raw.ndim == 3 and raw.shape[2] == 3:
                raw = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

            processed = apply_current_profile(raw)

            if pro_mist_enabled:
                processed = apply_pro_mist(processed)

            png_path = f"{PICTURES_DIR}/{dt}_{tag}_ISO{iso}_{shutter}.png"

            cv2.imwrite(png_path, processed)

            image_count += 1
            print(f"Captured {png_path}  (#{image_count})")

        except Exception as e:
            print("Capture error:", e)
            traceback.print_exc()

        finally:
            # Fast return to preview
            picam2.switch_mode(preview_config)

            picam2.set_controls({
                "AeMeteringMode": METERING_MODES[current_meter_idx][1],
                "ExposureValue":  EV_OPTIONS[current_ev_idx],
                "AwbMode":        AWB_MODES[current_awb_idx][1]
            })

            _apply_shutter()

            # Reset zoom after capture
            current_zoom_idx = 0

    # --- Preview ---
    frame = picam2.capture_array()
    # --- Apply zoom (tap-to-zoom) ---
    zoom = zoom_levels[current_zoom_idx]

    if zoom > 1.0:
        fh, fw = frame.shape[:2]

        # Center in pixels
        cx = int(zoom_center[0] * fw)
        cy = int(zoom_center[1] * fh)

        # Crop size
        crop_w = int(fw / zoom)
        crop_h = int(fh / zoom)

        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(fw, x1 + crop_w)
        y2 = min(fh, y1 + crop_h)

        frame = frame[y1:y2, x1:x2]
    
    meta  = picam2.capture_metadata()

    # picamera2 always returns RGB888 — convert to BGR once, up front
    if   frame.ndim == 3 and frame.shape[2] == 4: frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 3: frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # frame is now BGR for all downstream work

    fh, fw = frame.shape[:2]
    s      = min(SCREEN_W / fw, SCREEN_H / fh)
    new_w  = max(1, int(fw * s))
    new_h  = max(1, int(fh * s))
    scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Apply film profile (BGR in → BGR out)
    profiled = apply_current_profile(scaled)
    
    # Apply Pro-Mist to live preview if enabled. Disable in live view for speed.
    # if pro_mist_enabled:
    #     profiled = apply_pro_mist(profiled)
    # Focus peaking (BGR in → BGR out)
    if focus_peaking_enabled:
        disp, gray_hist = apply_focus_peaking(profiled)
    else:
        disp      = profiled
        gray_hist = cv2.cvtColor(profiled, cv2.COLOR_BGR2GRAY)

    # Compose canvas
    canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    x_off  = (SCREEN_W - new_w) // 2
    y_off  = (SCREEN_H - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = disp

    # Info bar (bottom strip)
    bar_y = y_off + new_h - BAR_H
    bar_y = max(bar_y, y_off)
    canvas = overlay_translucent(canvas,
                                  np.zeros((BAR_H, new_w, 3), dtype=np.uint8),
                                  x_off, bar_y, alpha=0.25)

    hist_w = min(128, new_w // 3)
    blit_add(canvas,
             draw_histogram(gray_hist, height=BAR_H, width=hist_w),
             x_off + 6, bar_y)

    shutter_us = meta.get("ExposureTime", 0)
    iso        = meta.get("AnalogueGain", 0) * 100
    status     = "SET" if shutter_set_mode else "RDY"
    tb = make_text_block([
        f"{status} {SHUTTER_LABELS[current_shutter_idx]}",
        f"{format_shutter(shutter_us)} ISO{int(iso)} #{image_count}",
    ], max_h=BAR_H - 6)
    blit_add(canvas, tb,
             x_off + 6 + hist_w + 8,
             bar_y + (BAR_H - tb.shape[0]) // 2)

# 1. Film profile button (top-left)
    name, _, accent = FILM_PROFILES[current_profile_idx]
    bx1, by1, bx2, by2 = draw_film_button(canvas, name, accent, x=x_off + 4, y=y_off + 4)
    btn_bounds["bx1"] = bx1;  btn_bounds["bx2"] = bx2
    btn_bounds["by1"] = by1;  btn_bounds["by2"] = by2

    # 2. Pro-Mist button (stacked below the film button)
    pm_y = y_off + 4 + FILM_BTN_H + 6  # 6px gap between buttons
    pm_label = "Pro-Mist: ON" if pro_mist_enabled else "Pro-Mist: OFF"
    pbx1, pby1, pbx2, pby2 = draw_toggle_button(canvas, pm_label, pro_mist_enabled, x=x_off + 4, y=pm_y)
    btn_bounds_pm["bx1"] = pbx1;  btn_bounds_pm["bx2"] = pbx2
    btn_bounds_pm["by1"] = pby1;  btn_bounds_pm["by2"] = pby2

    # 3. Metering Mode button (stacked below Pro-Mist)
    meter_y = pm_y + FILM_BTN_H + 6
    meter_name, _ = METERING_MODES[current_meter_idx]
    # We pass 'True' for is_active just so it draws with the green/active border style
    mbx1, mby1, mbx2, mby2 = draw_toggle_button(canvas, f"Meter: {meter_name}", True, x=x_off + 4, y=meter_y)
    btn_bounds_meter["bx1"] = mbx1;  btn_bounds_meter["bx2"] = mbx2
    btn_bounds_meter["by1"] = mby1;  btn_bounds_meter["by2"] = mby2

    # 4. EV Button (Stacked below Meter)
    ev_y = meter_y + FILM_BTN_H + 6
    ev_val = EV_OPTIONS[current_ev_idx]
    ebx1, eby1, ebx2, eby2 = draw_toggle_button(canvas, f"EV: {ev_val:+}", True, x=x_off + 4, y=ev_y)
    btn_bounds_ev.update({"bx1": ebx1, "by1": eby1, "bx2": ebx2, "by2": eby2})

    # 5. AWB Button (Stacked below EV)
    awb_y = ev_y + FILM_BTN_H + 6
    awb_name, _ = AWB_MODES[current_awb_idx]
    abx1, aby1, abx2, aby2 = draw_toggle_button(canvas, f"WB: {awb_name}", True, x=x_off + 4, y=awb_y)
    btn_bounds_awb.update({"bx1": abx1, "by1": aby1, "bx2": abx2, "by2": aby2})

    cv2.imshow("Camera", cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(1) == 27:   # ESC to quit
        break

cv2.destroyAllWindows()
picam2.stop()
