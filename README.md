# Open-Source Film Simulation Capable RPi Camera

Open source Raspberry Pi camera script with film simulation, tap-to-zoom, focus peaking, manual shutter control, and on-screen exposure tuning.

Built for Raspberry Pi 5 with the HQ Camera attachment. This project was inspired by the camera builds and film simulation ideas shared by [Camera Hacks by Malcolm Jay](https://substack.com/@camerahacksbymalcolmjay), [this](https://substack.com/home/post/p-171702270?source=queue) one in particular.

## Features

- Live preview with film simulation profiles
- On-screen controls for film profile, metering, EV, and white balance
- Physical shutter button on GPIO26
- Hold button for shutter set mode, short press to capture or cycle shutter speed
- Tap-to-zoom focus targeting with 1x / 2x / 4x zoom
- Focus peaking overlay for manual framing
- Pro-Mist bloom effect toggle for shoot-mode output
- Capture to `/home/pi/Pictures` (`camera.py` uses PNG; the Pi Zero path uses quality-92 JPEG)

## Film Simulations

The script includes the following film simulation profiles:

- Standard
- Classic Chrome
- Kodak Portra
- Fuji Velvia
- Fuji Astia
- Ilford B&W
- Kodak Gold
- CineStill 800T

## Hardware

- Raspberry Pi 5 (can use previous models as well)
- Official Raspberry Pi HQ Camera (IMX477)
- 6mm M12 mount lens
- Optionally compatible with C mount HQ camera and C mount lenses
- 3.5" GPIO touchscreen display, 480x320 ([Setup Guide](https://www.reddit.com/r/raspberry_pi/comments/1bnav0y/i_finally_have_the_35inch_gpio_spi_lcd_working/))
- Power source (I used a PD compatible power bank)
- Momentary Switch @ GPIO 26 (optional, create a UI shutter button if not using one)
- 3D Printed case for the build (I used https://www.thingiverse.com/thing:6571150 and https://www.thingiverse.com/thing:4878249 and hot glued them together)

## Installation

On Raspberry Pi OS Bookworm 64-bit:

```bash
sudo apt update && sudo apt install -y python3-pip python3-opencv libopencv-dev unclutter
pip3 install picamera2 gpiozero numpy --break-system-packages
```

## Setup

1. Place `camera.py` in `/home/pi` or the desired working directory.
   On an original Pi Zero W, use `camera-pi-zero.py` instead and update the service's `ExecStart` path below to match.
2. Ensure the picture folder exists:

```bash
mkdir -p /home/pi/Pictures
```

3. Create the service file:

```bash
sudo nano /etc/systemd/system/camera.service
```

Paste this content:

```ini
[Unit]
Description=HQ Camera
After=multi-user.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/camera.py
WorkingDirectory=/home/pi
StandardOutput=journal
StandardError=journal
Restart=on-failure
RestartSec=5
User=pi
Environment=DISPLAY=:0
#Environment=SDL_VIDEODRIVER=fbcon
#Environment=SDL_FBDEV=/dev/fb1
Environment=XDG_RUNTIME_DIR=/run/user/1000
Environment=WAYLAND_DISPLAY=wayland-1
Environment=DISPLAY=:0
Environment=LIBGL_ALWAYS_SOFTWARE=1

[Install]
WantedBy=multi-user.target
```

4. Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable camera.service
sudo systemctl start camera.service
```

## Useful Commands

```bash
sudo systemctl status camera.service
sudo journalctl -u camera.service -f
sudo systemctl restart camera.service
sudo systemctl stop camera.service
```

## Usage

- Tap the on-screen `FILM` button to cycle film profiles.
- Tap the `Meter`, `EV`, and `WB` buttons to cycle metering, exposure compensation, and white balance.
- In `camera-pi-zero.py`, tap `Photo: 12MP/3MP` to select the next still resolution; 12 MP is the default.
- Tap the screen outside the UI to change the zoom anchor point and zoom level.
- Hold the GPIO26 button for shutter-set mode, then tap to cycle shutter speed.
- Short press the GPIO26 button to capture an image.

## Notes

The scripts write captures to `/home/pi/Pictures` with timestamped filenames including the selected film profile, ISO, and shutter speed. The Pi Zero variant logs preview timing about every five seconds and prints acquisition, processing, and JPEG timing for each capture.

Sample images and camera photos are included in the repository.

## Experimental Pi Zero console/display setup

`camera-pi-zero.py` is an experimental path for the original single-core Zero W. It keeps the existing OpenCV fullscreen window on the Wayland desktop and depends on the SPI display's kernel/DRM driver being installed and exposing that display to the compositor. The physical SPI refresh rate can remain the visible frame-rate ceiling even when the script's processing log reports a higher rate.

Useful device checks are:

```bash
cat /proc/fb
ls -l /dev/fb* /dev/dri/card* /dev/dri/renderD*
ls -l /sys/class/drm/
for status in /sys/class/drm/card*-*/status; do printf '%s: ' "$status"; cat "$status"; done
```

If the SPI panel is absent from these results, fix its overlay/driver setup before debugging the Python window. The exact overlay and device name depend on the panel vendor; use the display's setup guide and verify it after reboot.

The service environment shown above does not create a console-only renderer. Variables such as `DISPLAY`, `WAYLAND_DISPLAY`, `SDL_VIDEODRIVER`, or `SDL_FBDEV` do not make an OpenCV HighGUI window write directly to a framebuffer. A true no-desktop implementation would need a separate direct framebuffer or DRM/KMS renderer plus touch input read and calibrated through `evdev`; that architecture is outside the current script.
