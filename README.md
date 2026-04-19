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
- Capture to PNG files in `/home/pi/Pictures`

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
- 3.5" GPIO touchscreen display, 480x320
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
Environment=SDL_VIDEODRIVER=fbcon
Environment=SDL_FBDEV=/dev/fb1

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
- Tap the screen outside the UI to change the zoom anchor point and zoom level.
- Hold the GPIO26 button for shutter-set mode, then tap to cycle shutter speed.
- Short press the GPIO26 button to capture an image.

## Notes

The script writes captures to `/home/pi/Pictures` with timestamped filenames including the selected film profile, ISO, and shutter speed.

Sample images and camera photos are included in the repository.
