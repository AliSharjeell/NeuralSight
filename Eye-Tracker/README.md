# Webcam Eye Tracker (Calibration + Live Tracking)

This project tracks where you are looking on the screen using your webcam.

## What you get
- `calibrate.py`: collects gaze samples while you look at dots on screen in a fullscreen calibration UI
- `track.py`: predicts gaze in real-time and controls the real cursor by default without taking over the screen

## Setup (Windows)

Use Python 3.9 for best MediaPipe compatibility.

```powershell
cd "C:\Users\pc\Desktop\Codex Workspace\webcam-eye-tracker"
py -3.9 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## 1) Calibrate

```powershell
.\.venv\Scripts\python calibrate.py
```

Controls:
- `SPACE`: start calibration
- `ESC`: cancel/exit

Calibration notes:
- The separate webcam preview window stays hidden during calibration so it does not block the dots.
- Each dot shows a live quality score, and the end screen shows per-dot accuracy plus the overall average.
- The default calibration now reaches closer to the screen edges to improve corner and side coverage.

This saves `calibration_model.pkl`.

## 2) Track

```powershell
.\.venv\Scripts\python track.py
```

Controls:
- `ESC`: quit

Tracking notes:
- `track.py` now stays on your current screen by default and does not open the fullscreen white overlay.
- `track.py` now controls the real Windows cursor by default when you run it.
- The webcam window shows status feedback only; use `--no-control-mouse` if you want preview-only mode instead.
- If you want the old fullscreen overlay back, run `track.py --show-overlay`.

Hybrid tuning:
- `--head-assist 0.0` = pure eye model
- `--head-assist 0.3` to `0.5` = eye + face/head assist (usually smoother)
- `--head-assist 1.0` = pure head/face model

Example:
```powershell
.\.venv\Scripts\python track.py --smooth 0.20 --head-assist 0.35 --sensitivity 0.96 --x-gain 1.06
```

Mouse control example:
```powershell
.\.venv\Scripts\python track.py --snap-clickables --smooth 0.20 --head-assist 0.35 --sensitivity 0.96 --x-gain 1.06 --y-gain 1.15
```

Mouse control flags:
- `--control-mouse` moves the real Windows cursor to the predicted gaze point and is enabled by default
- `--no-control-mouse` keeps tracking in preview-only mode without moving the real cursor
- `--mouth-open-click` uses mouth duration for clicks: short open = single click, long open = double click, and it now defaults to on when real cursor control is on
- `--no-mouth-open-click` disables mouth-based clicking while keeping cursor control active
- `--snap-clickables` latches onto nearby Windows buttons and fields while a mouth click is arming
- `--snap-browse-items` also allows list/tree/data items to snap in dense apps like File Explorer
- `--edge-zone-ratio` and `--edge-boost-px` help the cursor keep reaching true screen edges even when snapping shortens the range
- `--show-overlay` restores the fullscreen gaze overlay if you want the old white-screen tracking view
- `--sensitivity`, `--x-gain`, `--stability-window`, `--steady-radius-px`, and `--response-px` let you trade responsiveness for stability when the point jitters around a fixed gaze
- Re-run `calibrate.py` after updating, because the tracker now uses a stronger calibration fit and improved edge correction

## Tips for better accuracy
- Sit at normal viewing distance and keep posture steady during calibration.
- Use bright, even lighting.
- Re-run calibration if your monitor setup or seating position changes.
