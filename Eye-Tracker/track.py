import argparse
import ctypes
import pickle
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.gaze_core import build_face_mesh, draw_status, extract_features, get_screen_size, split_feature_vector
from src.model_utils import apply_calibration_correction, predict_xy


MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800


def enable_dpi_awareness():
    """Use real desktop pixels so SetCursorPos can reach the full screen."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def apply_model_correction(x: float, y: float, corr):
    corrected = apply_calibration_correction(np.array([x, y], dtype=np.float32), corr)
    return float(corrected[0]), float(corrected[1])


def clamp_xy(x: float, y: float, screen_w: int, screen_h: int):
    return int(np.clip(x, 0, screen_w - 1)), int(np.clip(y, 0, screen_h - 1))


def draw_gaze_minimap(frame: np.ndarray, gaze_x: int, gaze_y: int, screen_w: int, screen_h: int, active: bool, mouse_mode: bool):
    frame_h, frame_w = frame.shape[:2]
    outer_pad = 16
    header_h = 28
    inner_pad = 12
    max_map_w = min(240, max(120, frame_w - 2 * outer_pad - 24))
    max_map_h = min(150, max(90, frame_h - 2 * outer_pad - header_h - 24))
    scale = min(max_map_w / max(float(screen_w), 1.0), max_map_h / max(float(screen_h), 1.0))
    map_w = max(120, int(round(screen_w * scale)))
    map_h = max(70, int(round(screen_h * scale)))
    panel_w = map_w + inner_pad * 2
    panel_h = map_h + header_h + inner_pad * 2
    panel_x1 = frame_w - panel_w - outer_pad
    panel_y1 = outer_pad
    panel_x2 = panel_x1 + panel_w
    panel_y2 = panel_y1 + panel_h
    map_x1 = panel_x1 + inner_pad
    map_y1 = panel_y1 + header_h
    map_x2 = map_x1 + map_w
    map_y2 = map_y1 + map_h

    # Zinc 900 Panel
    cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (27, 24, 24), -1)
    # Zinc 800 Border
    cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (42, 39, 39), 2)
    # White inner map
    cv2.rectangle(frame, (map_x1, map_y1), (map_x2, map_y2), (250, 250, 250), -1)
    # Zinc 300 map border
    cv2.rectangle(frame, (map_x1, map_y1), (map_x2, map_y2), (212, 212, 216), 1)

    cv2.putText(frame, "NeuralSight Head", (panel_x1 + 12, panel_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (250, 250, 250), 1, cv2.LINE_AA)
    mode_text = "Mouse ON" if mouse_mode else "Preview"
    cv2.putText(frame, mode_text, (panel_x2 - 96, panel_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (161, 161, 170), 1, cv2.LINE_AA)

    center_x = map_x1 + map_w // 2
    center_y = map_y1 + map_h // 2
    cv2.line(frame, (center_x, map_y1 + 6), (center_x, map_y2 - 6), (228, 228, 231), 1)
    cv2.line(frame, (map_x1 + 6, center_y), (map_x2 - 6, center_y), (228, 228, 231), 1)

    if mouse_mode:
        status_text = "Controlling real cursor" if active else "Waiting for face"
        cv2.putText(frame, status_text, (panel_x1 + 12, panel_y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (212, 212, 216), 1, cv2.LINE_AA)
        cx = map_x1 + map_w // 2
        cy = map_y1 + map_h // 2
        cv2.arrowedLine(frame, (cx - 26, cy + 18), (cx + 18, cy - 16), (27, 24, 24), 3, cv2.LINE_AA, tipLength=0.28)
        cv2.circle(frame, (cx + 18, cy - 16), 5, (246, 130, 59), -1)
    else:
        dot_x = map_x1 + int(round(np.clip(gaze_x, 0, screen_w - 1) * (map_w - 1) / max(screen_w - 1, 1)))
        dot_y = map_y1 + int(round(np.clip(gaze_y, 0, screen_h - 1) * (map_h - 1) / max(screen_h - 1, 1)))
        dot_fill = (59, 130, 246) if active else (113, 113, 122)
        dot_ring = (37, 99, 235) if active else (82, 82, 91)
        cv2.circle(frame, (dot_x, dot_y), 7, dot_fill, -1)
        cv2.circle(frame, (dot_x, dot_y), 13, dot_ring, 2)
        status_text = "Preview dot active" if active else "Waiting for face"
        cv2.putText(frame, status_text, (panel_x1 + 12, panel_y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (212, 212, 216), 1, cv2.LINE_AA)


def apply_edge_assist_1d(value: float, size: int, zone_ratio: float, max_push_px: float):
    zone = max(1.0, float(size) * float(zone_ratio))
    pushed = float(value)
    strength = 0.0

    if pushed < zone:
        t = 1.0 - (pushed / zone)
        pushed -= float(max_push_px) * (t * t)
        strength = float(np.clip(t, 0.0, 1.0))
    elif pushed > float(size) - zone:
        t = (pushed - (float(size) - zone)) / zone
        pushed += float(max_push_px) * (t * t)
        strength = float(np.clip(t, 0.0, 1.0))

    return pushed, strength


def apply_edge_assist(x: float, y: float, screen_w: int, screen_h: int, zone_ratio: float, max_push_px: float):
    x2, edge_x = apply_edge_assist_1d(x, screen_w, zone_ratio, max_push_px)
    y2, edge_y = apply_edge_assist_1d(y, screen_h, zone_ratio, max_push_px)

    if edge_x > edge_y * 1.75:
        y2 = float(y + 0.25 * (y2 - y))
    elif edge_y > edge_x * 1.75:
        x2 = float(x + 0.25 * (x2 - x))

    return x2, y2, (edge_x > 0.0 or edge_y > 0.0)


class DirectFilter:
    """Zero-lag filter: median for outlier rejection + deadzone. No EMA."""

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        deadzone_px: float,
        median_window: int,
        initial_x: float,
        initial_y: float,
    ):
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)
        self.deadzone_px = max(0.0, float(deadzone_px))
        self.median_window = max(1, int(median_window))
        self.out_x = float(initial_x)
        self.out_y = float(initial_y)
        self.history = deque(maxlen=self.median_window)

    def reset(self, x: Optional[float] = None, y: Optional[float] = None):
        if x is not None:
            self.out_x = float(x)
        if y is not None:
            self.out_y = float(y)
        self.history.clear()

    def update(self, x: float, y: float, freeze_motion: bool = False):
        if freeze_motion:
            return clamp_xy(self.out_x, self.out_y, self.screen_w, self.screen_h)

        self.history.append((float(x), float(y)))

        # Median of recent frames to reject outlier spikes
        history = np.asarray(self.history, dtype=np.float32)
        target_x, target_y = np.median(history, axis=0)

        dx = float(target_x - self.out_x)
        dy = float(target_y - self.out_y)
        delta = float(np.hypot(dx, dy))

        # Deadzone: hold position if movement is tiny (stationary jitter)
        if delta <= self.deadzone_px:
            return clamp_xy(self.out_x, self.out_y, self.screen_w, self.screen_h)

        # Direct jump — no EMA, no lag
        self.out_x = float(target_x)
        self.out_y = float(target_y)
        return clamp_xy(self.out_x, self.out_y, self.screen_w, self.screen_h)


class MouseController:
    def __init__(self):
        self.user32 = ctypes.windll.user32
        try:
            self.user32.ClipCursor(None)
        except Exception:
            pass
        self._left_down = False
        self._right_down = False

    def move_to(self, x: int, y: int):
        self.user32.SetCursorPos(int(x), int(y))

    def left_click(self):
        self.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        self.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def right_click(self):
        self.user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        self.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

    def double_click(self, interval: float = 0.06):
        self.left_click()
        time.sleep(float(interval))
        self.left_click()

    def left_down(self):
        if not self._left_down:
            self.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            self._left_down = True

    def left_up(self):
        if self._left_down:
            self.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            self._left_down = False

    def right_down(self):
        if not self._right_down:
            self.user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
            self._right_down = True

    def right_up(self):
        if self._right_down:
            self.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
            self._right_down = False

    def scroll(self, amount: int):
        """Positive = scroll up, negative = scroll down."""
        self.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(amount), 0)

    def release_all(self):
        self.left_up()
        self.right_up()


class WinkClickDetector:
    """
    Detects winks and eye-hold gestures.
    - Left wink (quick close/open of left eye only) -> left click
    - Right wink (quick close/open of right eye only) -> right click
    - Double wink (left then right wink within window) -> double click
    - Left eye held closed + head moving -> drag mode
    """

    def __init__(
        self,
        close_threshold: float,
        open_threshold: float,
        min_wink_duration: float,
        max_wink_duration: float,
        double_wink_window: float,
        click_cooldown: float,
        drag_start_delay: float,
    ):
        self.close_threshold = float(close_threshold)
        self.open_threshold = float(open_threshold)
        self.min_wink_duration = float(min_wink_duration)
        self.max_wink_duration = float(max_wink_duration)
        self.double_wink_window = float(double_wink_window)
        self.click_cooldown = float(click_cooldown)
        self.drag_start_delay = float(drag_start_delay)

        # State tracking
        self.left_state = "open"
        self.right_state = "open"
        self.left_closed_at = 0.0
        self.right_closed_at = 0.0
        self.left_opened_at = 0.0
        self.right_opened_at = 0.0

        self.last_left_click_at = 0.0
        self.last_right_click_at = 0.0
        self.pending_double_side = None  # 'left' or 'right' if we saw one wink and are waiting for the other
        self.pending_double_at = 0.0

        self.drag_active = False
        self.drag_started_at = 0.0
        self.last_action = "idle"
        self.last_action_at = 0.0

    def reset(self):
        self.left_state = "open"
        self.right_state = "open"
        self.left_closed_at = 0.0
        self.right_closed_at = 0.0
        self.left_opened_at = 0.0
        self.right_opened_at = 0.0
        self.pending_double_side = None
        self.pending_double_at = 0.0
        self.drag_active = False
        self.drag_started_at = 0.0
        self.last_action = "idle"
        self.last_action_at = 0.0

    def update(self, left_blink: float, right_blink: float, now: float):
        action = None
        self.last_action = "idle"

        # Determine eye states
        left_closed = left_blink < self.close_threshold
        right_closed = right_blink < self.close_threshold
        left_open = left_blink > self.open_threshold
        right_open = right_blink > self.open_threshold

        # Track left eye transitions
        if self.left_state == "open" and left_closed:
            self.left_state = "closed"
            self.left_closed_at = now
        elif self.left_state == "closed" and left_open:
            self.left_state = "open"
            self.left_opened_at = now
            duration = self.left_opened_at - self.left_closed_at
            if self.min_wink_duration <= duration <= self.max_wink_duration:
                # Left wink completed
                if self.pending_double_side == "right" and (now - self.pending_double_at) <= self.double_wink_window:
                    action = "double"
                    self.pending_double_side = None
                elif now - self.last_left_click_at >= self.click_cooldown:
                    action = "left"
                    self.last_left_click_at = now
                    self.pending_double_side = "left"
                    self.pending_double_at = now
                else:
                    self.pending_double_side = None
            else:
                self.pending_double_side = None

        # Track right eye transitions
        if self.right_state == "open" and right_closed:
            self.right_state = "closed"
            self.right_closed_at = now
        elif self.right_state == "closed" and right_open:
            self.right_state = "open"
            self.right_opened_at = now
            duration = self.right_opened_at - self.right_closed_at
            if self.min_wink_duration <= duration <= self.max_wink_duration:
                # Right wink completed
                if self.pending_double_side == "left" and (now - self.pending_double_at) <= self.double_wink_window:
                    action = "double"
                    self.pending_double_side = None
                elif now - self.last_right_click_at >= self.click_cooldown:
                    action = "right"
                    self.last_right_click_at = now
                    self.pending_double_side = "right"
                    self.pending_double_at = now
                else:
                    self.pending_double_side = None
            else:
                self.pending_double_side = None

        # Expire pending double wink
        if self.pending_double_side is not None and (now - self.pending_double_at) > self.double_wink_window:
            self.pending_double_side = None

        # Drag detection: left eye held closed
        if left_closed and not right_closed:
            if not self.drag_active:
                if now - self.left_closed_at >= self.drag_start_delay:
                    self.drag_active = True
                    self.drag_started_at = now
                    self.last_action = "drag_start"
            else:
                self.last_action = "dragging"
        else:
            if self.drag_active:
                self.drag_active = False
                self.last_action = "drag_end"

        if action is not None:
            self.last_action = action
            self.last_action_at = now

        return action, self.drag_active


class BlinkGuard:
    def __init__(self, freeze_threshold: float, recovery_time: float):
        self.freeze_threshold = float(freeze_threshold)
        self.recovery_time = float(recovery_time)
        self.currently_closed = False
        self.last_blink_ended_at = 0.0

    def reset(self):
        self.currently_closed = False
        self.last_blink_ended_at = 0.0

    def update(self, left_blink: float, right_blink: float, now: float):
        mean_blink = (float(left_blink) + float(right_blink)) / 2.0
        both_closed = mean_blink < self.freeze_threshold
        if both_closed and not self.currently_closed:
            self.currently_closed = True
        elif not both_closed and self.currently_closed:
            self.currently_closed = False
            self.last_blink_ended_at = now

    def should_freeze_motion(self, left_blink: float, right_blink: float, now: float):
        mean_blink = (float(left_blink) + float(right_blink)) / 2.0
        partially_closed = mean_blink < self.freeze_threshold
        in_recovery = self.last_blink_ended_at > 0.0 and now - self.last_blink_ended_at <= self.recovery_time
        return self.currently_closed or partially_closed or in_recovery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=Path("calibration_model.pkl"))
    parser.add_argument("--deadzone-px", type=float, default=8.0, help="ignore head movements smaller than this (pixels)")
    parser.add_argument("--median-window", type=int, default=3, help="frames for outlier rejection")
    parser.add_argument("--range", type=float, default=0.8, help="cursor range multiplier (bigger = more screen coverage per head movement)")
    parser.add_argument("--head-assist", type=float, default=0.0, help="0..1 blend of eye data into prediction; 0 = pure head/nose tracking")
    parser.add_argument(
        "--control-mouse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="move the real Windows mouse cursor with head (default: on; use --no-control-mouse for preview only)",
    )
    parser.add_argument("--show-overlay", action="store_true", help="show the fullscreen gaze overlay instead of staying on your current screen")
    parser.add_argument("--show-webcam-preview", action="store_true", help="show the webcam preview while controlling the real desktop cursor")
    parser.add_argument("--cursor-deadzone-px", type=float, default=6.0, help="skip SetCursorPos when the cursor only moved this many pixels")
    parser.add_argument("--blink-freeze-threshold", type=float, default=0.195, help="freeze cursor when a blink compresses the eye ratio below this threshold")
    parser.add_argument("--blink-recovery", type=float, default=0.22, help="seconds to hold cursor steady after a blink ends")
    # Wink settings
    parser.add_argument("--wink-close-threshold", type=float, default=0.27, help="eye ratio below this counts as closed for winks")
    parser.add_argument("--wink-open-threshold", type=float, default=0.35, help="eye ratio above this counts as open")
    parser.add_argument("--min-wink-duration", type=float, default=0.02, help="minimum seconds an eye must stay closed to count as a wink")
    parser.add_argument("--max-wink-duration", type=float, default=0.80, help="maximum seconds an eye can stay closed to count as a wink")
    parser.add_argument("--double-wink-window", type=float, default=0.50, help="max seconds between two winks to register as double click")
    parser.add_argument("--click-cooldown", type=float, default=0.15, help="minimum seconds between clicks")
    parser.add_argument("--drag-start-delay", type=float, default=0.25, help="seconds left eye must be held closed before drag starts")
    parser.add_argument("--scroll-threshold-px", type=float, default=8.0, help="vertical head movement while left-eye-closed to trigger scroll instead of drag")
    parser.add_argument("--scroll-speed", type=float, default=40.0, help="scroll wheel units per frame when scrolling")
    args = parser.parse_args()

    enable_dpi_awareness()

    if args.median_window < 1:
        raise ValueError("--median-window must be at least 1.")

    if not args.model.exists():
        raise FileNotFoundError(f"Calibration model not found: {args.model}. Run calibrate.py first.")

    with args.model.open("rb") as f:
        payload = pickle.load(f)

    W = payload["W"]
    feature_dim = int(payload["feature_dim"])
    W_eye = payload.get("W_eye")
    W_head = payload.get("W_head")
    corr_full = payload.get("corr_full", {})
    corr_eye = payload.get("corr_eye", {})
    corr_head = payload.get("corr_head", {})
    eye_feature_dim = int(payload.get("eye_feature_dim", -1))
    head_feature_dim = int(payload.get("head_feature_dim", -1))
    feature_mode = str(payload.get("feature_mode", "linear"))
    default_head_assist = float(payload.get("head_assist_default", 0.0))
    head_assist = float(args.head_assist)
    head_assist = float(np.clip(head_assist, 0.0, 1.0))
    use_hybrid = W_eye is not None and W_head is not None
    model_screen_w, model_screen_h = payload["screen_size"]

    current_w, current_h = get_screen_size()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    face_mesh = build_face_mesh()
    mouse = MouseController() if args.control_mouse else None
    blink_guard = BlinkGuard(args.blink_freeze_threshold, args.blink_recovery)
    wink_detector = WinkClickDetector(
        close_threshold=args.wink_close_threshold,
        open_threshold=args.wink_open_threshold,
        min_wink_duration=args.min_wink_duration,
        max_wink_duration=args.max_wink_duration,
        double_wink_window=args.double_wink_window,
        click_cooldown=args.click_cooldown,
        drag_start_delay=args.drag_start_delay,
    )

    show_overlay = bool(args.show_overlay)
    preview_visible = (not args.control_mouse) or bool(args.show_webcam_preview)
    if preview_visible:
        cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Webcam", 420, 260)
        cv2.moveWindow("Webcam", 20, 20)
    if show_overlay:
        cv2.namedWindow("Gaze Dot", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Gaze Dot", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    smooth_x = current_w // 2
    smooth_y = current_h // 2
    head_filter = DirectFilter(
        current_w,
        current_h,
        deadzone_px=args.deadzone_px,
        median_window=args.median_window,
        initial_x=smooth_x,
        initial_y=smooth_y,
    )
    last_cursor_x = smooth_x
    last_cursor_y = smooth_y
    last_drag_y = smooth_y

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        feat = extract_features(frame, face_mesh)
        preview = frame.copy()
        canvas = np.full((current_h, current_w, 3), 255, dtype=np.uint8) if show_overlay else None
        click_action = None
        tracking_ready = False
        drag_active = False
        is_scrolling = False

        if feat is None:
            if args.control_mouse:
                blink_guard.reset()
                wink_detector.reset()
                if mouse is not None:
                    mouse.release_all()
            head_filter.reset(smooth_x, smooth_y)
            preview = draw_status(preview, "Face not detected", ok=False)
            if show_overlay:
                cv2.putText(canvas, "Face not detected", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        elif feat.vector.shape[0] != feature_dim:
            if args.control_mouse:
                blink_guard.reset()
                wink_detector.reset()
                if mouse is not None:
                    mouse.release_all()
            head_filter.reset(smooth_x, smooth_y)
            preview = draw_status(preview, "Feature mismatch: recalibrate", ok=False)
            if show_overlay:
                cv2.putText(canvas, "Feature mismatch: recalibrate", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            now_ts = time.time()
            freeze_motion = False
            if args.control_mouse:
                blink_guard.update(feat.left_blink, feat.right_blink, now_ts)
                freeze_motion = blink_guard.should_freeze_motion(feat.left_blink, feat.right_blink, now_ts)
                click_action, drag_active = wink_detector.update(feat.left_blink, feat.right_blink, now_ts)

                if mouse is not None:
                    if click_action == "left":
                        mouse.left_click()
                    elif click_action == "right":
                        mouse.right_click()
                    elif click_action == "double":
                        mouse.double_click()

            # Head-dominant prediction
            if use_hybrid:
                _, eye_vec, head_vec = split_feature_vector(feat.vector)
                if eye_vec.shape[0] != eye_feature_dim or head_vec.shape[0] != head_feature_dim:
                    if args.control_mouse:
                        blink_guard.reset()
                        wink_detector.reset()
                        if mouse is not None:
                            mouse.release_all()
                    head_filter.reset(smooth_x, smooth_y)
                    preview = draw_status(preview, "Hybrid feature mismatch: recalibrate", ok=False)
                    if show_overlay:
                        cv2.putText(canvas, "Hybrid feature mismatch: recalibrate", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    else:
                        draw_gaze_minimap(preview, smooth_x, smooth_y, current_w, current_h, active=False, mouse_mode=args.control_mouse)
                    if preview_visible:
                        cv2.imshow("Webcam", preview)
                    if show_overlay:
                        cv2.imshow("Gaze Dot", canvas)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break
                    continue
                px_eye, py_eye = predict_xy(eye_vec, W_eye, feature_mode=feature_mode)
                px_head, py_head = predict_xy(head_vec, W_head, feature_mode=feature_mode)
                px_eye, py_eye = apply_model_correction(px_eye, py_eye, corr_eye)
                px_head, py_head = apply_model_correction(px_head, py_head, corr_head)
                # Make head dominant: flip default assist so head drives cursor
                px = (1.0 - head_assist) * px_head + head_assist * px_eye
                py = (1.0 - head_assist) * py_head + head_assist * py_eye
            else:
                px, py = predict_xy(feat.vector, W, feature_mode=feature_mode)
                px, py = apply_model_correction(px, py, corr_full)

            # Scale prediction to current screen + range multiplier
            px *= current_w / max(float(model_screen_w), 1.0)
            py *= current_h / max(float(model_screen_h), 1.0)
            cx, cy = current_w / 2.0, current_h / 2.0
            px = (px - cx) * float(args.range) + cx
            py = (py - cy) * float(args.range) + cy

            px, py = clamp_xy(px, py, current_w, current_h)
            smooth_x, smooth_y = head_filter.update(px, py, freeze_motion=freeze_motion)
            tracking_ready = True

            # Drag / Scroll logic when left eye is held closed
            if drag_active and mouse is not None:
                dy = smooth_y - last_drag_y
                if abs(dy) > args.scroll_threshold_px:
                    is_scrolling = True
                    scroll_amount = -int(np.sign(dy) * args.scroll_speed)
                    mouse.scroll(scroll_amount)
                    mouse.left_up()  # release drag while scrolling
                else:
                    is_scrolling = False
                    mouse.left_down()
            elif not drag_active and mouse is not None:
                mouse.left_up()

            last_drag_y = smooth_y

            mode_text = f"head:{1.0 - head_assist:.2f}" if use_hybrid else "full"
            preview = draw_status(preview, f"Head: ({smooth_x}, {smooth_y}) {mode_text}")

            # Overlays
            cv2.putText(
                preview,
                f"blink L:{feat.left_blink:.2f} R:{feat.right_blink:.2f} mouth:{feat.mouth_open:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (212, 212, 216),
                1,
                cv2.LINE_AA,
            )
            if freeze_motion:
                cv2.putText(preview, "Blink hold", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (246, 130, 59), 2)

            action_text = {
                "left": "L-CLICK",
                "right": "R-CLICK",
                "double": "DOUBLE-CLICK",
                "drag_start": "DRAG START",
                "dragging": "DRAGGING",
                "drag_end": "DRAG END",
            }.get(wink_detector.last_action, "")
            if action_text:
                color = (92, 206, 132) if "CLICK" in action_text else (59, 130, 246)
                cv2.putText(preview, action_text, (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if drag_active:
                if is_scrolling:
                    cv2.putText(preview, "SCROLLING", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 171, 72), 2)
                else:
                    cv2.putText(preview, "DRAG ACTIVE", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 38, 38), 2)

            if args.control_mouse and mouse is not None:
                cursor_delta = float(np.hypot(smooth_x - last_cursor_x, smooth_y - last_cursor_y))
                if not freeze_motion and cursor_delta >= args.cursor_deadzone_px:
                    mouse.move_to(smooth_x, smooth_y)
                    last_cursor_x = smooth_x
                    last_cursor_y = smooth_y

            if show_overlay:
                cv2.circle(canvas, (smooth_x, smooth_y), 16, (59, 130, 246), -1)
                cv2.circle(canvas, (smooth_x, smooth_y), 30, (246, 130, 59), 2)

        if show_overlay:
            overlay_text = "NeuralSight Active" if args.control_mouse else "Preview Mode"
            cv2.putText(canvas, overlay_text, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (27, 24, 24), 2)
            cv2.putText(canvas, "Press ESC to quit", (40, current_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (27, 24, 24), 2)
        elif preview_visible:
            draw_gaze_minimap(preview, smooth_x, smooth_y, current_w, current_h, active=tracking_ready, mouse_mode=args.control_mouse)

        if preview_visible:
            cv2.imshow("Webcam", preview)
        if show_overlay:
            cv2.imshow("Gaze Dot", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    if mouse is not None:
        mouse.release_all()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
