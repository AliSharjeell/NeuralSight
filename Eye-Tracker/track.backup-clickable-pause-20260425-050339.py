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
CLICKABLE_CONTROL_TYPES = {
    "ButtonControl",
    "CheckBoxControl",
    "ComboBoxControl",
    "EditControl",
    "HyperlinkControl",
    "MenuItemControl",
    "RadioButtonControl",
    "TabItemControl",
}
DENSE_CLICKABLE_CONTROL_TYPES = {
    "DataItemControl",
    "ListItemControl",
    "TreeItemControl",
}


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

    cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (18, 22, 34), -1)
    cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (92, 106, 140), 2)
    cv2.rectangle(frame, (map_x1, map_y1), (map_x2, map_y2), (240, 244, 250), -1)
    cv2.rectangle(frame, (map_x1, map_y1), (map_x2, map_y2), (120, 132, 160), 1)

    cv2.putText(frame, "Desktop Tracking", (panel_x1 + 12, panel_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (236, 240, 246), 1, cv2.LINE_AA)
    mode_text = "Mouse ON" if mouse_mode else "Preview"
    cv2.putText(frame, mode_text, (panel_x2 - 96, panel_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 192, 214), 1, cv2.LINE_AA)

    center_x = map_x1 + map_w // 2
    center_y = map_y1 + map_h // 2
    cv2.line(frame, (center_x, map_y1 + 6), (center_x, map_y2 - 6), (216, 222, 232), 1)
    cv2.line(frame, (map_x1 + 6, center_y), (map_x2 - 6, center_y), (216, 222, 232), 1)

    if mouse_mode:
        status_text = "Controlling real cursor" if active else "Waiting for face"
        cv2.putText(frame, status_text, (panel_x1 + 12, panel_y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 234, 241), 1, cv2.LINE_AA)
        cx = map_x1 + map_w // 2
        cy = map_y1 + map_h // 2
        cv2.arrowedLine(frame, (cx - 26, cy + 18), (cx + 18, cy - 16), (40, 58, 88), 3, cv2.LINE_AA, tipLength=0.28)
        cv2.circle(frame, (cx + 18, cy - 16), 5, (0, 120, 255), -1)
    else:
        dot_x = map_x1 + int(round(np.clip(gaze_x, 0, screen_w - 1) * (map_w - 1) / max(screen_w - 1, 1)))
        dot_y = map_y1 + int(round(np.clip(gaze_y, 0, screen_h - 1) * (map_h - 1) / max(screen_h - 1, 1)))
        dot_fill = (0, 0, 255) if active else (150, 156, 168)
        dot_ring = (0, 120, 255) if active else (112, 118, 130)
        cv2.circle(frame, (dot_x, dot_y), 7, dot_fill, -1)
        cv2.circle(frame, (dot_x, dot_y), 13, dot_ring, 2)
        status_text = "Preview dot active" if active else "Waiting for face"
        cv2.putText(frame, status_text, (panel_x1 + 12, panel_y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 234, 241), 1, cv2.LINE_AA)


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


class AdaptiveGazeFilter:
    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        smooth: float,
        median_window: int,
        steady_radius_px: float,
        response_px: float,
        sensitivity: float,
        initial_x: float,
        initial_y: float,
    ):
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)
        self.center_x = float(screen_w) / 2.0
        self.center_y = float(screen_h) / 2.0
        self.smooth = float(np.clip(smooth, 0.01, 1.0))
        self.median_window = max(1, int(median_window))
        self.steady_radius_px = max(0.0, float(steady_radius_px))
        self.response_px = max(1.0, float(response_px))
        self.sensitivity = float(np.clip(sensitivity, 0.6, 1.2))
        self.min_alpha = max(0.015, self.smooth * 0.22)
        self.hold_radius_px = max(2.0, self.steady_radius_px * 0.45)
        self.filtered_x = float(initial_x)
        self.filtered_y = float(initial_y)
        self.history = deque(maxlen=self.median_window)

    def reset(self, x: Optional[float] = None, y: Optional[float] = None):
        if x is not None:
            self.filtered_x = float(x)
        if y is not None:
            self.filtered_y = float(y)
        self.history.clear()

    def _apply_sensitivity(self, x: float, y: float):
        x = (float(x) - self.center_x) * self.sensitivity + self.center_x
        y = (float(y) - self.center_y) * self.sensitivity + self.center_y
        return x, y

    def update(self, x: float, y: float, freeze_motion: bool = False):
        if freeze_motion:
            return clamp_xy(self.filtered_x, self.filtered_y, self.screen_w, self.screen_h)

        x, y = self._apply_sensitivity(x, y)
        self.history.append((x, y))

        history = np.asarray(self.history, dtype=np.float32)
        target_x, target_y = np.median(history, axis=0)

        dx = float(target_x - self.filtered_x)
        dy = float(target_y - self.filtered_y)
        delta = float(np.hypot(dx, dy))

        if delta <= self.hold_radius_px:
            alpha = 0.0
        elif delta <= self.steady_radius_px:
            progress = (delta - self.hold_radius_px) / max(self.steady_radius_px - self.hold_radius_px, 1.0)
            alpha = self.min_alpha * float(np.clip(progress, 0.0, 1.0))
        else:
            move_ratio = min(1.0, (delta - self.steady_radius_px) / self.response_px)
            alpha = self.min_alpha + (self.smooth - self.min_alpha) * move_ratio

        self.filtered_x += alpha * dx
        self.filtered_y += alpha * dy
        return clamp_xy(self.filtered_x, self.filtered_y, self.screen_w, self.screen_h)


class MouseController:
    def __init__(self):
        self.user32 = ctypes.windll.user32
        try:
            self.user32.ClipCursor(None)
        except Exception:
            pass

    def move_to(self, x: int, y: int):
        self.user32.SetCursorPos(int(x), int(y))

    def left_click(self):
        self.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        self.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def double_click(self, interval: float = 0.06):
        self.left_click()
        time.sleep(float(interval))
        self.left_click()


class ClickableSnapper:
    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        snap_radius_px: float,
        hold_time: float,
        dwell_time: float,
        dwell_radius_px: float,
        unlock_distance_px: float,
        include_dense_items: bool,
        scan_step_px: int = 24,
    ):
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)
        self.snap_radius_px = float(snap_radius_px)
        self.hold_time = float(hold_time)
        self.dwell_time = float(dwell_time)
        self.dwell_radius_px = float(dwell_radius_px)
        self.unlock_distance_px = float(unlock_distance_px)
        self.include_dense_items = bool(include_dense_items)
        self.scan_step_px = max(12, int(scan_step_px))
        self.max_area = float(screen_w * screen_h) * 0.18
        self.max_width = float(screen_w) * 0.9
        self.max_height = float(screen_h) * 0.45
        self.max_dense_area = float(screen_w * screen_h) * 0.035
        self.max_dense_width = float(screen_w) * 0.55
        self.max_dense_height = float(screen_h) * 0.18
        self.locked_target = None
        self.locked_until = 0.0
        self.locked_label = ""
        self.pending_anchor = None
        self.pending_started_at = 0.0
        self.enabled = False
        self.auto = None
        try:
            import uiautomation as auto

            self.auto = auto
            self.enabled = True
        except Exception:
            self.auto = None

    def reset(self):
        self.locked_target = None
        self.locked_until = 0.0
        self.locked_label = ""
        self.pending_anchor = None
        self.pending_started_at = 0.0

    def _control_key(self, control, rect):
        return (
            str(getattr(control, "ControlTypeName", "")),
            str(getattr(control, "Name", "")),
            int(rect.left),
            int(rect.top),
            int(rect.right),
            int(rect.bottom),
        )

    def _is_clickable(self, control, rect):
        if rect is None or rect.isempty():
            return False
        control_type = str(getattr(control, "ControlTypeName", ""))
        width = float(rect.width())
        height = float(rect.height())
        if width < 8 or height < 8:
            return False
        if not bool(getattr(control, "IsEnabled", True)):
            return False

        allowed = set(CLICKABLE_CONTROL_TYPES)
        if self.include_dense_items:
            allowed.update(DENSE_CLICKABLE_CONTROL_TYPES)
        if control_type not in allowed:
            return False

        if control_type in DENSE_CLICKABLE_CONTROL_TYPES:
            if width > self.max_dense_width or height > self.max_dense_height:
                return False
            if width * height > self.max_dense_area:
                return False
        else:
            if width > self.max_width or height > self.max_height:
                return False
            if width * height > self.max_area:
                return False

        return True

    def _candidate_from_control(self, control, raw_x: int, raw_y: int):
        current = control
        for _ in range(5):
            if current is None:
                break
            rect = getattr(current, "BoundingRectangle", None)
            if self._is_clickable(current, rect):
                cx = int(rect.xcenter())
                cy = int(rect.ycenter())
                dist = float(np.hypot(cx - raw_x, cy - raw_y))
                if dist <= self.snap_radius_px:
                    return {
                        "x": cx,
                        "y": cy,
                        "dist": dist,
                        "key": self._control_key(current, rect),
                        "label": f"{current.ControlTypeName}:{current.Name}" if current.Name else current.ControlTypeName,
                    }
            try:
                current = current.GetParentControl()
            except Exception:
                break
        return None

    def _scan_points(self, raw_x: int, raw_y: int):
        step = self.scan_step_px
        offsets = [
            (0, 0),
            (-step, 0),
            (step, 0),
            (0, -step),
            (0, step),
            (-step, -step),
            (step, -step),
            (-step, step),
            (step, step),
            (-2 * step, 0),
            (2 * step, 0),
            (0, -2 * step),
            (0, 2 * step),
        ]
        for dx, dy in offsets:
            yield (
                int(np.clip(raw_x + dx, 0, self.screen_w - 1)),
                int(np.clip(raw_y + dy, 0, self.screen_h - 1)),
            )

    def _find_candidate(self, raw_x: int, raw_y: int):
        if not self.enabled or self.auto is None:
            return None

        best = None
        seen = set()
        for px, py in self._scan_points(raw_x, raw_y):
            try:
                control = self.auto.ControlFromPoint(px, py)
            except Exception:
                continue
            candidate = self._candidate_from_control(control, raw_x, raw_y)
            if not candidate:
                continue
            if candidate["key"] in seen:
                continue
            seen.add(candidate["key"])
            if best is None or candidate["dist"] < best["dist"]:
                best = candidate
        return best

    def apply(self, raw_x: int, raw_y: int, request_snap: bool, now: float):
        if not self.enabled:
            return raw_x, raw_y, False, "uia-off"

        if self.locked_target is not None:
            lock_dist = float(np.hypot(raw_x - self.locked_target[0], raw_y - self.locked_target[1]))
            if lock_dist > self.unlock_distance_px:
                self.reset()

        if request_snap:
            if self.pending_anchor is None:
                self.pending_anchor = (raw_x, raw_y)
                self.pending_started_at = now
            else:
                anchor_dist = float(np.hypot(raw_x - self.pending_anchor[0], raw_y - self.pending_anchor[1]))
                if anchor_dist > self.dwell_radius_px:
                    self.pending_anchor = (raw_x, raw_y)
                    self.pending_started_at = now

            if self.locked_target is None and now - self.pending_started_at >= self.dwell_time:
                candidate = self._find_candidate(raw_x, raw_y)
                if candidate is not None:
                    self.locked_target = (candidate["x"], candidate["y"])
                    self.locked_until = now + self.hold_time
                    self.locked_label = candidate["label"]
                    self.pending_anchor = None
                    self.pending_started_at = 0.0
        else:
            self.pending_anchor = None
            self.pending_started_at = 0.0

        if self.locked_target is not None and now <= self.locked_until:
            return self.locked_target[0], self.locked_target[1], True, self.locked_label

        self.reset()
        return raw_x, raw_y, False, ""


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


class MouthClickDetector:
    def __init__(
        self,
        open_threshold: float,
        release_threshold: float,
        single_open_time: float,
        double_open_time: float,
        click_cooldown: float,
    ):
        self.open_threshold = float(open_threshold)
        self.release_threshold = float(release_threshold)
        self.single_open_time = float(single_open_time)
        self.double_open_time = float(double_open_time)
        self.click_cooldown = float(click_cooldown)
        self.currently_open = False
        self.open_started_at = 0.0
        self.action_fired_this_open = False
        self.last_action_at = 0.0
        self.peak_open = 0.0

    def reset(self):
        self.currently_open = False
        self.open_started_at = 0.0
        self.action_fired_this_open = False
        self.peak_open = 0.0

    def update(self, mouth_open: float, now: float):
        mouth_open = float(mouth_open)
        action = None
        state = "closed"

        if not self.currently_open:
            if mouth_open >= self.open_threshold:
                self.currently_open = True
                self.open_started_at = now
                self.action_fired_this_open = False
                self.peak_open = mouth_open
                state = "opening"
        else:
            open_duration = now - self.open_started_at
            cooldown_ready = now - self.last_action_at >= self.click_cooldown
            self.peak_open = max(self.peak_open, mouth_open)
            release_trigger = max(
                self.release_threshold,
                min(self.open_threshold - 0.01, self.peak_open * 0.62),
            )

            if (
                not self.action_fired_this_open
                and open_duration >= self.double_open_time
                and cooldown_ready
            ):
                self.action_fired_this_open = True
                self.last_action_at = now
                action = "double"
                state = "double"
            elif mouth_open <= release_trigger:
                if (
                    not self.action_fired_this_open
                    and open_duration >= self.single_open_time
                    and cooldown_ready
                ):
                    self.last_action_at = now
                    action = "single"
                    state = "single"
                self.currently_open = False
                self.open_started_at = 0.0
                self.action_fired_this_open = False
                self.peak_open = 0.0
                if action is None:
                    state = "closed"
            else:
                if self.action_fired_this_open:
                    state = "open"
                elif open_duration >= self.single_open_time:
                    state = "armed"
                else:
                    state = "opening"

        return action, state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=Path("calibration_model.pkl"))
    parser.add_argument("--smooth", type=float, default=0.10, help="0..1, higher is snappier")
    parser.add_argument("--sensitivity", type=float, default=1.0, help="final cursor range multiplier; keep near 1.0 so all screen edges remain reachable")
    parser.add_argument("--stability-window", type=int, default=13, help="number of recent predictions used for median stabilization")
    parser.add_argument("--steady-radius-px", type=float, default=36.0, help="keep the gaze point steadier while predictions stay within this radius")
    parser.add_argument("--response-px", type=float, default=220.0, help="distance over which the filter ramps from steady to responsive")
    parser.add_argument("--head-assist", type=float, default=None, help="0..1 blend weight for head/face assist in hybrid mode")
    parser.add_argument("--x-gain", type=float, default=1.65, help="horizontal range multiplier for reaching left and right screen edges")
    parser.add_argument("--y-gain", type=float, default=1.55, help="vertical range multiplier for reaching top and bottom screen edges")
    parser.add_argument(
        "--control-mouse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="move the real Windows mouse cursor with gaze (default: on; use --no-control-mouse for preview only)",
    )
    parser.add_argument(
        "--mouth-open-click",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="trigger mouth-based clicking (defaults to on when real cursor control is on)",
    )
    parser.add_argument("--show-overlay", action="store_true", help="show the fullscreen gaze overlay instead of staying on your current screen")
    parser.add_argument("--show-webcam-preview", action="store_true", help="show the webcam preview while controlling the real desktop cursor")
    parser.add_argument("--cursor-deadzone-px", type=float, default=18.0, help="ignore tiny cursor moves under this pixel distance")
    parser.add_argument("--edge-zone-ratio", type=float, default=0.30, help="outer screen band ratio where edge assist starts pushing outward")
    parser.add_argument("--edge-boost-px", type=float, default=430.0, help="maximum outward push near the screen edges")
    parser.add_argument("--snap-clickables", action="store_true", default=True, help="snap to nearby clickable desktop controls while a mouth click is arming")
    parser.add_argument("--snap-radius-px", type=float, default=150.0, help="max distance for snapping to a nearby clickable control")
    parser.add_argument("--snap-hold", type=float, default=0.95, help="seconds to hold onto a snapped control after acquisition")
    parser.add_argument("--snap-dwell", type=float, default=0.05, help="seconds gaze must stay steady before snapping to a control")
    parser.add_argument("--snap-dwell-radius-px", type=float, default=48.0, help="allowed movement during snap dwell acquisition")
    parser.add_argument("--snap-unlock-distance-px", type=float, default=230.0, help="release a snapped target early if gaze moves this far away")
    parser.add_argument("--snap-browse-items", action="store_true", default=True, help="also allow list/tree/data items as snap targets in dense apps like File Explorer")
    parser.add_argument("--blink-freeze-threshold", type=float, default=0.195, help="freeze cursor when a blink compresses the eye ratio below this threshold")
    parser.add_argument("--blink-recovery", type=float, default=0.22, help="seconds to hold cursor steady after a blink ends")
    parser.add_argument("--mouth-open-threshold", type=float, default=0.14, help="mouth-open ratio required to arm a click")
    parser.add_argument("--mouth-release-threshold", type=float, default=0.08, help="mouth-open ratio required before another mouth click can arm")
    parser.add_argument("--mouth-single-open", type=float, default=0.08, help="mouth-open duration that becomes a single click when you close your mouth")
    parser.add_argument("--mouth-double-open", type=float, default=0.38, help="mouth-open duration that becomes a double click while still open")
    parser.add_argument("--click-cooldown", type=float, default=0.45, help="minimum seconds between mouth-triggered clicks")
    args = parser.parse_args()

    enable_dpi_awareness()

    if args.mouth_open_click is None:
        args.mouth_open_click = bool(args.control_mouse)

    if args.mouth_release_threshold >= args.mouth_open_threshold:
        raise ValueError("--mouth-release-threshold must be lower than --mouth-open-threshold.")
    if args.mouth_double_open <= args.mouth_single_open:
        raise ValueError("--mouth-double-open must be greater than --mouth-single-open.")
    if not (0.0 < args.edge_zone_ratio < 0.5):
        raise ValueError("--edge-zone-ratio must be between 0 and 0.5.")
    if args.stability_window < 1:
        raise ValueError("--stability-window must be at least 1.")
    if args.steady_radius_px < 0.0:
        raise ValueError("--steady-radius-px must be non-negative.")
    if args.response_px <= 0.0:
        raise ValueError("--response-px must be greater than 0.")
    if args.x_gain <= 0.0:
        raise ValueError("--x-gain must be greater than 0.")

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
    head_assist = default_head_assist if args.head_assist is None else float(args.head_assist)
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
    mouse = MouseController() if args.control_mouse or args.mouth_open_click else None
    blink_guard = BlinkGuard(args.blink_freeze_threshold, args.blink_recovery)
    snapper = ClickableSnapper(
        current_w,
        current_h,
        snap_radius_px=args.snap_radius_px,
        hold_time=args.snap_hold,
        dwell_time=args.snap_dwell,
        dwell_radius_px=args.snap_dwell_radius_px,
        unlock_distance_px=args.snap_unlock_distance_px,
        include_dense_items=args.snap_browse_items,
    )
    mouth_detector = MouthClickDetector(
        open_threshold=args.mouth_open_threshold,
        release_threshold=args.mouth_release_threshold,
        single_open_time=args.mouth_single_open,
        double_open_time=args.mouth_double_open,
        click_cooldown=args.click_cooldown,
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
    gaze_filter = AdaptiveGazeFilter(
        current_w,
        current_h,
        smooth=args.smooth,
        median_window=args.stability_window,
        steady_radius_px=args.steady_radius_px,
        response_px=args.response_px,
        sensitivity=args.sensitivity,
        initial_x=smooth_x,
        initial_y=smooth_y,
    )
    last_cursor_x = smooth_x
    last_cursor_y = smooth_y

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        feat = extract_features(frame, face_mesh)
        preview = frame.copy()
        canvas = np.full((current_h, current_w, 3), 255, dtype=np.uint8) if show_overlay else None
        click_action = None
        mouth_state = "idle"
        snapped_to_control = False
        snap_label = ""
        tracking_ready = False

        if feat is None:
            if args.control_mouse:
                blink_guard.reset()
            if args.snap_clickables:
                snapper.reset()
            if args.mouth_open_click:
                mouth_detector.reset()
            gaze_filter.reset(smooth_x, smooth_y)
            preview = draw_status(preview, "Face not detected", ok=False)
            if show_overlay:
                cv2.putText(canvas, "Face not detected", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        elif feat.vector.shape[0] != feature_dim:
            if args.control_mouse:
                blink_guard.reset()
            if args.snap_clickables:
                snapper.reset()
            if args.mouth_open_click:
                mouth_detector.reset()
            gaze_filter.reset(smooth_x, smooth_y)
            preview = draw_status(preview, "Feature mismatch: recalibrate", ok=False)
            if show_overlay:
                cv2.putText(canvas, "Feature mismatch: recalibrate", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            now_ts = time.time()
            freeze_motion = False
            if args.control_mouse:
                blink_guard.update(feat.left_blink, feat.right_blink, now_ts)
                freeze_motion = blink_guard.should_freeze_motion(feat.left_blink, feat.right_blink, now_ts)

            if args.mouth_open_click:
                click_action, mouth_state = mouth_detector.update(feat.mouth_open, now_ts)
                if mouse is not None and click_action == "single":
                    mouse.left_click()
                elif mouse is not None and click_action == "double":
                    mouse.double_click()

            if use_hybrid:
                _, eye_vec, head_vec = split_feature_vector(feat.vector)
                if eye_vec.shape[0] != eye_feature_dim or head_vec.shape[0] != head_feature_dim:
                    if args.control_mouse:
                        blink_guard.reset()
                    if args.mouth_open_click:
                        mouth_detector.reset()
                    gaze_filter.reset(smooth_x, smooth_y)
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
                px = (1.0 - head_assist) * px_eye + head_assist * px_head
                py = (1.0 - head_assist) * py_eye + head_assist * py_head
            else:
                px, py = predict_xy(feat.vector, W, feature_mode=feature_mode)
                px, py = apply_model_correction(px, py, corr_full)

            px *= current_w / max(float(model_screen_w), 1.0)
            py *= current_h / max(float(model_screen_h), 1.0)
            px = (px - current_w / 2.0) * float(args.x_gain) + current_w / 2.0
            py = (py - current_h / 2.0) * float(args.y_gain) + current_h / 2.0
            px, py, edge_assist_active = apply_edge_assist(
                px,
                py,
                current_w,
                current_h,
                zone_ratio=args.edge_zone_ratio,
                max_push_px=args.edge_boost_px,
            )

            px, py = clamp_xy(px, py, current_w, current_h)
            smooth_x, smooth_y = gaze_filter.update(px, py, freeze_motion=freeze_motion)
            tracking_ready = True

            request_snap = (
                args.snap_clickables
                and args.control_mouse
                and args.mouth_open_click
                and not freeze_motion
                and not edge_assist_active
                and mouth_state in {"armed", "open", "double"}
            )
            if args.snap_clickables and args.control_mouse:
                if edge_assist_active:
                    snapper.reset()
                smooth_x, smooth_y, snapped_to_control, snap_label = snapper.apply(
                    smooth_x,
                    smooth_y,
                    request_snap=request_snap,
                    now=now_ts,
                )

            mode_text = f"hybrid:{head_assist:.2f}" if use_hybrid else "eye-only"
            preview = draw_status(preview, f"Gaze: ({smooth_x}, {smooth_y}) {mode_text}")
            cv2.putText(
                preview,
                f"blink L:{feat.left_blink:.2f} R:{feat.right_blink:.2f} mouth:{feat.mouth_open:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (40, 40, 40),
                2,
            )
            if freeze_motion:
                cv2.putText(preview, "Blink hold", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 120, 220), 2)
            if args.mouth_open_click:
                mouth_color = (0, 180, 0) if click_action is not None else (50, 50, 50)
                cv2.putText(preview, f"Mouth click: {mouth_state}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mouth_color, 2)
            if edge_assist_active:
                cv2.putText(preview, "Edge assist", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 90, 20), 2)
            if snapped_to_control:
                cv2.putText(preview, f"Snap: {snap_label}", (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 80, 20), 2)

            if args.control_mouse and mouse is not None:
                cursor_delta = float(np.hypot(smooth_x - last_cursor_x, smooth_y - last_cursor_y))
                if not freeze_motion and cursor_delta >= args.cursor_deadzone_px:
                    mouse.move_to(smooth_x, smooth_y)
                    last_cursor_x = smooth_x
                    last_cursor_y = smooth_y

            if show_overlay:
                cv2.circle(canvas, (smooth_x, smooth_y), 16, (0, 0, 255), -1)
                cv2.circle(canvas, (smooth_x, smooth_y), 30, (0, 100, 255), 2)

        if show_overlay:
            overlay_text = "Mouse mode ON" if args.control_mouse else "Mouse mode OFF"
            cv2.putText(canvas, overlay_text, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2)
            cv2.putText(canvas, "Press ESC to quit", (40, current_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
        elif preview_visible:
            draw_gaze_minimap(preview, smooth_x, smooth_y, current_w, current_h, active=tracking_ready, mouse_mode=args.control_mouse)

        if preview_visible:
            cv2.imshow("Webcam", preview)
        if show_overlay:
            cv2.imshow("Gaze Dot", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()





