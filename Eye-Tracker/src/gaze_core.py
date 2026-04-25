import ctypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarkerResult,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode
import numpy as np

MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"

EYE_LANDMARK_COUNT = 10
FACE_LANDMARK_COUNT = 6
FLAT_FEATURE_DIM = (EYE_LANDMARK_COUNT + FACE_LANDMARK_COUNT) * 2
EXTRA_FEATURE_DIM = 3


@dataclass
class FrameFeatures:
    vector: np.ndarray
    frame: np.ndarray
    left_blink: float
    right_blink: float
    mouth_open: float

    @property
    def mean_blink(self) -> float:
        return float((self.left_blink + self.right_blink) / 2.0)


def get_screen_size() -> Tuple[int, int]:
    user32 = ctypes.windll.user32
    return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))


def build_face_mesh():
    base_options = BaseOptions(model_asset_path=str(MODEL_PATH))
    options = FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.7,
    )
    return FaceLandmarker.create_from_options(options)


def _landmark_xy(landmarks, idx: int, width: int, height: int) -> Tuple[float, float]:
    p = landmarks[idx]
    return p.x * width, p.y * height


def _normalize_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    out = points.copy()
    out[:, 0] /= float(width)
    out[:, 1] /= float(height)
    return out.reshape(-1)


def split_feature_vector(vector: np.ndarray):
    flat = vector[:FLAT_FEATURE_DIM]
    left_blink = vector[FLAT_FEATURE_DIM]
    right_blink = vector[FLAT_FEATURE_DIM + 1]
    wh_ratio = vector[FLAT_FEATURE_DIM + 2]

    eye_flat_dim = EYE_LANDMARK_COUNT * 2
    eye_flat = flat[:eye_flat_dim]
    face_flat = flat[eye_flat_dim:]

    eye_vec = np.concatenate([eye_flat, np.array([left_blink, right_blink, wh_ratio], dtype=np.float32)])
    head_vec = np.concatenate([face_flat, np.array([wh_ratio], dtype=np.float32)])
    return vector, eye_vec.astype(np.float32), head_vec.astype(np.float32)


def extract_features(frame: np.ndarray, face_mesh) -> Optional[FrameFeatures]:
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(time.monotonic() * 1000)
    results = face_mesh.detect_for_video(mp_img, timestamp_ms)
    if not results.face_landmarks:
        return None

    lm = results.face_landmarks[0]

    # Stable subset around eyes + key face anchors.
    indices = [
        33, 133, 159, 145,  # left eye
        362, 263, 386, 374,  # right eye
        468, 473,  # iris centers
        1, 4,  # nose bridge / tip
        61, 291,  # mouth corners (head orientation cue)
        10, 152,  # forehead/chin
    ]

    pts = np.array([_landmark_xy(lm, i, w, h) for i in indices], dtype=np.float32)

    # Eye geometry features.
    left_w = np.linalg.norm(pts[0] - pts[1]) + 1e-6
    right_w = np.linalg.norm(pts[4] - pts[5]) + 1e-6
    left_h = np.linalg.norm(pts[2] - pts[3])
    right_h = np.linalg.norm(pts[6] - pts[7])
    left_blink = left_h / left_w
    right_blink = right_h / right_w
    upper_lip = np.array(_landmark_xy(lm, 13, w, h), dtype=np.float32)
    lower_lip = np.array(_landmark_xy(lm, 14, w, h), dtype=np.float32)
    mouth_w = np.linalg.norm(pts[12] - pts[13]) + 1e-6
    mouth_h = np.linalg.norm(upper_lip - lower_lip)
    mouth_open = mouth_h / mouth_w

    flat = _normalize_points(pts, w, h)
    extra = np.array([
        left_blink,
        right_blink,
        float(w) / max(float(h), 1.0),
    ], dtype=np.float32)

    return FrameFeatures(
        vector=np.concatenate([flat, extra]),
        frame=frame,
        left_blink=float(left_blink),
        right_blink=float(right_blink),
        mouth_open=float(mouth_open),
    )


def draw_status(frame: np.ndarray, text: str, ok: bool = True) -> np.ndarray:
    color = (246, 130, 59) if ok else (38, 38, 220)
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return frame