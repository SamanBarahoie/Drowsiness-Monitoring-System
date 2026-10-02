"""
Usage
-----
    python drowsiness_monitor.py --input video.mp4 --output out.mp4 --show
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import platform
import queue
import sys
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_TQDM = False


LOGGER = logging.getLogger("drowsiness_monitor")


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class Config:
    # A path to a video file, OR an integer webcam index (e.g. 0) for a
    # live camera feed. "0", "1", ... typed on the CLI are auto-converted.
    input_video: Union[str, int] = "P1042789_na.mp4"
    output_video: str = "result_ml.mp4"
    output_csv: str = "drowsiness_ml_log.csv"
    model_path: str = "face_landmarker.task"
    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    )
    tflite_model_path: str = "model/best_mobilenetv3_drowsiness.tflite"

    # ML optimisation: run the heavier MobileNetV3 model every N frames only.
    ml_inference_interval: int = 3

    # Optional down-scale of the frame handed to the face landmarker.
    # Set to None to always use full resolution.
    detection_max_width: Optional[int] = 960

    # Thresholds (unchanged from the original pipeline).
    blink_threshold: float = 0.42
    ear_ratio_threshold: float = 0.70
    eye_warning_sec: float = 0.50
    eye_alert_sec: float = 1.00
    microsleep_eye_min_sec: float = 0.25
    microsleep_head_min_sec: float = 0.25
    head_drop_speed_thres: float = 30.0
    head_drop_angle_thres: float = 8.0
    head_down_angle_thres: float = 16.0
    perclos_window_sec: float = 30.0
    perclos_warning: float = 0.20
    perclos_alert: float = 0.30

    jaw_open_threshold: float = 0.45
    yawn_min_sec: float = 0.80
    yawn_lookback_window: float = 120.0

    alert_score: int = 70
    warning_score: int = 40
    beep_interval_sec: float = 1.2

    show_preview: bool = False
    codec: str = "mp4v"

    # Real-time behaviour -----------------------------------------------
    # When True, the preview window paces itself to the *source* frame
    # rate (via a small sleep each iteration) instead of blasting through
    # frames as fast as the CPU allows — this is what makes a video *file*
    # look like a live feed. A webcam is already real-time by nature, so
    # pacing is skipped automatically when the source reports no usable
    # duration/fps mismatch worth compensating for.
    realtime_playback: bool = True
    # Skip writing the annotated video / CSV log entirely — useful for a
    # pure "just show me the live monitor" session (webcam or file).
    save_output: bool = True


# =============================================================================
# CROSS-PLATFORM, NON-BLOCKING AUDIO ALERT
# =============================================================================
class AudioAlerter:
    """Fires a short beep without blocking the main processing loop."""

    def __init__(self, min_interval_sec: float):
        self._min_interval = min_interval_sec
        self._last_time = 0.0
        self._lock = threading.Lock()

    def trigger(self, frequency: int = 2000, duration_ms: int = 350) -> None:
        now = time.time()
        with self._lock:
            if now - self._last_time < self._min_interval:
                return
            self._last_time = now
        threading.Thread(
            target=self._play, args=(frequency, duration_ms), daemon=True
        ).start()

    @staticmethod
    def _play(frequency: int, duration_ms: int) -> None:
        try:
            if platform.system() == "Windows":
                import winsound

                winsound.Beep(frequency, duration_ms)
            else:
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:  # pragma: no cover - best-effort alert
            pass


# =============================================================================
# 1D KALMAN FILTER
# =============================================================================
class KalmanFilter1D:
    """Simple scalar Kalman filter used to smooth EAR and head-pose signals."""

    __slots__ = ("q", "r", "x", "p")

    def __init__(
        self, process_noise: float = 1e-3, measurement_noise: float = 1e-1,
        initial_value: float = 0.0,
    ):
        self.q = process_noise
        self.r = measurement_noise
        self.x = initial_value
        self.p = 1.0

    def update(self, measurement: float) -> float:
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (measurement - self.x)
        self.p = (1 - k) * self.p
        return float(self.x)


# =============================================================================
# ADAPTIVE EAR / HEAD-POSE BASELINE TRACKER
# =============================================================================
class AdaptiveDriverTracker:
    """Dynamically calibrates baseline EAR and neutral head pitch.

    The percentile/median recalibration is the expensive part of this class
    (it sorts up to 300 samples). Since the result only feeds a slow-moving
    EMA anyway, it is recomputed every ``recalibration_stride`` frames rather
    than on every single frame — this is visually indistinguishable but
    meaningfully cheaper on long videos.
    """

    def __init__(
        self,
        ema_alpha: float = 0.005,
        min_ear_baseline: float = 0.20,
        default_ear: float = 0.28,
        recalibration_stride: int = 5,
    ):
        self.alpha = ema_alpha
        self.min_ear_baseline = min_ear_baseline
        self.ear_baseline = default_ear
        self.head_pitch_baseline = 0.0
        self.ear_window: Deque[float] = deque(maxlen=300)
        self.pitch_window: Deque[float] = deque(maxlen=300)

        self.kf_ear = KalmanFilter1D(1e-4, 1e-2, initial_value=default_ear)
        self.kf_pitch = KalmanFilter1D(1e-3, 5e-2)
        self.kf_yaw = KalmanFilter1D(1e-3, 5e-2)
        self.kf_roll = KalmanFilter1D(1e-3, 5e-2)

        self._stride = max(1, recalibration_stride)
        self._tick = 0
        self._cached_ear_pctl = default_ear
        self._cached_pitch_median = 0.0

    def process_measurements(
        self,
        raw_ear: float,
        raw_pitch: float,
        raw_yaw: float,
        raw_roll: float,
        is_eyes_open_candidate: bool,
    ) -> Tuple[float, float, float, float, float, float]:
        smooth_ear = self.kf_ear.update(raw_ear)
        smooth_pitch = self.kf_pitch.update(raw_pitch)
        smooth_yaw = self.kf_yaw.update(raw_yaw)
        smooth_roll = self.kf_roll.update(raw_roll)

        self._tick += 1
        due_for_recalc = self._tick % self._stride == 0

        if raw_ear > 0.12:
            self.ear_window.append(raw_ear)

        if len(self.ear_window) >= 45:
            if due_for_recalc:
                self._cached_ear_pctl = float(np.percentile(self.ear_window, 85))
            self.ear_baseline = (
                (1 - self.alpha) * self.ear_baseline + self.alpha * self._cached_ear_pctl
            )
            self.ear_baseline = max(self.ear_baseline, self.min_ear_baseline)

        if is_eyes_open_candidate:
            self.pitch_window.append(smooth_pitch)

        if len(self.pitch_window) >= 45:
            if due_for_recalc:
                self._cached_pitch_median = float(np.median(self.pitch_window))
            self.head_pitch_baseline = (
                (1 - self.alpha) * self.head_pitch_baseline
                + self.alpha * self._cached_pitch_median
            )

        return (
            smooth_ear,
            smooth_pitch,
            smooth_yaw,
            smooth_roll,
            self.ear_baseline,
            self.head_pitch_baseline,
        )


# =============================================================================
# TFLITE MOBILENETV3 DROWSINESS MODEL
# =============================================================================
class DrowsinessMLModel:

    def __init__(self, model_path: str):
        self.available = False
        self.interpreter: Optional[tf.lite.Interpreter] = None
        self.input_details = None
        self.output_details = None
        self._input_size: Tuple[int, int] = (224, 224)

        if not os.path.exists(model_path):
            LOGGER.warning(
                "TFLite model not found at '%s'. ML inference will be skipped.",
                model_path,
            )
            return

        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            shape = self.input_details[0]["shape"]
            if len(shape) == 4:
                self._input_size = (int(shape[2]), int(shape[1]))  # (w, h)
            self.available = True
            LOGGER.info("TFLite MobileNetV3 model loaded from '%s'.", model_path)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to load TFLite model '%s': %s", model_path, exc)

    def predict(self, face_crop: Optional[np.ndarray]) -> float:
        if not self.available or face_crop is None or face_crop.size == 0:
            return 0.0

        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(face_rgb, self._input_size)
        batch = np.expand_dims(resized.astype(np.float32), axis=0)
        batch = tf.keras.applications.mobilenet_v3.preprocess_input(batch)

        self.interpreter.set_tensor(self.input_details[0]["index"], batch)
        self.interpreter.invoke()
        preds = self.interpreter.get_tensor(self.output_details[0]["index"])

        return float(preds[0][0]) if preds.shape[-1] == 1 else float(preds[0][1])


# =============================================================================
# FACE GEOMETRY (vectorised, computed once per frame and reused everywhere)
# =============================================================================
LEFT_EYE_IDX = np.array([362, 385, 387, 263, 373, 380])
RIGHT_EYE_IDX = np.array([33, 160, 158, 133, 153, 144])
HEAD_LANDMARK_IDS = np.array([1, 152, 33, 263, 61, 291])

HEAD_POINTS_3D = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0],
    ],
    dtype=np.float64,
)


def landmarks_to_pixels(landmarks, width: int, height: int) -> np.ndarray:
    """Convert every MediaPipe landmark to pixel coordinates in one pass.

    The result is reused for EAR, head pose, cropping and drawing instead of
    each of those re-deriving pixel coordinates independently.
    """
    pts = np.empty((len(landmarks), 2), dtype=np.float64)
    for i, lm in enumerate(landmarks):
        pts[i, 0] = lm.x * width
        pts[i, 1] = lm.y * height
    return pts


def eye_aspect_ratio(pts: np.ndarray, idx: np.ndarray) -> float:
    p = pts[idx]
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    return float((v1 + v2) / (2.0 * h)) if h > 1e-6 else 0.0


def calculate_head_pose(
    pts: np.ndarray, width: int, height: int
) -> Tuple[float, float, float, bool]:
    try:
        image_points = pts[HEAD_LANDMARK_IDS].astype(np.float64)
        camera_matrix = np.array(
            [[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        ok, rot_vec, _trans_vec = cv2.solvePnP(
            HEAD_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return 0.0, 0.0, 0.0, False

        rot_mat, _ = cv2.Rodrigues(rot_vec)
        sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
        if sy >= 1e-6:
            pitch = np.degrees(np.arctan2(rot_mat[2, 1], rot_mat[2, 2]))
            yaw = np.degrees(np.arctan2(-rot_mat[2, 0], sy))
            roll = np.degrees(np.arctan2(rot_mat[1, 0], rot_mat[0, 0]))
        else:
            pitch = np.degrees(np.arctan2(-rot_mat[1, 2], rot_mat[1, 1]))
            yaw = np.degrees(np.arctan2(-rot_mat[2, 0], sy))
            roll = 0.0
        return float(pitch), float(yaw), float(roll), True
    except Exception:  # pragma: no cover - defensive, mirrors original behaviour
        return 0.0, 0.0, 0.0, False


def crop_face_region(
    frame: np.ndarray, pts: np.ndarray, width: int, height: int,
    padding_ratio: float = 0.2,
) -> Optional[np.ndarray]:
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    w, h = x2 - x1, y2 - y1
    pad_w, pad_h = w * padding_ratio, h * padding_ratio

    x1 = max(0, int(x1 - pad_w))
    y1 = max(0, int(y1 - pad_h))
    x2 = min(width, int(x2 + pad_w))
    y2 = min(height, int(y2 + pad_h))

    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def get_blendshape_scores(result) -> Dict[str, float]:
    if not result.face_blendshapes:
        return {}
    return {cat.category_name: float(cat.score) for cat in result.face_blendshapes[0]}


# =============================================================================
# LOW-LIGHT ENHANCEMENT (CLAHE + adaptive gamma, both cached across frames)
# =============================================================================
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_GAMMA_LUT_CACHE: Dict[float, np.ndarray] = {}


def _gamma_lut(gamma: float) -> np.ndarray:
    lut = _GAMMA_LUT_CACHE.get(gamma)
    if lut is None:
        lut = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)]
        ).astype(np.uint8)
        _GAMMA_LUT_CACHE[gamma] = lut
    return lut


def enhance_low_light(frame: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))

    denoised = cv2.GaussianBlur(frame, (3, 3), 0)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    if brightness < 35:
        gamma = 0.50
    elif brightness < 50:
        gamma = 0.60
    elif brightness < 70:
        gamma = 0.72
    elif brightness < 90:
        gamma = 0.82
    elif brightness < 110:
        gamma = 0.92
    else:
        gamma = 1.0

    if gamma != 1.0:
        enhanced = cv2.LUT(enhanced, _gamma_lut(gamma))

    return enhanced, brightness, brightness < 85


# =============================================================================
# UI RENDERING (Tesla / Autopilot inspired dashboard)
# =============================================================================
class PanelMaskCache:
    """Rounded-rectangle alpha masks depend only on (w, h, radius), never on
    frame content or color, so they are built once and reused forever."""

    _cache: Dict[Tuple[int, int, int], np.ndarray] = {}

    @classmethod
    def get(cls, w: int, h: int, radius: int) -> np.ndarray:
        key = (w, h, radius)
        mask = cls._cache.get(key)
        if mask is None:
            radius = max(0, min(radius, w // 2, h // 2))
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask, (radius, 0), (w - radius, h), 255, -1)
            cv2.rectangle(mask, (0, radius), (w, h - radius), 255, -1)
            for cx, cy in (
                (radius, radius),
                (w - radius, radius),
                (radius, h - radius),
                (w - radius, h - radius),
            ):
                cv2.circle(mask, (cx, cy), radius, 255, -1)
            cls._cache[key] = mask
        return mask


def draw_panel(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    radius: int = 12,
    alpha: float = 0.82,
    border_color: Optional[Tuple[int, int, int]] = None,
    shadow: bool = True,
) -> None:
    """Draw a translucent rounded panel using a cached mask (fast) with an
    optional soft drop shadow and border for a bit of visual depth."""
    x1, y1 = pt1
    x2, y2 = pt2
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1:
        return

    if shadow:
        sx1, sy1 = x1 + 4, y1 + 5
        sx2, sy2 = min(w_img, x2 + 4), min(h_img, y2 + 5)
        sw, sh = sx2 - sx1, sy2 - sy1
        if sw > 1 and sh > 1:
            shadow_mask = PanelMaskCache.get(sw, sh, radius)
            sub = img[sy1:sy2, sx1:sx2].astype(np.float32)
            mask_f = (shadow_mask.astype(np.float32) / 255.0 * 0.35)[..., None]
            img[sy1:sy2, sx1:sx2] = (sub * (1 - mask_f)).astype(np.uint8)

    mask = PanelMaskCache.get(w, h, radius)
    sub = img[y1:y2, x1:x2].astype(np.float32)
    color_layer = np.full_like(sub, color, dtype=np.float32)
    mask_f = (mask.astype(np.float32) / 255.0 * alpha)[..., None]
    img[y1:y2, x1:x2] = (sub * (1 - mask_f) + color_layer * mask_f).astype(np.uint8)

    if border_color is not None:
        _draw_rounded_border(img, (x1, y1), (x2, y2), border_color, radius)


def _draw_rounded_border(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    radius: int,
    thickness: int = 1,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)


_SCORE_GREEN = np.array([100.0, 210.0, 120.0])
_SCORE_AMBER = np.array([60.0, 175.0, 245.0])
_SCORE_RED = np.array([60.0, 50.0, 235.0])


def score_to_color(score: float) -> Tuple[int, int, int]:
    """Continuous green -> amber -> red gradient (BGR) driven by the score,
    instead of three abruptly-switching flat color bands."""
    score = float(np.clip(score, 0, 100))
    if score <= 40:
        t = score / 40.0
        c = _SCORE_GREEN * (1 - t) + _SCORE_AMBER * t
    else:
        t = (score - 40) / 60.0
        c = _SCORE_AMBER * (1 - t) + _SCORE_RED * t
    return int(c[0]), int(c[1]), int(c[2])


def draw_sparkline(
    img: np.ndarray,
    origin: Tuple[int, int],
    size: Tuple[int, int],
    values: Deque[float],
    max_value: float = 100.0,
    color: Tuple[int, int, int] = (210, 215, 225),
) -> None:
    """Tesla-style trend line of recent drowsiness scores."""
    if len(values) < 2:
        return
    x, y = origin
    w, h = size
    vals = np.clip(np.array(values, dtype=np.float32), 0, max_value)
    xs = np.linspace(x, x + w, len(vals)).astype(np.int32)
    ys = (y + h - (vals / max_value) * h).astype(np.int32)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)


def draw_top_badge(
    img: np.ndarray, status_text: str, accent_color: Tuple[int, int, int],
    width: int, pulse_phase: float,
) -> None:
    badge_w, badge_h = 400, 44
    x1 = (width - badge_w) // 2
    y1 = 20
    x2, y2 = x1 + badge_w, y1 + badge_h

    draw_panel(img, (x1, y1), (x2, y2), (20, 22, 28), radius=11,
               border_color=(60, 65, 75))

    # Gently pulsing status dot instead of a flat circle.
    pulse = 0.65 + 0.35 * (0.5 + 0.5 * np.sin(pulse_phase))
    dot_color = tuple(int(c * pulse) for c in accent_color)
    cv2.circle(img, (x1 + 26, y1 + 22), 5, dot_color, -1, cv2.LINE_AA)

    cv2.putText(img, "DRIVER MONITOR", (x1 + 42, y1 + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (165, 172, 182), 1, cv2.LINE_AA)
    cv2.putText(img, "|", (x1 + 185, y1 + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 88, 98), 1, cv2.LINE_AA)
    cv2.putText(img, status_text, (x1 + 200, y1 + 27),
                cv2.FONT_HERSHEY_DUPLEX, 0.48, accent_color, 1, cv2.LINE_AA)


def draw_telemetry_card(
    img: np.ndarray,
    score: int,
    ml_prob: float,
    perclos: float,
    ear_baseline: float,
    pitch: float,
    night_mode: bool,
    yawn_count: int,
    score_history: Deque[float],
) -> None:
    card_w, card_h = 320, 210
    x1, y1 = 20, 20
    x2, y2 = x1 + card_w, y1 + card_h
    draw_panel(img, (x1, y1), (x2, y2), (18, 20, 26), radius=13,
               border_color=(45, 48, 56))

    bar_x, bar_y = x1 + 20, y1 + 24
    bar_w, bar_h = 280, 7
    fill_w = int(bar_w * (min(100, score) / 100.0))
    bar_color = score_to_color(score)

    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (45, 50, 60), -1, cv2.LINE_AA)
    if fill_w > 0:
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      bar_color, -1, cv2.LINE_AA)

    cv2.putText(img, "DROWSINESS INDEX", (x1 + 20, y1 + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 148, 158), 1, cv2.LINE_AA)
    cv2.putText(img, f"{score}%", (x1 + 265, y1 + 48),
                cv2.FONT_HERSHEY_DUPLEX, 0.46, (240, 245, 250), 1, cv2.LINE_AA)

    cv2.putText(img, f"MobileNetV3 Prob: {ml_prob * 100:.1f}%", (x1 + 20, y1 + 73),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"PERCLOS: {perclos:.2f}", (x1 + 20, y1 + 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 205, 215), 1, cv2.LINE_AA)
    cv2.putText(img, f"EAR Base: {ear_baseline:.2f}", (x1 + 165, y1 + 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 205, 215), 1, cv2.LINE_AA)

    pitch_txt = f"{pitch:+.1f} deg" if pitch != 0 else "0.0 deg"
    cv2.putText(img, f"Head Pitch: {pitch_txt}", (x1 + 20, y1 + 123),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 205, 215), 1, cv2.LINE_AA)
    cv2.putText(img, f"Yawns (2m): {yawn_count}", (x1 + 165, y1 + 123),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 180, 100), 1, cv2.LINE_AA)

    # Trend sparkline of the drowsiness score.
    cv2.putText(img, "TREND", (x1 + 20, y1 + 146),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 128, 138), 1, cv2.LINE_AA)
    draw_sparkline(img, (x1 + 20, y1 + 152), (280, 30), score_history,
                   color=bar_color)

    mode_txt = "NIGHT CAM" if night_mode else "DAY CAM"
    cv2.putText(img, mode_txt, (x1 + 20, y1 + 196),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 160, 190), 1, cv2.LINE_AA)


def draw_face_vector(
    img: np.ndarray, pts: np.ndarray, status_color: Tuple[int, int, int]
) -> None:
    xs = pts[:, 0]
    ys = pts[:, 1]
    x1, x2 = int(xs.min()) - 10, int(xs.max()) + 10
    y1, y2 = int(ys.min()) - 10, int(ys.max()) + 10

    line_len = 12
    for (cx, cy), (dx1, dy1), (dx2, dy2) in (
        ((x1, y1), (line_len, 0), (0, line_len)),
        ((x2, y1), (-line_len, 0), (0, line_len)),
        ((x1, y2), (line_len, 0), (0, -line_len)),
        ((x2, y2), (-line_len, 0), (0, -line_len)),
    ):
        cv2.line(img, (cx, cy), (cx + dx1, cy + dy1), status_color, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx + dx2, cy + dy2), status_color, 1, cv2.LINE_AA)

    for idx in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
        center = tuple(pts[idx].mean(axis=0).astype(int))
        cv2.circle(img, center, 2, (255, 255, 255), -1, cv2.LINE_AA)


def draw_critical_vignette(img: np.ndarray, pulse_phase: float) -> None:
    """A pulsing red border drawn around the whole frame during a critical
    (micro-sleep / high score) alert — cheap and very noticeable."""
    h, w = img.shape[:2]
    pulse = 0.5 + 0.5 * np.sin(pulse_phase)
    thickness = int(5 + pulse * 9)
    color = (50, 40, 220)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, thickness, cv2.LINE_AA)


# =============================================================================
# THREADED VIDEO FRAME READER (overlaps decoding with processing)
# =============================================================================
class ThreadedFrameReader:
    def __init__(self, cap: cv2.VideoCapture, queue_size: int = 8):
        self._cap = cap
        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "ThreadedFrameReader":
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stopped.is_set():
            ret, frame = self._cap.read()
            if not ret:
                self._queue.put(None)
                return
            self._queue.put(frame)

    def read(self) -> Optional[np.ndarray]:
        return self._queue.get()

    def stop(self) -> None:
        self._stopped.set()


# =============================================================================
# PER-FRAME METRICS CONTAINER
# =============================================================================
@dataclass
class FrameMetrics:
    face_found: bool = False
    brightness: float = 0.0
    night_mode: bool = False
    ml_prob: float = 0.0
    left_ear: float = 0.0
    right_ear: float = 0.0
    ear_baseline: float = 0.0
    perclos: float = 0.0
    pitch: float = 0.0
    pitch_baseline: float = 0.0
    head_drop_speed: float = 0.0
    eye_closed_duration: float = 0.0
    yawn_duration: float = 0.0
    yawn_count: int = 0
    head_down_duration: float = 0.0
    micro_sleep: bool = False
    score: int = 0
    status: str = "ATTENTIVE"
    accent_color: Tuple[int, int, int] = (120, 210, 100)


def ensure_model_downloaded(model_path: str, model_url: str) -> None:
    if os.path.exists(model_path):
        return
    LOGGER.info("Downloading MediaPipe Face Landmarker model...")
    try:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        urllib.request.urlretrieve(model_url, model_path)
        LOGGER.info("Model downloaded to '%s'.", model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download the face landmarker model from {model_url}: {exc}"
        ) from exc


# =============================================================================
# MAIN PIPELINE
# =============================================================================
class DrowsinessMonitor:
    def __init__(self, config: Config):
        self.cfg = config
        self.tracker = AdaptiveDriverTracker()
        self.ml_model = DrowsinessMLModel(config.tflite_model_path)
        self.alerter = AudioAlerter(config.beep_interval_sec)

        # Rolling state (previously module-level globals).
        self.last_ml_prob = 0.0
        self.closed_start: Optional[float] = None
        self.yawn_start: Optional[float] = None
        self.head_down_start: Optional[float] = None
        self.yawn_logged = False

        self.perclos_history: Deque[Tuple[float, bool]] = deque()
        self.pitch_time_history: Deque[Tuple[float, float]] = deque(maxlen=5)
        self.yawn_history: Deque[float] = deque()
        self.score_history: Deque[float] = deque(maxlen=150)  # ~ trend window

        self.frame_index = 0
        self._pulse_clock = 0.0

    # ------------------------------------------------------------------
    def _detection_frame(self, enhanced_frame: np.ndarray, width: int, height: int):
        """Optionally down-scale the frame fed to the landmarker for speed.
        Landmarks are normalized, so this has no effect on downstream pixel
        calculations, which always use the full-resolution width/height."""
        max_w = self.cfg.detection_max_width
        if not max_w or width <= max_w:
            return enhanced_frame
        scale = max_w / float(width)
        new_size = (max_w, max(1, int(height * scale)))
        return cv2.resize(enhanced_frame, new_size, interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------
    def _update_yawn_history(self, video_time: float) -> int:
        while self.yawn_history and (
            video_time - self.yawn_history[0] > self.cfg.yawn_lookback_window
        ):
            self.yawn_history.popleft()
        return len(self.yawn_history)

    # ------------------------------------------------------------------
    def _process_face(
        self, landmarker_result, enhanced_frame: np.ndarray, width: int, height: int,
        video_time: float,
    ) -> FrameMetrics:
        cfg = self.cfg
        m = FrameMetrics(face_found=True)

        landmarks = landmarker_result.face_landmarks[0]
        pts = landmarks_to_pixels(landmarks, width, height)
        scores = get_blendshape_scores(landmarker_result)

        # --- MobileNetV3 inference (throttled) ---------------------------
        if self.frame_index % cfg.ml_inference_interval == 0:
            face_crop = crop_face_region(enhanced_frame, pts, width, height)
            if face_crop is not None:
                self.last_ml_prob = self.ml_model.predict(face_crop)
        m.ml_prob = self.last_ml_prob

        left_blink = scores.get("eyeBlinkLeft", 0.0)
        right_blink = scores.get("eyeBlinkRight", 0.0)
        jaw_open = scores.get("jawOpen", 0.0)
        mouth_close = scores.get("mouthClose", 0.0)

        m.left_ear = eye_aspect_ratio(pts, LEFT_EYE_IDX)
        m.right_ear = eye_aspect_ratio(pts, RIGHT_EYE_IDX)
        raw_avg_ear = (m.left_ear + m.right_ear) / 2.0

        raw_pitch, raw_yaw, raw_roll, pose_ok = calculate_head_pose(pts, width, height)

        (
            smooth_ear, smooth_pitch, _smooth_yaw, _smooth_roll,
            ear_baseline, pitch_baseline,
        ) = self.tracker.process_measurements(
            raw_avg_ear, raw_pitch, raw_yaw, raw_roll,
            is_eyes_open_candidate=(raw_avg_ear > 0.20),
        )
        m.ear_baseline = ear_baseline
        m.pitch_baseline = pitch_baseline
        m.pitch = smooth_pitch if pose_ok else 0.0

        ear_closed = smooth_ear < (ear_baseline * cfg.ear_ratio_threshold)
        blink_closed = ((left_blink + right_blink) / 2.0) > cfg.blink_threshold
        both_eyes_closed = ear_closed and blink_closed

        if both_eyes_closed:
            if self.closed_start is None:
                self.closed_start = video_time
            m.eye_closed_duration = video_time - self.closed_start
        else:
            self.closed_start = None

        self.perclos_history.append((video_time, both_eyes_closed))
        while self.perclos_history and (
            video_time - self.perclos_history[0][0] > cfg.perclos_window_sec
        ):
            self.perclos_history.popleft()
        if self.perclos_history:
            closed_count = sum(1 for _, state in self.perclos_history if state)
            m.perclos = closed_count / len(self.perclos_history)

        # --- Yawn detection ------------------------------------------------
        is_yawn_frame = (jaw_open > cfg.jaw_open_threshold) and (mouth_close < 0.25)
        if is_yawn_frame:
            if self.yawn_start is None:
                self.yawn_start = video_time
                self.yawn_logged = False
            m.yawn_duration = video_time - self.yawn_start
            if m.yawn_duration >= cfg.yawn_min_sec and not self.yawn_logged:
                self.yawn_history.append(video_time)
                self.yawn_logged = True
        else:
            self.yawn_start = None

        m.yawn_count = self._update_yawn_history(video_time)

        # --- Head pose based events -----------------------------------
        head_down = False
        head_drop = False
        pitch_change = 0.0
        head_drop_speed = 0.0

        if pose_ok:
            pitch_change = smooth_pitch - pitch_baseline
            head_down = pitch_change > cfg.head_down_angle_thres

            if head_down:
                if self.head_down_start is None:
                    self.head_down_start = video_time
                m.head_down_duration = video_time - self.head_down_start
            else:
                self.head_down_start = None

            self.pitch_time_history.append((video_time, smooth_pitch))
            if len(self.pitch_time_history) >= 2:
                dt = video_time - self.pitch_time_history[0][0]
                if dt > 0.04:
                    head_drop_speed = (
                        smooth_pitch - self.pitch_time_history[0][1]
                    ) / dt

            head_drop = (
                pitch_change > cfg.head_drop_angle_thres
                and head_drop_speed > cfg.head_drop_speed_thres
            )
        else:
            self.head_down_start = None

        m.head_drop_speed = head_drop_speed

        # --- Micro-sleep detection --------------------------------------
        cond_a = (
            both_eyes_closed
            and m.eye_closed_duration >= cfg.microsleep_eye_min_sec
            and head_down
        )
        cond_b = (
            both_eyes_closed
            and m.eye_closed_duration >= cfg.microsleep_eye_min_sec
            and head_drop
        )
        cond_c = both_eyes_closed and m.eye_closed_duration >= 0.50
        m.micro_sleep = cond_a or cond_b or cond_c

        # --- Rule-based score --------------------------------------------
        rule_score = 0
        if m.eye_closed_duration >= cfg.eye_warning_sec:
            rule_score += 30
        if m.eye_closed_duration >= cfg.eye_alert_sec:
            rule_score += 25
        if m.micro_sleep:
            rule_score += 45
        if m.perclos >= cfg.perclos_warning:
            rule_score += 15
        if m.perclos >= cfg.perclos_alert:
            rule_score += 15
        if m.yawn_duration >= cfg.yawn_min_sec:
            rule_score += 15
        if m.yawn_count >= 2:
            rule_score += 15
        if m.head_down_duration >= 0.5:
            rule_score += 10

        ml_score = int(m.ml_prob * 100)

        if self.ml_model.available:
            fused_score = int(0.45 * rule_score + 0.55 * ml_score)
        else:
            fused_score = rule_score

        # Safety floor: in clearly dangerous states the model cannot pull
        # the score below a minimum, no matter what it predicts.
        if m.micro_sleep:
            fused_score = max(fused_score, 90)
        elif m.eye_closed_duration >= cfg.eye_alert_sec:
            fused_score = max(fused_score, 75)

        m.score = int(np.clip(fused_score, 0, 100))
        return m

    # ------------------------------------------------------------------
    def _classify_status(self, m: FrameMetrics) -> None:
        cfg = self.cfg
        if not m.face_found:
            m.status, m.accent_color = "NO DRIVER DETECTED", (130, 138, 148)
        elif m.micro_sleep:
            m.status, m.accent_color = "CRITICAL: MICRO-SLEEP", (60, 50, 235)
        elif m.score >= cfg.alert_score:
            m.status, m.accent_color = "TAKE A BREAK", (60, 50, 235)
        elif m.yawn_duration >= cfg.yawn_min_sec or m.yawn_count >= 2:
            m.status, m.accent_color = "FREQUENT YAWNING DETECTED", (60, 175, 245)
        elif m.score >= cfg.warning_score:
            m.status, m.accent_color = "EARLY DROWSINESS", (60, 175, 245)
        else:
            m.status, m.accent_color = "ATTENTIVE", (120, 210, 100)

    # ------------------------------------------------------------------
    def _maybe_alert(self, m: FrameMetrics) -> None:
        cfg = self.cfg
        if m.micro_sleep or m.score >= cfg.alert_score or m.yawn_duration >= 1.5:
            freq = 2400 if m.micro_sleep else (1800 if m.score >= cfg.alert_score else 1200)
            dur = 450 if m.micro_sleep else 300
            self.alerter.trigger(frequency=freq, duration_ms=dur)

    # ------------------------------------------------------------------
    def _render(
        self, enhanced_frame: np.ndarray, pts: Optional[np.ndarray], m: FrameMetrics,
        width: int, dt: float,
    ) -> np.ndarray:
        output = enhanced_frame
        self._pulse_clock += dt

        if pts is not None:
            draw_face_vector(output, pts, m.accent_color)

        draw_top_badge(output, m.status, m.accent_color, width, self._pulse_clock * 3.0)
        draw_telemetry_card(
            output,
            score=m.score,
            ml_prob=m.ml_prob,
            perclos=m.perclos,
            ear_baseline=m.ear_baseline if m.face_found else self.tracker.ear_baseline,
            pitch=m.pitch,
            night_mode=m.night_mode,
            yawn_count=m.yawn_count,
            score_history=self.score_history,
        )

        if m.micro_sleep or m.score >= self.cfg.alert_score:
            draw_critical_vignette(output, self._pulse_clock * 6.0)

        return output

    # ------------------------------------------------------------------
    def run(self) -> None:
        cfg = self.cfg
        ensure_model_downloaded(cfg.model_path, cfg.model_url)

        base_options = mp.tasks.BaseOptions
        face_landmarker = mp.tasks.vision.FaceLandmarker
        face_landmarker_options = mp.tasks.vision.FaceLandmarkerOptions
        running_mode = mp.tasks.vision.RunningMode

        options = face_landmarker_options(
            base_options=base_options(model_asset_path=cfg.model_path),
            running_mode=running_mode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45,
            min_tracking_confidence=0.45,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )

        # A webcam is opened by an integer index; a file by a path string.
        is_camera = isinstance(cfg.input_video, int)
        cap = cv2.VideoCapture(cfg.input_video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open input source: {cfg.input_video}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if is_camera and fps <= 1.0:
            # Many webcams misreport FPS (or report 0) until frames start
            # flowing; fall back to a sane default rather than pacing at 1fps.
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = None if is_camera else (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None)

        writer = None
        csv_file = None
        csv_writer = None
        if cfg.save_output:
            os.makedirs(os.path.dirname(cfg.output_video) or ".", exist_ok=True)
            os.makedirs(os.path.dirname(cfg.output_csv) or ".", exist_ok=True)

            writer = cv2.VideoWriter(
                cfg.output_video, cv2.VideoWriter_fourcc(*cfg.codec), fps, (width, height)
            )
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Cannot open video writer for: {cfg.output_video}")

            csv_file = open(cfg.output_csv, "w", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "time", "brightness", "night", "face", "ml_drowsiness_prob",
                "left_ear", "right_ear", "ear_baseline", "perclos", "pitch",
                "pitch_baseline", "pitch_speed", "eye_closed_time", "yawn_time",
                "yawn_count_2m", "head_down_time", "micro_sleep", "score", "status",
            ])

        reader = ThreadedFrameReader(cap).start()
        progress = None
        if _HAS_TQDM and not cfg.show_preview:
            # A progress bar and a live preview window fight over stdout /
            # the event loop's attention; skip it when we're actually
            # showing a live window.
            progress = tqdm(total=total_frames, unit="frame", desc="Processing")

        # Open the preview window immediately (before the first frame is
        # even processed) so the person sees a window pop up right away,
        # rather than waiting for processing to finish.
        window_name = "Tesla Autopilot Driver Monitoring (FSD UI)"
        if cfg.show_preview:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, width, height)

        target_frame_dt = 1.0 / fps
        prev_wall_time = time.time()

        try:
            with face_landmarker.create_from_options(options) as landmarker:
                while True:
                    frame_start = time.time()
                    frame = reader.read()
                    if frame is None:
                        break

                    self.frame_index += 1
                    timestamp_ms = int(self.frame_index * 1000 / fps)
                    video_time = timestamp_ms / 1000.0

                    now = time.time()
                    dt = max(1e-3, now - prev_wall_time)
                    prev_wall_time = now

                    enhanced_frame, brightness, night_mode = enhance_low_light(frame)
                    detect_frame = self._detection_frame(enhanced_frame, width, height)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB),
                    )
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)

                    pts: Optional[np.ndarray] = None
                    if result.face_landmarks:
                        m = self._process_face(result, enhanced_frame, width, height, video_time)
                        pts = landmarks_to_pixels(result.face_landmarks[0], width, height)
                    else:
                        self.closed_start = None
                        self.yawn_start = None
                        self.head_down_start = None
                        self.perclos_history.clear()
                        m = FrameMetrics(face_found=False)
                        m.yawn_count = self._update_yawn_history(video_time)

                    m.brightness = brightness
                    m.night_mode = night_mode
                    self._classify_status(m)
                    self._maybe_alert(m)
                    self.score_history.append(float(m.score))

                    output = self._render(enhanced_frame, pts, m, width, dt)

                    if csv_writer is not None:
                        csv_writer.writerow([
                            video_time, brightness, night_mode, m.face_found, m.ml_prob,
                            m.left_ear, m.right_ear, m.ear_baseline or self.tracker.ear_baseline,
                            m.perclos, m.pitch, self.tracker.head_pitch_baseline,
                            m.head_drop_speed, m.eye_closed_duration, m.yawn_duration,
                            m.yawn_count, m.head_down_duration, m.micro_sleep, m.score,
                            m.status,
                        ])

                    if writer is not None:
                        writer.write(output)

                    if cfg.show_preview:
                        # Real-time pacing: for a video *file* we want the
                        # preview to play back at the source's own fps
                        # instead of racing ahead as fast as the CPU can
                        # process frames. A live camera is already paced by
                        # the hardware, so no extra sleep is needed there.
                        if cfg.realtime_playback and not is_camera:
                            elapsed = time.time() - frame_start
                            remaining = target_frame_dt - elapsed
                            wait_ms = max(1, int(remaining * 1000))
                        else:
                            wait_ms = 1

                        cv2.imshow(window_name, output)
                        key = cv2.waitKey(wait_ms) & 0xFF
                        if key == ord("q") or key == 27:  # 'q' or ESC
                            LOGGER.info("Stopped by user.")
                            break

                    if progress is not None:
                        progress.update(1)
                    elif not cfg.show_preview and self.frame_index % 100 == 0:
                        LOGGER.info("Processed %d frames...", self.frame_index)

        finally:
            reader.stop()
            if progress is not None:
                progress.close()
            cap.release()
            if writer is not None:
                writer.release()
            if csv_file is not None:
                csv_file.close()
            if cfg.show_preview:
                cv2.destroyAllWindows()

        if cfg.save_output:
            LOGGER.info("Done. Output video: %s | CSV log: %s", cfg.output_video, cfg.output_csv)
        else:
            LOGGER.info("Done. (live preview only, nothing was saved)")


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv: Optional[List[str]] = None) -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Driver drowsiness monitoring pipeline")
    parser.add_argument(
        "--input", dest="input_video", default=cfg.input_video,
        help="Path to a video file, OR a webcam index such as 0, 1, ...",
    )
    parser.add_argument("--output", dest="output_video", default=cfg.output_video)
    parser.add_argument("--csv", dest="output_csv", default=cfg.output_csv)
    parser.add_argument("--model", dest="model_path", default=cfg.model_path)
    parser.add_argument("--tflite-model", dest="tflite_model_path", default=cfg.tflite_model_path)
    parser.add_argument("--detection-max-width", type=int, default=cfg.detection_max_width)
    parser.add_argument(
        "--show", dest="show_preview", action="store_true",
        help="Open a live preview window while processing (real-time view).",
    )
    parser.add_argument(
        "--live", dest="live", action="store_true",
        help="Shortcut for --show plus --no-save: just watch the monitor "
             "live (webcam or file) without writing an output video/CSV.",
    )
    parser.add_argument(
        "--no-save", dest="save_output", action="store_false",
        help="Do not write the annotated video / CSV log at all.",
    )
    parser.add_argument(
        "--no-realtime-pace", dest="realtime_playback", action="store_false",
        help="When previewing a video file, run as fast as possible instead "
             "of pacing playback to the source's own frame rate.",
    )
    parser.add_argument("--codec", default=cfg.codec)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # "0", "1", ... on the CLI means "open webcam index N", not a filename.
    input_video = args.input_video
    if isinstance(input_video, str) and input_video.isdigit():
        input_video = int(input_video)
    cfg.input_video = input_video

    cfg.output_video = args.output_video
    cfg.output_csv = args.output_csv
    cfg.model_path = args.model_path
    cfg.tflite_model_path = args.tflite_model_path
    cfg.detection_max_width = args.detection_max_width
    cfg.codec = args.codec
    cfg.realtime_playback = args.realtime_playback
    cfg.save_output = args.save_output

    if args.live:
        cfg.show_preview = True
        cfg.save_output = False
    else:
        cfg.show_preview = args.show_preview

    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    try:
        DrowsinessMonitor(cfg).run()
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user.")
        return 130
    except Exception as exc:  # pragma: no cover - top-level safety net
        LOGGER.error("Fatal error: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())