import argparse
import pickle
import time
from pathlib import Path

import cv2
import numpy as np

from src.gaze_core import build_face_mesh, extract_features, get_screen_size, split_feature_vector
from src.model_utils import apply_calibration_correction, fit_affine_correction, fit_ridge_model, predict_points


def build_grid_points(screen_w: int, screen_h: int, cols: int, rows: int, margin_ratio: float = 0.08):
    xs = np.linspace(margin_ratio, 1.0 - margin_ratio, cols)
    ys = np.linspace(margin_ratio, 1.0 - margin_ratio, rows)
    points = []
    for row_idx, y in enumerate(ys):
        row_xs = xs if row_idx % 2 == 0 else xs[::-1]
        for x in row_xs:
            points.append((int(round(x * screen_w)), int(round(y * screen_h))))
    return points


def filter_point_samples(samples: list, keep_ratio: float = 0.65):
    arr = np.asarray(samples, dtype=np.float32)
    if len(arr) <= 3:
        return arr
    center = np.median(arr, axis=0)
    dist = np.linalg.norm(arr - center, axis=1)
    threshold = float(np.quantile(dist, keep_ratio))
    mask = dist <= threshold
    kept = arr[mask]
    if len(kept) < 3:
        return arr
    return kept


def make_background(screen_w: int, screen_h: int):
    # Zinc 950/900 deep dark theme
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    # Subtle gradient: slightly lighter at the top-left
    x = np.linspace(0, 1, screen_w)
    y = np.linspace(0, 1, screen_h)
    xv, yv = np.meshgrid(x, y)
    
    # Base: Zinc 950 (9, 9, 11)
    # Gradient to Zinc 900 (24, 24, 27)
    canvas[:, :, 0] = (11 + 16 * (1 - yv) * (1 - xv)).astype(np.uint8) # B
    canvas[:, :, 1] = (9 + 15 * (1 - yv) * (1 - xv)).astype(np.uint8)  # G
    canvas[:, :, 2] = (9 + 15 * (1 - yv) * (1 - xv)).astype(np.uint8)  # R
    return canvas


def put_text(canvas, text: str, xy: tuple, scale: float, color: tuple, thickness: int = 2):
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_progress_bar(canvas, x: int, y: int, width: int, height: int, progress: float, fill_color: tuple):
    progress = float(np.clip(progress, 0.0, 1.0))
    # Zinc 800 background
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (39, 39, 42), -1)
    # Zinc 700 border
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (63, 63, 71), 1)
    fill_width = int(round(width * progress))
    if fill_width > 0:
        cv2.rectangle(canvas, (x, y), (x + fill_width, y + height), fill_color, -1)


def draw_metric_card(canvas, x: int, y: int, width: int, height: int, title: str, value: str, accent: tuple):
    # Zinc 900 background
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (27, 24, 24), -1)
    # Zinc 800 border
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (42, 39, 39), 1)
    # Accent line at bottom
    cv2.rectangle(canvas, (x, y + height - 3), (x + width, y + height), accent, -1)
    
    put_text(canvas, title, (x + 18, y + 28), 0.55, (161, 161, 170), 1) # Zinc 400
    put_text(canvas, value, (x + 18, y + 68), 0.95, (250, 250, 250), 2) # Zinc 50


def draw_target_map(canvas, targets: list, current_idx: int, box: tuple):
    x, y, width, height = box
    pad = 16
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (16, 20, 32), -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (72, 82, 110), 2)
    put_text(canvas, "Dot Map", (x + 16, y + 28), 0.55, (175, 190, 220), 1)

    min_x = min(pt[0] for pt in targets)
    max_x = max(pt[0] for pt in targets)
    min_y = min(pt[1] for pt in targets)
    max_y = max(pt[1] for pt in targets)
    span_x = max(max_x - min_x, 1)
    span_y = max(max_y - min_y, 1)

    for idx, (tx, ty) in enumerate(targets, start=1):
        px = x + pad + int(round((tx - min_x) * (width - 2 * pad) / span_x))
        py = y + pad + int(round((ty - min_y) * (height - 2 * pad) / span_y))
        if idx < current_idx:
            color = (92, 206, 132)
            radius = 6
        elif idx == current_idx:
            color = (0, 176, 255)
            radius = 8
        else:
            color = (98, 107, 132)
            radius = 5
        cv2.circle(canvas, (px, py), radius, color, -1)


def compute_point_quality(samples: list):
    arr = np.asarray(samples, dtype=np.float32)
    if len(arr) < 3:
        return None, None
    center = np.median(arr, axis=0)
    dist = np.linalg.norm(arr - center, axis=1)
    spread = float(np.median(dist))
    score = float(np.clip(100.0 * np.exp(-spread / 0.08), 0.0, 100.0))
    return score, spread


def accuracy_from_error(error_px: float, screen_w: int, screen_h: int):
    ref = max(80.0, float(np.hypot(screen_w, screen_h)) * 0.08)
    return float(np.clip(100.0 * (1.0 - float(error_px) / ref), 0.0, 100.0))


def format_score(score):
    return "--" if score is None else f"{float(score):.1f}%"


def mean_or_none(values):
    return None if not values else float(np.mean(values))


def render_intro_screen(base_bg: np.ndarray, total_points: int, samples_per_point: int, countdown: int = 5):
    canvas = base_bg.copy()
    screen_w = base_bg.shape[1]
    right_margin = 60
    gap = 22
    card2_x = screen_w - right_margin - 220
    card1_x = card2_x - gap - 190

    # Header area
    cv2.rectangle(canvas, (56, 54), (840, 350), (27, 24, 24), -1) # Zinc 900
    cv2.rectangle(canvas, (56, 54), (840, 350), (246, 130, 59), 2) # Blue 500

    put_text(canvas, "NeuralSight Calibration", (82, 110), 1.2, (250, 250, 250), 2)
    put_text(canvas, "Focus on the dots to calibrate your unique gaze profile.", (82, 152), 0.72, (161, 161, 170), 2)

    steps = [
        "1. Sit naturally and keep your head steady.",
        "2. Follow each blue dot until capture finishes.",
        "3. Quality and accuracy are measured in real-time."
    ]
    for i, step in enumerate(steps):
        put_text(canvas, step, (82, 212 + i * 36), 0.72, (212, 212, 216), 1) # Zinc 300

    draw_metric_card(canvas, card1_x, 70, 190, 96, "Dots", str(total_points), (246, 130, 59))
    draw_metric_card(canvas, card2_x, 70, 220, 96, "Samples / Dot", str(samples_per_point), (92, 206, 132))

    if countdown > 0:
        # Show countdown overlay
        countdown_text = f"Starting in {countdown}..."
        put_text(canvas, countdown_text, (84, base_bg.shape[0] - 74), 0.84, (250, 250, 250), 2)
        put_text(canvas, "ESC to cancel", (340, base_bg.shape[0] - 74), 0.84, (113, 113, 122), 2) # Zinc 500
    else:
        put_text(canvas, "Go!", (84, base_bg.shape[0] - 74), 0.84, (92, 206, 132), 2)
        put_text(canvas, "ESC to cancel", (340, base_bg.shape[0] - 74), 0.84, (113, 113, 122), 2)
    return canvas


def render_calibration_screen(
    base_bg: np.ndarray,
    targets: list,
    current_idx: int,
    target: tuple,
    collected: int,
    samples_per_point: int,
    phase_text: str,
    face_ready: bool,
    status_text: str,
    dot_quality,
    avg_quality,
):
    canvas = base_bg.copy()
    screen_h, screen_w = canvas.shape[:2]
    total_points = len(targets)
    progress = ((current_idx - 1) + (float(collected) / max(samples_per_point, 1))) / max(total_points, 1)
    sample_progress = float(collected) / max(samples_per_point, 1)

    # Info Box
    cv2.rectangle(canvas, (54, 52), (650, 334), (27, 24, 24), -1) # Zinc 900
    cv2.rectangle(canvas, (54, 52), (650, 334), (246, 130, 59), 2) # Blue 500
    put_text(canvas, "NeuralSight Calibration", (78, 92), 1.0, (250, 250, 250), 2)
    put_text(canvas, f"{phase_text} dot {current_idx}/{total_points}", (78, 128), 0.72, (161, 161, 170), 2)
    
    draw_progress_bar(canvas, 78, 148, 540, 18, progress, (246, 130, 59))
    put_text(canvas, "Overall progress", (78, 142), 0.46, (113, 113, 122), 1)
    draw_progress_bar(canvas, 78, 178, 540, 14, sample_progress, (92, 206, 132))
    put_text(canvas, "Current dot capture", (78, 173), 0.46, (113, 113, 122), 1)

    draw_metric_card(canvas, 78, 214, 156, 84, "Samples", f"{collected}/{samples_per_point}", (92, 206, 132))
    draw_metric_card(canvas, 254, 214, 156, 84, "Dot Quality", format_score(dot_quality), (246, 130, 59))
    draw_metric_card(canvas, 430, 214, 156, 84, "Average", format_score(avg_quality), (255, 171, 72))

    draw_target_map(canvas, targets, current_idx, (screen_w - 310, 54, 250, 170))

    # Face Status
    status_color = (92, 206, 132) if face_ready else (246, 130, 59)
    cv2.rectangle(canvas, (screen_w - 310, 248), (screen_w - 60, 332), (27, 24, 24), -1)
    cv2.rectangle(canvas, (screen_w - 310, 248), (screen_w - 60, 332), status_color, 2)
    put_text(canvas, "Face Status", (screen_w - 292, 276), 0.55, (161, 161, 170), 1)
    put_text(canvas, "Ready" if face_ready else "Waiting", (screen_w - 292, 314), 0.92, (250, 250, 250), 2)

    # Dot styling
    dot_outer = 44
    dot_mid = 24
    cv2.circle(canvas, target, dot_outer, (255, 112, 78), 2) # Outer ring
    cv2.circle(canvas, target, dot_mid, (246, 130, 59), -1)  # Inner circle
    cv2.circle(canvas, target, 7, (250, 250, 250), -1)       # Center core
    cv2.line(canvas, (target[0] - 34, target[1]), (target[0] + 34, target[1]), (242, 232, 224), 2)
    cv2.line(canvas, (target[0], target[1] - 34), (target[0], target[1] + 34), (242, 232, 224), 2)

    put_text(canvas, status_text, (80, screen_h - 62), 0.75, (230, 235, 240), 2)
    put_text(canvas, "ESC to cancel calibration", (80, screen_h - 28), 0.56, (113, 113, 122), 1)
    return canvas


def render_summary_screen(
    base_bg: np.ndarray,
    output_path: Path,
    point_metrics: list,
    average_accuracy: float,
    average_error: float,
    average_quality: float,
    feature_mode: str,
):
    canvas = base_bg.copy()
    screen_h, screen_w = canvas.shape[:2]
    cv2.rectangle(canvas, (54, 52), (screen_w - 54, screen_h - 92), (27, 24, 24), -1)
    cv2.rectangle(canvas, (54, 52), (screen_w - 54, screen_h - 92), (92, 206, 132), 2)

    put_text(canvas, "Calibration Complete", (82, 100), 1.1, (250, 250, 250), 2)
    put_text(canvas, f"Model saved: {output_path.name} ({feature_mode})", (82, 138), 0.68, (161, 161, 170), 2)

    draw_metric_card(canvas, 82, 174, 220, 96, "Average Accuracy", f"{average_accuracy:.1f}%", (92, 206, 132))
    draw_metric_card(canvas, 326, 174, 220, 96, "Average Error", f"{average_error:.0f} px", (246, 130, 59))
    draw_metric_card(canvas, 570, 174, 220, 96, "Capture Quality", f"{average_quality:.1f}%", (255, 171, 72))

    put_text(canvas, "Per-dot performance", (82, 322), 0.72, (212, 212, 216), 2)
    half = (len(point_metrics) + 1) // 2
    left_items = point_metrics[:half]
    right_items = point_metrics[half:]
    start_y = 362
    row_gap = 40

    for row_idx, metric in enumerate(left_items):
        text = f"P{metric['index']:02d}   {metric['accuracy']:.1f}%   {metric['error_px']:.0f}px"
        put_text(canvas, text, (92, start_y + row_idx * row_gap), 0.66, (212, 212, 216), 2)

    for row_idx, metric in enumerate(right_items):
        text = f"P{metric['index']:02d}   {metric['accuracy']:.1f}%   {metric['error_px']:.0f}px"
        put_text(canvas, text, (screen_w // 2 + 60, start_y + row_idx * row_gap), 0.66, (212, 212, 216), 2)

    put_text(canvas, str(output_path), (82, screen_h - 118), 0.52, (113, 113, 122), 1)
    put_text(canvas, "Press any key to finish", (82, screen_h - 74), 0.78, (250, 250, 250), 2)
    return canvas


def evaluate_point_metrics(
    point_records: list,
    W: np.ndarray,
    W_eye: np.ndarray,
    W_head: np.ndarray,
    corr_full,
    corr_eye,
    corr_head,
    feature_mode: str,
    head_assist: float,
    screen_w: int,
    screen_h: int,
):
    metrics = []
    use_hybrid = W_eye is not None and W_head is not None

    for record in point_records:
        sample_arr = np.asarray(record["samples"], dtype=np.float32)
        target = np.asarray(record["target"], dtype=np.float32)
        if use_hybrid:
            eye_samples = []
            head_samples = []
            for sample in sample_arr:
                _, eye_vec, head_vec = split_feature_vector(sample)
                eye_samples.append(eye_vec)
                head_samples.append(head_vec)
            eye_np = np.asarray(eye_samples, dtype=np.float32)
            head_np = np.asarray(head_samples, dtype=np.float32)
            pred_eye = apply_calibration_correction(predict_points(eye_np, W_eye, feature_mode=feature_mode), corr_eye)
            pred_head = apply_calibration_correction(predict_points(head_np, W_head, feature_mode=feature_mode), corr_head)
            pred = (1.0 - head_assist) * pred_eye + head_assist * pred_head
        else:
            pred = apply_calibration_correction(predict_points(sample_arr, W, feature_mode=feature_mode), corr_full)

        errors = np.linalg.norm(pred - target[None, :], axis=1)
        mean_error = float(np.mean(errors))
        metrics.append(
            {
                "index": int(record["index"]),
                "target": (int(target[0]), int(target[1])),
                "error_px": mean_error,
                "accuracy": accuracy_from_error(mean_error, screen_w, screen_h),
                "capture_quality": float(record["quality"] or 0.0),
            }
        )

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--samples-per-point", type=int, default=28)
    parser.add_argument("--grid", type=str, default="5x4", help="format: colsxrows")
    parser.add_argument("--output", type=Path, default=Path("calibration_model.pkl"))
    parser.add_argument("--ridge", type=float, default=10.0, help="regularization strength for calibration fit")
    parser.add_argument("--keep-ratio", type=float, default=0.65, help="per-point sample retention ratio after outlier filtering")
    parser.add_argument("--feature-mode", type=str, default="quadratic", choices=("linear", "quadratic"), help="feature basis used during calibration")
    parser.add_argument("--settle-time", type=float, default=0.75, help="seconds to settle on each dot before sampling")
    parser.add_argument("--margin-ratio", type=float, default=0.08, help="screen margin ratio used for the outermost calibration dots")
    parser.add_argument("--blink-threshold", type=float, default=0.205, help="minimum mean eye-open ratio required before a sample is accepted")
    args = parser.parse_args()

    if args.samples_per_point < 8:
        raise ValueError("--samples-per-point must be at least 8.")
    if not (0.03 <= args.margin_ratio <= 0.18):
        raise ValueError("--margin-ratio must be between 0.03 and 0.18.")
    if not (0.3 <= args.keep_ratio <= 0.95):
        raise ValueError("--keep-ratio must be between 0.3 and 0.95.")

    cols, rows = [int(v) for v in args.grid.lower().split("x")]
    screen_w, screen_h = get_screen_size()
    targets = build_grid_points(screen_w, screen_h, cols, rows, margin_ratio=float(args.margin_ratio))

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    face_mesh = build_face_mesh()
    base_bg = make_background(screen_w, screen_h)

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    intro = render_intro_screen(base_bg, len(targets), args.samples_per_point, countdown=5)
    cv2.imshow("Calibration", intro)
    cv2.waitKey(1)

    countdown_start = time.time()
    while True:
        elapsed = time.time() - countdown_start
        remaining = max(0, 5 - int(elapsed))
        intro = render_intro_screen(base_bg, len(targets), args.samples_per_point, countdown=remaining)
        cv2.imshow("Calibration", intro)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return
        if remaining == 0:
            break

    point_records = []
    completed_quality_scores = []

    for idx, target in enumerate(targets, start=1):
        point_samples = []

        settle_start = time.time()
        while time.time() - settle_start < float(args.settle_time):
            ok, frame = cap.read()
            if not ok:
                continue

            feat = extract_features(frame, face_mesh)
            if feat is None:
                face_ready = False
                status_text = "Center your face in the camera before capture starts."
            elif feat.mean_blink < float(args.blink_threshold):
                face_ready = False
                status_text = "Open your eyes naturally and hold steady."
            else:
                face_ready = True
                status_text = "Face locked. Keep looking at the dot."

            canvas = render_calibration_screen(
                base_bg,
                targets,
                idx,
                target,
                collected=0,
                samples_per_point=args.samples_per_point,
                phase_text="Settle on",
                face_ready=face_ready,
                status_text=status_text,
                dot_quality=None,
                avg_quality=mean_or_none(completed_quality_scores),
            )
            cv2.imshow("Calibration", canvas)
            if (cv2.waitKey(1) & 0xFF) == 27:
                cap.release()
                cv2.destroyAllWindows()
                return

        collected = 0
        while collected < args.samples_per_point:
            ok, frame = cap.read()
            if not ok:
                continue

            feat = extract_features(frame, face_mesh)
            face_ready = False
            if feat is None:
                status_text = "Face lost. Return to the camera view."
            elif feat.mean_blink < float(args.blink_threshold):
                status_text = "Blink detected. Waiting for open eyes."
            else:
                point_samples.append(feat.vector)
                collected += 1
                face_ready = True
                status_text = "Good capture. Hold your gaze on the dot."

            dot_quality, _ = compute_point_quality(point_samples)
            avg_quality = mean_or_none(completed_quality_scores + ([dot_quality] if dot_quality is not None else []))
            canvas = render_calibration_screen(
                base_bg,
                targets,
                idx,
                target,
                collected=collected,
                samples_per_point=args.samples_per_point,
                phase_text="Capturing",
                face_ready=face_ready,
                status_text=status_text,
                dot_quality=dot_quality,
                avg_quality=avg_quality,
            )
            cv2.imshow("Calibration", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return

        cleaned = filter_point_samples(point_samples, keep_ratio=float(args.keep_ratio))
        quality_score, spread = compute_point_quality(cleaned.tolist())
        completed_quality_scores.append(float(quality_score or 0.0))
        point_records.append(
            {
                "index": idx,
                "target": target,
                "samples": cleaned,
                "quality": quality_score,
                "spread": spread,
            }
        )

        flash_until = time.time() + 0.35
        while time.time() < flash_until:
            avg_quality = mean_or_none(completed_quality_scores)
            canvas = render_calibration_screen(
                base_bg,
                targets,
                idx,
                target,
                collected=args.samples_per_point,
                samples_per_point=args.samples_per_point,
                phase_text="Saved",
                face_ready=True,
                status_text=f"Dot {idx} captured with {format_score(quality_score)} quality.",
                dot_quality=quality_score,
                avg_quality=avg_quality,
            )
            cv2.imshow("Calibration", canvas)
            if (cv2.waitKey(1) & 0xFF) == 27:
                cap.release()
                cv2.destroyAllWindows()
                return

    X = []
    X_eye = []
    X_head = []
    Y = []
    for record in point_records:
        target = record["target"]
        for sample in record["samples"]:
            full_vec, eye_vec, head_vec = split_feature_vector(sample)
            X.append(full_vec)
            X_eye.append(eye_vec)
            X_head.append(head_vec)
            Y.append(target)

    X_np = np.asarray(X, dtype=np.float32)
    X_eye_np = np.asarray(X_eye, dtype=np.float32)
    X_head_np = np.asarray(X_head, dtype=np.float32)
    Y_np = np.asarray(Y, dtype=np.float32)

    if len(X_np) < 20:
        raise RuntimeError("Not enough calibration samples. Try again.")

    W = fit_ridge_model(X_np, Y_np, ridge_lambda=float(args.ridge), feature_mode=args.feature_mode)
    W_eye = fit_ridge_model(X_eye_np, Y_np, ridge_lambda=float(args.ridge), feature_mode=args.feature_mode)
    W_head = fit_ridge_model(X_head_np, Y_np, ridge_lambda=float(args.ridge) * 0.8, feature_mode=args.feature_mode)

    corr_full = fit_affine_correction(predict_points(X_np, W, feature_mode=args.feature_mode), Y_np)
    corr_eye = fit_affine_correction(predict_points(X_eye_np, W_eye, feature_mode=args.feature_mode), Y_np)
    corr_head = fit_affine_correction(predict_points(X_head_np, W_head, feature_mode=args.feature_mode), Y_np)

    head_assist_default = 0.35
    point_metrics = evaluate_point_metrics(
        point_records,
        W,
        W_eye,
        W_head,
        corr_full,
        corr_eye,
        corr_head,
        feature_mode=args.feature_mode,
        head_assist=head_assist_default,
        screen_w=screen_w,
        screen_h=screen_h,
    )
    average_accuracy = float(np.mean([item["accuracy"] for item in point_metrics]))
    average_error = float(np.mean([item["error_px"] for item in point_metrics]))
    average_quality = float(np.mean(completed_quality_scores)) if completed_quality_scores else 0.0

    payload = {
        "W": W,
        "W_eye": W_eye,
        "W_head": W_head,
        "corr_full": corr_full,
        "corr_eye": corr_eye,
        "corr_head": corr_head,
        "screen_size": (screen_w, screen_h),
        "feature_dim": int(X_np.shape[1]),
        "eye_feature_dim": int(X_eye_np.shape[1]),
        "head_feature_dim": int(X_head_np.shape[1]),
        "feature_mode": args.feature_mode,
        "head_assist_default": head_assist_default,
        "grid": (cols, rows),
        "samples_per_point": args.samples_per_point,
        "ridge": float(args.ridge),
        "keep_ratio": float(args.keep_ratio),
        "margin_ratio": float(args.margin_ratio),
        "blink_threshold": float(args.blink_threshold),
        "average_accuracy": average_accuracy,
        "average_error_px": average_error,
        "capture_quality_avg": average_quality,
        "point_metrics": point_metrics,
    }

    with args.output.open("wb") as f:
        pickle.dump(payload, f)

    summary = render_summary_screen(
        base_bg,
        args.output,
        point_metrics,
        average_accuracy=average_accuracy,
        average_error=average_error,
        average_quality=average_quality,
        feature_mode=args.feature_mode,
    )
    cv2.imshow("Calibration", summary)
    cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

