"""Run classroom armor detection, pose estimation, and position tracking."""

import argparse
from contextlib import ExitStack
import csv
import math
from pathlib import Path
import time

import cv2
import numpy as np

from armor_detector import OnnxArmorDetector
from calibration import load_calibration
from capture import open_camera
from tracking import PositionTracker, armor_points, solve_pose


CSV_HEADER = [
    "frame", "time_s", "track_id", "status",
    "raw_x_m", "raw_y_m", "raw_z_m", "raw_distance_m",
    "filtered_x_m", "filtered_y_m", "filtered_z_m", "filtered_distance_m",
    "velocity_x_mps", "velocity_y_mps", "velocity_z_mps",
    "predicted_x_m", "predicted_y_m", "predicted_z_m", "prediction_horizon_s",
    "reprojection_rmse_px", "rvec_x_rad", "rvec_y_rad", "rvec_z_rad",
]


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("请输入有限的正数。")
    return number


def arguments():
    parser = argparse.ArgumentParser(description="装甲板关键点检测、PnP 位姿解算与卡尔曼滤波")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, metavar="INDEX", help="普通 OpenCV 相机（默认 0）")
    source.add_argument("--mvs", type=int, metavar="INDEX", help="海康 MVS 相机编号")
    source.add_argument("--video", type=Path, help="同一台已标定相机录制的原始视频")
    parser.add_argument("--model", type=Path, required=True, help="课堂 armor.onnx 模型")
    parser.add_argument("--calibration", type=Path, default=Path("calibration.json"))
    parser.add_argument("--armor-width-mm", type=positive, required=True)
    parser.add_argument("--armor-height-mm", type=positive, required=True)
    parser.add_argument("--confidence", type=positive, default=0.25)
    parser.add_argument("--input-scale", type=positive, default=0.5)
    parser.add_argument("--digital-gain", type=positive, default=1.0)
    parser.add_argument("--max-reprojection-px", type=positive, default=4.0)
    parser.add_argument("--measurement-std-m", type=positive, nargs=3, default=(0.015, 0.015, 0.05),
                        metavar=("SX", "SY", "SZ"))
    parser.add_argument("--acceleration-std-mps2", type=positive, default=3.0)
    parser.add_argument("--prediction-s", type=positive, default=0.15)
    parser.add_argument("--max-gap-s", type=positive, default=0.35)
    parser.add_argument("--match-px", type=positive, default=120.0, help="同类别目标的最大帧间匹配距离")
    parser.add_argument("--csv", type=Path, help="逐帧数据输出路径")
    parser.add_argument("--record", type=Path, help="标注视频输出路径（.mp4）")
    parser.add_argument("--record-fps", type=positive, help="指定输出视频帧率")
    parser.add_argument("--snapshot", type=Path, help="退出时保存最后一帧成功解算的画面")
    parser.add_argument("--headless", action="store_true", help="关闭实时窗口，仍可输出日志、视频和截图")
    parser.add_argument("--duration-s", type=positive, help="运行时长上限，从第一帧开始计时")
    return parser.parse_args()


def project_position(position, calibration):
    if position[2] <= 0:
        return None
    zero = np.zeros(3)
    pixels, _ = cv2.projectPoints(position.reshape(1, 3), zero, zero,
                                  calibration.camera_matrix, calibration.dist_coeffs)
    return pixels.reshape(2)


def choose_target(detections, tracker, time_s, last_detection, calibration, match_px):
    if not detections:
        return None
    if not tracker.active_at(time_s):
        return max(detections, key=lambda item: item[0].confidence)
    state = tracker.filter.statePost.ravel()
    guess = project_position(state[:3] + (time_s - tracker.last_time) * state[3:], calibration)
    if guess is None or not np.isfinite(guess).all():
        guess = last_detection.center
    matching = [item for item in detections if item[0].label == last_detection.label]
    if not matching:
        return None
    nearest = min(matching, key=lambda item: np.linalg.norm(item[0].center - guess))
    if np.linalg.norm(nearest[0].center - guess) > match_px:
        return None
    return nearest


def value_line(label, vector, unit):
    return f"{label}  X {vector[0]:+.3f}  Y {vector[1]:+.3f}  Z {vector[2]:+.3f} {unit}"


def draw_frame(frame, detections, selected, pose, estimate, calibration, horizon):
    raw_color, filtered_color, future_color = (75, 220, 255), (100, 240, 130), (255, 160, 110)
    for detection in detections:
        corners = np.rint(detection.corners).astype(np.int32)
        color = raw_color if detection is selected else (160, 160, 160)
        cv2.polylines(frame, [corners], True, color, 2)
        x, y = corners[0]
        cv2.putText(frame, f"{detection.label} {detection.confidence:.2f}",
                    (int(x), max(17, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if detection is selected:
            for index, point in enumerate(corners, 1):
                cv2.circle(frame, tuple(point), 4, color, -1)
                cv2.putText(frame, str(index), tuple(point + (5, -4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if pose is not None:
        cv2.drawFrameAxes(frame, calibration.camera_matrix, calibration.dist_coeffs,
                          pose.rotation, pose.position, 0.04, 2)

    lines = [
        ("Target lost" if estimate is None else f"Track {estimate.track_id} | {estimate.status.upper()}",
         (240, 240, 240)),
        (value_line("Raw", pose.position, "m") if pose else "Raw   --", raw_color),
    ]
    if pose is not None:
        lines.append((f"Raw range {np.linalg.norm(pose.position):.3f} m | PnP RMSE {pose.reprojection_rmse:.2f} px",
                      raw_color))
        lines.append((value_line("Rodrigues", pose.rotation, "rad"), raw_color))
    if estimate is not None:
        lines.extend([
            (value_line("Filtered", estimate.position, "m"), filtered_color),
            (f"Filtered range {np.linalg.norm(estimate.position):.3f} m", filtered_color),
            (value_line("Velocity", estimate.velocity, "m/s"), filtered_color),
            (value_line(f"Future +{horizon:.2f}s", estimate.future, "m"), future_color),
        ])
        height, width = frame.shape[:2]
        for position, color, marker in [(estimate.position, filtered_color, cv2.MARKER_CROSS),
                                        (estimate.future, future_color, cv2.MARKER_DIAMOND)]:
            point = project_position(position, calibration)
            if point is not None and np.isfinite(point).all() and np.all(point >= 0) and np.all(point < (width, height)):
                cv2.drawMarker(frame, tuple(np.rint(point).astype(int)), color, marker, 18, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(0.6, frame.shape[1] / 850)
    line_height = max(13, round(36 * scale))
    panel_width = max(cv2.getTextSize(line, font, scale, 1)[0][0] for line, _ in lines) + 20
    panel = frame[:(len(lines) + 1) * line_height, :panel_width]
    cv2.addWeighted(panel, 0.28, np.zeros_like(panel), 0.72, 0, panel)
    for index, (line, color) in enumerate(lines, 1):
        cv2.putText(frame, line, (9, index * line_height), font, scale, color, 1, cv2.LINE_AA)


def output_row(number, elapsed, pose, estimate, horizon):
    raw = [*pose.position, np.linalg.norm(pose.position)] if pose else [None] * 4
    filtered = ([*estimate.position, np.linalg.norm(estimate.position), *estimate.velocity, *estimate.future]
                if estimate else [None] * 10)
    rotation = [pose.reprojection_rmse, *pose.rotation] if pose else [None] * 4
    return [number, elapsed, estimate.track_id if estimate else None, estimate.status if estimate else "lost",
            *raw, *filtered, horizon, *rotation]


def run(args, stack):
    calibration = load_calibration(args.calibration)
    object_points = armor_points(args.armor_width_mm / 1000, args.armor_height_mm / 1000)
    if not args.headless and not cv2.currentUIFramework():
        raise RuntimeError("当前 OpenCV 没有图形界面支持，请安装 opencv-python 或加 --headless。")
    if args.record and args.record.suffix.lower() != ".mp4":
        raise ValueError("--record 请使用 .mp4 后缀。")
    if args.snapshot and args.snapshot.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        raise ValueError("--snapshot 请使用 .png 或 .jpg 后缀。")
    paths = [p.resolve() for p in (args.video, args.model, args.calibration, args.csv,
                                   args.record, args.snapshot) if p is not None]
    if len(set(paths)) != len(paths):
        raise ValueError("输入和输出必须使用不同文件路径。")

    if args.video is not None:
        capture = cv2.VideoCapture(str(args.video))
        stack.callback(capture.release)
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频 {args.video}。")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("无法读取输入视频帧率。")
    elif args.mvs is not None:
        capture = open_camera(args.mvs, mvs=True)
        stack.callback(capture.release)
        fps = None
    else:
        width, height = calibration.image_size
        capture = open_camera(args.camera if args.camera is not None else 0, width=width, height=height)
        stack.callback(capture.release)
        fps = capture.get(cv2.CAP_PROP_FPS)

    detector = OnnxArmorDetector(args.model, confidence_threshold=args.confidence,
                                 input_scale=args.input_scale, digital_gain=args.digital_gain)
    tracker = PositionTracker(args.prediction_s, args.max_gap_s, np.array(args.measurement_std_m),
                              args.acceleration_std_mps2)
    writer = log = log_file = last_measured_frame = last_detection = start = None
    pad_right = calibration.image_size[0] % 2
    pad_bottom = calibration.image_size[1] % 2
    number = measured = 0
    last_print = -math.inf
    window = "RoboMaster Armor Tracking"
    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, min(calibration.image_size[0], 1280),
                         round(min(calibration.image_size[0], 1280) * calibration.image_size[1] / calibration.image_size[0]))
        stack.callback(cv2.destroyAllWindows)

    try:
        while True:
            ok, frame = capture.read()
            captured_at = time.monotonic()
            if not ok:
                if args.video is None or number == 0:
                    raise RuntimeError("未收到相机或视频画面。")
                break
            calibration.require_image_size(frame.shape[1], frame.shape[0])
            if start is None:
                start = captured_at
                if args.csv:
                    args.csv.parent.mkdir(parents=True, exist_ok=True)
                    log_file = stack.enter_context(args.csv.open("w", encoding="utf-8", newline=""))
                    log = csv.writer(log_file)
                    log.writerow(CSV_HEADER)
                if args.record:
                    args.record.parent.mkdir(parents=True, exist_ok=True)
                    record_fps = args.record_fps or (fps if fps and math.isfinite(fps) and fps > 0 else 20.0)
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*"mp4v"),
                                             record_fps, (width + pad_right, height + pad_bottom))
                    stack.callback(writer.release)
                    if not writer.isOpened():
                        raise RuntimeError(f"无法创建录像 {args.record}。")
            elapsed = number / fps if args.video is not None else captured_at - start
            if args.duration_s and elapsed >= args.duration_s:
                break

            detections = detector.detect(frame)
            valid = []
            for detection in detections:
                pose = solve_pose(object_points, detection.corners, calibration, args.max_reprojection_px)
                if pose is not None:
                    valid.append((detection, pose))
            match = choose_target(valid, tracker, elapsed, last_detection, calibration, args.match_px)
            detection, pose = match if match else (None, None)
            estimate = tracker.update(elapsed, pose.position if pose else None)
            if detection is not None:
                last_detection = detection
                measured += 1
            if log is not None:
                log.writerow(output_row(number, elapsed, pose, estimate, args.prediction_s))
            if elapsed - last_print >= 0.5:
                message = f"{elapsed:7.2f}s {estimate.status if estimate else 'lost'}"
                if pose:
                    message += f" | {value_line('Raw', pose.position, 'm')} | range {np.linalg.norm(pose.position):.3f} m"
                if estimate:
                    message += f" | {value_line('Filtered', estimate.position, 'm')} | {value_line('Future', estimate.future, 'm')}"
                print(message, flush=True)
                if log_file is not None:
                    log_file.flush()
                last_print = elapsed

            if not args.headless or writer is not None or args.snapshot:
                draw_frame(frame, detections, detection, pose, estimate, calibration, args.prediction_s)
            if args.snapshot and pose is not None:
                last_measured_frame = frame
            if writer is not None:
                if pad_right or pad_bottom:
                    writer.write(cv2.copyMakeBorder(frame, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT))
                else:
                    writer.write(frame)
            number += 1
            if not args.headless:
                cv2.imshow(window, frame)
                delay = max(1, round(1000 * (1 / fps - (time.monotonic() - captured_at)))) if args.video is not None else 1
                if cv2.waitKey(delay) & 0xff in (27, ord("q"), ord("Q")):
                    break
                if cv2.currentUIFramework() != "WAYLAND" and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        pass

    if args.snapshot:
        if last_measured_frame is None:
            print("没有成功解算的帧，未保存截图。")
        else:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode(args.snapshot.suffix, last_measured_frame)
            if not ok:
                raise RuntimeError(f"无法保存截图 {args.snapshot}。")
            args.snapshot.write_bytes(encoded.tobytes())
    print(f"处理 {number} 帧，成功解算 {measured} 帧。")


def main():
    try:
        with ExitStack() as stack:
            run(arguments(), stack)
    except (ValueError, OSError, RuntimeError, cv2.error) as exc:
        raise SystemExit(f"运行失败：{exc}") from None


if __name__ == "__main__":
    main()
