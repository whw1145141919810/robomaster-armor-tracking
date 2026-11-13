"""Collect chessboard views and save OpenCV camera intrinsics and distortion."""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from calibration import (
    MIN_VIEWS,
    RECOMMENDED_VIEWS,
    calibrate_from_views,
    find_chessboard,
    save_calibration,
)


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class ChessboardViews:
    def __init__(self, pattern_size: tuple[int, int]):
        self.pattern_size = pattern_size
        self.image_size: tuple[int, int] | None = None
        self.points: list[np.ndarray] = []
        self.labels: list[str] = []

    def add(self, frame: np.ndarray, corners: np.ndarray, label: str) -> bool:
        height, width = frame.shape[:2]
        size = (width, height)
        if self.image_size is None:
            self.image_size = size
        elif self.image_size != size:
            expected_width, expected_height = self.image_size
            raise ValueError(
                f"{label} is {width}x{height}, expected {expected_width}x{expected_height}. "
                "Use images captured at a single resolution."
            )

        # A 180-degree ordering reversal can describe the same symmetric board view.
        tolerance_px = 0.003 * math.hypot(width, height)
        for previous in self.points:
            direct = np.linalg.norm(corners - previous, axis=2)
            reversed_order = np.linalg.norm(corners - previous[::-1], axis=2)
            if min(direct.mean(), reversed_order.mean()) < tolerance_px:
                print(f"Skipped {label}: move or tilt the board further.")
                return False

        self.points.append(corners)
        self.labels.append(label)
        print(f"Accepted view {len(self.points)}: {label}")
        return True


def collect_images(paths: list[Path], views: ChessboardViews) -> None:
    image_paths = []
    for path in paths:
        if path.is_dir():
            image_paths.extend(
                item for item in sorted(path.iterdir())
                if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.is_file():
            image_paths.append(path)
        else:
            raise ValueError(f"Image path does not exist: {path}")
    if not image_paths:
        raise ValueError("No calibration images found.")

    for path in dict.fromkeys(item.resolve() for item in image_paths):
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        frame = cv2.imdecode(encoded, flags) if encoded.size else None
        if frame is None:
            print(f"Skipped {path}: could not decode image.")
            continue
        corners = find_chessboard(frame, views.pattern_size)
        if corners is None:
            print(f"Skipped {path}: no complete chessboard found.")
            continue
        views.add(frame, corners, str(path))


def collect_camera(args: argparse.Namespace, views: ChessboardViews) -> bool:
    from capture import open_camera

    backend = cv2.currentUIFramework()
    if not backend:
        raise RuntimeError("Live preview requires opencv-python with GUI support. Use --images for offline calibration.")
    is_mvs = args.mvs is not None
    camera = args.mvs if is_mvs else (args.camera or 0)
    window = "Chessboard calibration"
    if args.save_images is not None:
        args.save_images.mkdir(parents=True, exist_ok=True)
        if any(args.save_images.glob("view_*.png")):
            raise ValueError("Choose an image directory without previously saved views.")
    capture = open_camera(camera, width=args.width, height=args.height, mvs=is_mvs)
    window_open = False
    print("Space: accept view | Enter: calibrate | Esc or Q: quit")
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        window_open = True
        cv2.resizeWindow(window, 1000, 700)
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Unable to read a camera frame.")
            corners = find_chessboard(frame, views.pattern_size, live=True)
            preview = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(preview, views.pattern_size, corners, True)
            status = "board found" if corners is not None else "find full board"
            cv2.rectangle(preview, (0, 0), (preview.shape[1], 72), (15, 15, 15), -1)
            for y, line in (
                (26, f"Views: {len(views.points)} (aim for {RECOMMENDED_VIEWS}+) | {status}"),
                (55, "Space: capture   Enter: solve   Esc/Q: quit"),
            ):
                cv2.putText(preview, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return False
            if backend != "WAYLAND" and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return False
            if key == ord(" "):
                if corners is None:
                    print("Place the complete board in the image before capturing.")
                else:
                    filename = f"view_{len(views.points) + 1:03d}.png"
                    if args.save_images is None:
                        views.add(frame, corners, filename)
                    else:
                        path = args.save_images / filename
                        if views.add(frame, corners, str(path)):
                            encoded_ok, encoded = cv2.imencode(".png", frame)
                            if not encoded_ok:
                                raise RuntimeError(f"Could not encode {path}.")
                            with path.open("xb") as handle:
                                handle.write(encoded.tobytes())
            if key in (10, 13):
                if len(views.points) >= MIN_VIEWS:
                    return True
                print(f"Capture at least {MIN_VIEWS} distinct views first.")
    finally:
        capture.release()
        if window_open:
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, help="OpenCV camera index (default: 0)")
    source.add_argument("--mvs", type=int, help="Hikrobot MVS camera index")
    source.add_argument("--images", type=Path, nargs="+", help="input images or directories")
    parser.add_argument("--width", type=int, help="requested OpenCV capture width")
    parser.add_argument("--height", type=int, help="requested OpenCV capture height")
    parser.add_argument("--cols", type=int, default=6, help="inner corner columns (default: 6)")
    parser.add_argument("--rows", type=int, default=9, help="inner corner rows (default: 9)")
    parser.add_argument("--square-mm", type=float, required=True, help="measured square side in mm")
    parser.add_argument("--save-images", type=Path, help="directory for accepted live frames")
    parser.add_argument("--output", type=Path, default=Path("calibration.json"), help="output JSON path")
    args = parser.parse_args()
    if args.cols < 3 or args.rows < 3:
        parser.error("--cols and --rows must each be at least 3")
    if not math.isfinite(args.square_mm) or args.square_mm <= 0:
        parser.error("--square-mm must be a positive finite number")
    if any(value is not None and value < 0 for value in (args.camera, args.mvs)):
        parser.error("camera index must be nonnegative")
    if any(value is not None and value <= 0 for value in (args.width, args.height)):
        parser.error("--width and --height must be positive")
    if args.mvs is not None and (args.width is not None or args.height is not None):
        parser.error("Set the resolution for --mvs in MVS before opening the camera")
    if args.images is not None and any(
        value is not None for value in (args.width, args.height, args.save_images)
    ):
        parser.error("--width, --height, and --save-images are live capture options")
    return args


def main() -> int:
    args = parse_args()
    views = ChessboardViews((args.cols, args.rows))
    try:
        if args.images is not None:
            collect_images(args.images, views)
        elif not collect_camera(args, views):
            return 0

        if views.image_size is None:
            raise ValueError("No complete chessboard views collected.")
        calibration, rms, view_rms = calibrate_from_views(
            views.points, views.image_size, views.pattern_size, args.square_mm
        )
        if len(views.points) < RECOMMENDED_VIEWS:
            print(f"For a more reliable result, collect at least {RECOMMENDED_VIEWS} varied views.")
        errors = dict(zip(views.labels, view_rms))
        for label, error in errors.items():
            print(f"{label}: {error:.3f} px RMS")
        save_calibration(args.output, calibration, views.pattern_size, args.square_mm, rms, errors)
        width, height = calibration.image_size
        print(f"Saved {args.output} ({width}x{height}, {len(views.points)} views, {rms:.3f} px RMS)")
        return 0
    except (cv2.error, OSError, RuntimeError, ValueError) as error:
        print(f"Calibration failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
