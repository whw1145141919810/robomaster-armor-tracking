"""Chessboard calibration and calibration file I/O."""

import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


MIN_VIEWS = 3
RECOMMENDED_VIEWS = 10


@dataclass(frozen=True)
class CameraCalibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]

    def require_image_size(self, width: int, height: int) -> None:
        if (width, height) != self.image_size:
            expected_width, expected_height = self.image_size
            raise ValueError(
                f"Frame is {width}x{height}, but calibration is "
                f"{expected_width}x{expected_height}. Use the calibrated resolution "
                "and the same camera settings, or recalibrate."
            )


def find_chessboard(
    frame: np.ndarray, pattern_size: tuple[int, int], *, live: bool = False
) -> np.ndarray | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if live:
        flags |= cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = corners.reshape(-1, 1, 2)
        return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    if not live:
        found, corners = cv2.findChessboardCornersSB(
            gray, pattern_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if found:
            return corners.reshape(-1, 1, 2)
    return None


def calibrate_from_views(
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
    pattern_size: tuple[int, int],
    square_mm: float,
) -> tuple[CameraCalibration, float, list[float]]:
    if len(image_points) < MIN_VIEWS:
        raise ValueError(f"Need at least {MIN_VIEWS} distinct chessboard views.")
    cols, rows = pattern_size
    board = np.zeros((cols * rows, 3), dtype=np.float32)
    board[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    boards = [board] * len(image_points)
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        boards, image_points, image_size, None, None
    )
    if not math.isfinite(rms):
        raise ValueError("Calibration did not converge. Capture more varied board poses.")
    calibration = _validated_calibration(matrix, distortion, image_size)
    view_rms = []
    for measured, rvec, tvec in zip(image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(board, rvec, tvec, matrix, distortion)
        residual = measured.reshape(-1, 2) - projected.reshape(-1, 2)
        view_rms.append(float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))))
    return calibration, float(rms), view_rms


def save_calibration(
    path: Path,
    calibration: CameraCalibration,
    pattern_size: tuple[int, int],
    square_mm: float,
    rms_px: float,
    view_rms_px: dict[str, float],
) -> None:
    width, height = calibration.image_size
    data = {
        "image_width": width,
        "image_height": height,
        "camera_matrix": calibration.camera_matrix.tolist(),
        "dist_coeffs": calibration.dist_coeffs.ravel().tolist(),
        "pattern_size": list(pattern_size),
        "square_mm": square_mm,
        "rms_px": rms_px,
        "view_rms_px": view_rms_px,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_calibration(path: str | Path) -> CameraCalibration:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            data = json.load(handle)
        return _validated_calibration(
            data["camera_matrix"],
            data["dist_coeffs"],
            (data["image_width"], data["image_height"]),
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid calibration file {path}: {error}") from error


def _validated_calibration(matrix, distortion, image_size) -> CameraCalibration:
    matrix = np.asarray(matrix, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1, 1)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
        or not np.allclose(matrix[2], [0, 0, 1])
    ):
        raise ValueError("Expected a finite 3x3 camera matrix with positive focal lengths.")
    if distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
        raise ValueError("Expected 4, 5, 8, 12, or 14 finite OpenCV distortion coefficients.")
    if any(type(value) is not int or value <= 0 for value in image_size):
        raise ValueError("Calibration image width and height must be positive integers.")
    return CameraCalibration(matrix, distortion, image_size)
