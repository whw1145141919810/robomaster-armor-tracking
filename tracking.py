"""Planar pose and a single-target constant-velocity Kalman filter."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Pose:
    position: np.ndarray
    rotation: np.ndarray
    reprojection_rmse: float


def armor_points(width_m, height_m):
    """Object frame centered on the rectangle, in TL, TR, BR, BL order."""
    x, y = width_m / 2, height_m / 2
    return np.array([[-x, -y, 0], [x, -y, 0], [x, y, 0], [-x, y, 0]], dtype=np.float64)


def solve_pose(object_points, corners, calibration, max_error):
    pixels = np.ascontiguousarray(corners, dtype=np.float64).reshape(4, 2)
    if not np.isfinite(pixels).all():
        return None
    contour = pixels.astype(np.float32)
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 4:
        return None

    try:
        ok, rotation, translation = cv2.solvePnP(
            object_points, pixels, calibration.camera_matrix, calibration.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            return None
        matrix, _ = cv2.Rodrigues(rotation)
        depths = (object_points @ matrix.T + translation.ravel())[:, 2]
        if np.any(depths <= 0):
            return None
        projected, _ = cv2.projectPoints(
            object_points, rotation, translation, calibration.camera_matrix, calibration.dist_coeffs
        )
    except cv2.error:
        return None

    residuals = projected.reshape(4, 2) - pixels
    rmse = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
    if not np.isfinite(rmse) or rmse > max_error:
        return None
    return Pose(translation.ravel(), rotation.ravel(), rmse)


@dataclass
class Estimate:
    track_id: int
    status: str
    position: np.ndarray
    velocity: np.ndarray
    future: np.ndarray


class PositionTracker:
    def __init__(self, horizon_s, max_gap_s, measurement_std, acceleration_std):
        self.horizon_s = horizon_s
        self.max_gap_s = max_gap_s
        self.measurement_std = measurement_std
        self.acceleration_std = acceleration_std
        self.track_id = 0
        self.filter = None
        self.last_time = None
        self.last_measurement = None

    def active_at(self, time_s):
        return self.filter is not None and time_s - self.last_measurement <= self.max_gap_s

    def _initialize(self, position):
        self.track_id += 1
        self.filter = cv2.KalmanFilter(6, 3, 0, cv2.CV_64F)
        self.filter.measurementMatrix = np.eye(3, 6)
        self.filter.measurementNoiseCov = np.diag(self.measurement_std ** 2)
        self.filter.errorCovPost = np.diag([*(self.measurement_std ** 2), 1.0, 1.0, 1.0])
        self.filter.statePost = np.r_[position, np.zeros(3)].reshape(6, 1)

    def update(self, time_s, position):
        if not self.active_at(time_s):
            self.filter = None
            if position is None:
                return None
            self._initialize(position)
            state = self.filter.statePost
        else:
            dt = max(time_s - self.last_time, 1e-6)
            transition = np.eye(6)
            transition[:3, 3:] = np.eye(3) * dt
            self.filter.transitionMatrix = transition
            # Per-axis random acceleration integrated over this frame interval.
            acceleration = np.vstack((0.5 * dt ** 2 * np.eye(3), dt * np.eye(3)))
            self.filter.processNoiseCov = (
                acceleration @ acceleration.T * self.acceleration_std ** 2
            )
            state = self.filter.predict()
            if position is not None:
                state = self.filter.correct(position.reshape(3, 1))

        self.last_time = time_s
        if position is not None:
            self.last_measurement = time_s
        filtered = state[:3, 0].copy()
        velocity = state[3:, 0].copy()
        future = filtered + self.horizon_s * velocity
        return Estimate(self.track_id, "measured" if position is not None else "coasting",
                        filtered, velocity, future)
