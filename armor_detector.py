"""Armor landmarks from the classroom Ultralytics Pose ONNX export."""

from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class ArmorDetection:
    """Corners in original frame pixels: top-left, top-right, bottom-right, bottom-left."""

    corners: np.ndarray
    confidence: float
    label: str

    @property
    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)


def order_corners(points: np.ndarray) -> np.ndarray | None:
    """Convert the classroom model's TL, BL, BR, TR order to PnP's order."""
    corners = np.ascontiguousarray(points[[0, 3, 2, 1]], dtype=np.float32)
    if not cv2.isContourConvex(corners) or cv2.contourArea(corners, oriented=True) < 16:
        return None
    if np.any(np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1) < 2):
        return None
    return corners


class OnnxArmorDetector:
    """Run a fixed-size four-keypoint YOLO Pose export using ONNX Runtime on CPU."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.25,
        input_scale: float = 0.5,
        digital_gain: float = 1.0,
    ) -> None:
        if not 0 < confidence_threshold < 1:
            raise ValueError("Detector confidence threshold must be between 0 and 1.")
        if not 0 < input_scale <= 1:
            raise ValueError("Model input scale must be in (0, 1].")
        if not np.isfinite(digital_gain) or digital_gain <= 0:
            raise ValueError("Digital gain must be finite and positive.")
        self.confidence_threshold = confidence_threshold
        self.input_scale = input_scale
        self.digital_gain = digital_gain

        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Armor model file does not exist: {path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        try:
            self.session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as error:
            raise RuntimeError(f"Could not load armor model {path}: {error}") from error

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Armor model must have one input and one raw YOLO Pose output.")
        model_input = inputs[0]
        if (
            model_input.type != "tensor(float)"
            or len(model_input.shape) != 4
            or model_input.shape[:2] != [1, 3]
            or not all(isinstance(dim, int) and dim > 0 for dim in model_input.shape[2:])
        ):
            raise ValueError("Armor model input must be fixed-size float32 [1, 3, H, W].")
        self.height, self.width = model_input.shape[2:]
        self.input_name = model_input.name
        self.output_name = outputs[0].name

        metadata = self.session.get_modelmeta().custom_metadata_map
        if metadata.get("task") != "pose":
            raise ValueError("Armor model must be an Ultralytics Pose ONNX export.")
        try:
            self.labels = literal_eval(metadata["names"])
            points, self.point_dimensions = literal_eval(metadata["kpt_shape"])
        except (KeyError, ValueError, TypeError, SyntaxError) as error:
            raise ValueError("Armor model is missing valid names or kpt_shape metadata.") from error
        if points != 4 or self.point_dimensions not in (2, 3):
            raise ValueError("Armor model must provide four keypoints with 2 or 3 values each.")
        if (
            not isinstance(self.labels, dict)
            or not self.labels
            or set(self.labels) != set(range(len(self.labels)))
            or not all(isinstance(label, str) for label in self.labels.values())
        ):
            raise ValueError("Armor class names must map consecutive class IDs to strings.")
        self.point_start = 4 + len(self.labels)
        self.channels = self.point_start + 4 * self.point_dimensions
        if (
            outputs[0].type != "tensor(float)"
            or len(outputs[0].shape) != 3
            or outputs[0].shape[:2] != [1, self.channels]
        ):
            raise ValueError(f"Armor model output must have shape [1, {self.channels}, N].")

    def detect(self, frame: np.ndarray) -> list[ArmorDetection]:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0 or frame.dtype != np.uint8:
            raise ValueError("Detector requires a nonempty uint8 BGR image.")
        frame_height, frame_width = frame.shape[:2]
        scale = self.input_scale * min(self.width / frame_width, self.height / frame_height)
        resized_width = max(1, round(frame_width * scale))
        resized_height = max(1, round(frame_height * scale))
        left = (self.width - resized_width) // 2
        top = (self.height - resized_height) // 2
        if self.digital_gain != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=self.digital_gain)
        resized = cv2.resize(frame, (resized_width, resized_height))
        canvas = np.full((self.height, self.width, 3), 114, dtype=np.uint8)
        canvas[top:top + resized_height, left:left + resized_width] = resized
        tensor = cv2.dnn.blobFromImage(canvas, scalefactor=1 / 255, swapRB=True)
        try:
            output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        except Exception as error:
            raise RuntimeError(f"Armor model inference failed: {error}") from error
        if output.ndim != 3 or output.shape[:2] != (1, self.channels):
            raise RuntimeError(f"Unexpected armor model output shape: {output.shape}")

        predictions = output[0].T
        class_scores = predictions[:, 4:self.point_start]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores.max(axis=1)
        candidates = np.flatnonzero(
            np.isfinite(predictions).all(axis=1)
            & (scores > self.confidence_threshold)
            & (scores <= 1)
            & (predictions[:, 2:4] > 0).all(axis=1)
        )
        candidates = candidates[np.argsort(-scores[candidates], kind="stable")][:300]
        offset = np.array([left, top], dtype=np.float32)
        image_scale = np.array(
            [resized_width / frame_width, resized_height / frame_height], dtype=np.float32
        )
        frame_bounds = np.array([frame_width - 1, frame_height - 1])
        boxes = []
        detections = []
        for index in candidates:
            points = predictions[index, self.point_start:].reshape(4, self.point_dimensions)
            pixels = (points[:, :2] - offset) / image_scale
            if np.any(pixels < 0) or np.any(pixels > frame_bounds):
                continue
            corners = order_corners(pixels)
            if corners is None:
                continue
            cx, cy, width, height = (float(value) for value in predictions[index, :4])
            boxes.append((cx - width / 2, cy - height / 2, width, height))
            detections.append(ArmorDetection(corners, float(scores[index]), self.labels[class_ids[index]]))

        if not detections:
            return []
        retained = cv2.dnn.NMSBoxes(
            boxes, [item.confidence for item in detections], self.confidence_threshold, 0.45
        )
        return [detections[index] for index in np.asarray(retained, dtype=int).ravel()]
