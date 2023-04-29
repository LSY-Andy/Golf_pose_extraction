from .mp_detector import MediaPipeDetector
from .yolo_detector import YOLODetector

detector_factory = {
    'mediapipe': MediaPipeDetector,
    'yolo': YOLODetector,
}