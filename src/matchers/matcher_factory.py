from .mp_matcher import MediaPipeMatcher
from .yolo_matcher import YOLOMatcher

matcher_factory = {
    'mediapipe': MediaPipeMatcher,
    'yolo': YOLOMatcher,
}