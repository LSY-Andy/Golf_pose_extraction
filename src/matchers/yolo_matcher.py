from ..detectors.yolo_detector import YOLODetector
from .base_matcher import BaseMatcher
import numpy as np
from util.frame_data import Frame, Frames


class YOLOMatcher(BaseMatcher):
    # TODO: complement the matcher
    def __init__(self, standard_source: str, display: bool = False) -> None:
        pass

    def kp_load(self, source: str) -> Frames:
        pass
