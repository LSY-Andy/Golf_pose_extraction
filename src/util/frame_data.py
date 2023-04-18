import cv2
import os
from typing import List
import numpy as np
import mediapipe as mp
from dataclasses import dataclass

mp_pose = mp.solutions.pose

KEYPOINTS = [
            mp_pose.PoseLandmark.NOSE,
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_PINKY,
            mp_pose.PoseLandmark.RIGHT_PINKY,
            mp_pose.PoseLandmark.LEFT_INDEX,
            mp_pose.PoseLandmark.RIGHT_INDEX,
            mp_pose.PoseLandmark.LEFT_THUMB,
            mp_pose.PoseLandmark.RIGHT_THUMB,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_ANKLE,
            mp_pose.PoseLandmark.LEFT_HEEL,
            mp_pose.PoseLandmark.RIGHT_HEEL,
            mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
        ]


@dataclass
class Frame:
    """
    Frame is a container of points
    It is implemented in np.array
    Anyway, point does not have to be a single class
    """
    point_list: np.ndarray
    
    def get_point_by_position(self, position: str) -> np.ndarray:
        assert(position in KEYPOINTS), f"Invalid position: {position}"
        return self.point_list[KEYPOINTS.index(position)]
    
    def get_point_by_idx(self, idx: int) -> np.ndarray:
        return self[idx]
    
    def __getitem__(self, idx: int) -> np.ndarray:
        if idx >= len(self.point_list):
            raise IndexError("Index out of range")
        return self.point_list[idx]
    
    def __len__(self) -> int:
        return len(self.point_list)


@dataclass
class Frames:
    """
    Frames is a container of frame
    """
    frame_list: List[Frame]

    def __getitem__(self, idx: int) -> Frame:
        if idx >= len(self.frame_list):
            raise IndexError("Index out of range")
        return self.frame_list[idx]
    
    def __len__(self) -> int:
        return len(self.frame_list)
