from ..detectors.mp_detector import MediaPipeDetector
from .base_matcher import BaseMatcher
import mediapipe as mp
import numpy as np
from util.frame_data import Frame, Frames
mp_pose = mp.solutions.pose


class MediaPipeMatcher(BaseMatcher):
    def __init__(self, standard_source: str, display: bool = False) -> None:
        self.keypoints = [
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
        self.pose_extractor = MediaPipeDetector(display=display)
        self.standard = self.kp_load(standard_source)

    def kp_load(self, source: str) -> Frames:
        '''
        get all the keypoints from a source
        '''
        # TODO: the Kalman Filter could be added here
        self.pose_extractor.set_source(source)
        kp_list = []
        for frame in self.pose_extractor.extract():
            kp_frame_list = []
            for kp in self.keypoints:
                kp_frame_list.append(np.array([frame.pose_world_landmarks.landmark[kp].x,
                                               frame.pose_world_landmarks.landmark[kp].y,
                                               frame.pose_world_landmarks.landmark[kp].z]))
            kp_list.append(Frame(kp_frame_list))
        return Frames(kp_list)
