from pose_est.pose_extraction import PoseExtractor
from typing import List
import mediapipe as mp
import numpy as np
mp_pose = mp.solutions.pose


class Matching():
    def __init__(self, standard_source: str) -> None:
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
        self.pose_extractor = PoseExtractor()
        self.standard = self.kp_reading(standard_source)

    def match(self, destinate):
        # TODO: this function is to be extended in the future development. For now is just calculating the L2 loss
        '''
        compare the poses in keyframes of standard video with learner's video
        calculate the loss
        '''
        pass

    def single_match(self, frame) -> float:
        '''
        match the exact frame of coach pose and learner pose
        return: Loss between two frames
        '''
        pass

    def kp_load(self, source: str) -> List:
        '''
        get all the keypoints from a source
        '''
        # TODO: the Kalman Filter could be added here
        self.pose_extractor.set_name(source)
        kp_list = []
        for frame in self.pose_extractor.extract():
            kp_frame_list = []
            for kp in self.keypoints:
                kp_frame_list.append(np.array([frame.landmark[kp].x,
                                               frame.landmark[kp].y,
                                               frame.landmark[kp].z]))
            kp_list.append(kp_frame_list)
        return kp_list
