from .pose_extraction import PoseExtractor
from typing import List
import mediapipe as mp
import numpy as np
from util.frame_data import Frame, Frames
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
        self.standard = self.kp_load(standard_source)

    def match(self, destination: str):
        # TODO: this function is to be extended in the future development. For now is just calculating the L2 loss
        '''
        compare the poses in keyframes of standard video with learner's video
        calculate the loss
        '''
        self.learner = self.kp_load(destination)
        losses = []
        for keyframe in self.standard:
            loss = float('inf')
            for frame in self.learner:
                loss = min(loss, self.single_match(keyframe, frame))
            losses.append(loss)
        for idx in range(len(losses)):
            print(f'the {idx+1}th keyfram has loss {losses[idx]}')

    def single_match(self, source, dest) -> float:
        '''
        match the exact frame of coach pose and learner pose
        return: Loss between two frames
        '''
        loss = 0
        for src, dst in zip(source, dest):
            l2 = np.sum(np.power((src - dst), 2))
            l2_norm = np.sqrt(l2)
            loss += l2_norm
        return loss

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

def main():
    src = '../data/standard/'
    dst = '../data/practice_case/practice_1.mp4'

    matcher = Matching(src)
    matcher.match(dst)

if __name__ == '__main__':
    main()