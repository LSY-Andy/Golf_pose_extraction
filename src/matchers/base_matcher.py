import numpy as np
from util.frame_data import Frame, Frames


class BaseMatcher():
    def __init__(self, standard_source: str, display: bool = False) -> None:
        self.keypoints
        self.pose_extractor
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
            keyframe = keyframe.to_relative_position()
            loss = float('inf')
            for frame in self.learner:
                frame = frame.to_relative_position()
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
        raise NotImplementedError