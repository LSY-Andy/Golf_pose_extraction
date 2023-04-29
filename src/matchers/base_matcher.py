import numpy as np
from util.frame_data import Frame, Frames


class BaseMatcher():
    def __init__(self, standard_source: str, display: bool = False) -> None:
        self.keypoints
        self.pose_extractor
        self.standard = self.kp_load(standard_source)

    def match(self, destinate: str):
        # TODO: this function is to be extended in the future development. For now is just calculating the L2 loss
        # TODO: Save the frame position corresponding to this minimum loss value
        # Issue: code don't show what is the keyframe of standard
        '''
        compare the poses in keyframes of standard video with learner's video
        calculate the loss
        '''
        self.learner = self.kp_load(destinate)
        losses = []
        matched_frames = []
        kp_extract = []

        #self.standard(251*23)
        #return the type: Frame
        for keyframe_idx in [0,1,2,3,4]:
            keyframe = self.standard[keyframe_idx]
            min_loss = float('inf')
            matched_frame_idx = 0
            for i, frame in enumerate(self.learner):
                loss = self.single_match(keyframe, frame)
                if loss < min_loss:
                    min_loss = loss
                    matched_frame_idx = i
            losses.append(min_loss)
            matched_frames.append(matched_frame_idx)
        for idx in range(len(matched_frames)):
            kp_extract = self.learner[matched_frames[idx]]
        #Store the key points of each frame corresponding to loss

        for idx in range(len(matched_frames)):
            print(matched_frames[idx])
            print(losses[idx])

        return matched_frames, losses, kp_extract

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