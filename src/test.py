import util.process as process
from util.frame_data import Frame, Frames
import numpy as np
import mediapipe as mp

if __name__ == "__main__":
    process.extract_keyframe(video_path="../data/standard/standard.mp4",save_path="../data/standard", frame_interval=0.5)

    frame1 = Frame([
        [1, 2, 3],
        [4, 5, 6]
    ])

    frame2 = Frame([
        [1, 2, 3],
        [4, 5, 6]
    ])

    frames = Frames(frame_list=[frame1, frame2])

    print(frames[0])
    print(frames[0].get_point_by_position("NOSE"))
    print(frames[0][0])
