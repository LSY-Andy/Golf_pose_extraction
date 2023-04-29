# import util.process as process
# from util.frame_data import Frame, Frames
from src.detectors.yolo_detector import YOLODetector
# import numpy as np
# import mediapipe as mp

def main():
    # process.extract_keyframe(video_path="../data/standard/standard.mp4",save_path="../data/standard", frame_interval=0.5)
    # frame1 = Frame(np.random.rand(23, 3))
    # frame2 = Frame(np.random.rand(23, 3))
    # frames = Frames(frame_list=[frame1, frame2])

    # print(frames[0])
    # print(frames[0].get_point_by_position("NOSE"))
    # print(frames[0][0])
    # frame1_new = frame1.to_relative_position()
    # print(frame1_new)
    detector = YOLODetector('data/standard/standard.mp4')
    detector.extract()

if __name__ == "__main__":
    main()
