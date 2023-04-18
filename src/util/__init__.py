import cv2
import os
import numpy as np
import mediapipe as mp
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


class Frame():
    def __init__(self, point_list) -> None:
        '''
        [
            [x, y, z],
            [x, y, z],
            ...,
            [x, y, z]
        ]
        '''
        self.point_list = np.array([point_list])

    
    def get_point_by_position(self, position) -> np.ndarray:
        assert(position in KEYPOINTS), f"Invalid position: {position}"
        return self.point_list[KEYPOINTS.index(position)]
    
    def get_point_by_idx(self, idx) -> np.ndarray:
        return self.__getitem__(idx)
    
    def __getitem__(self, idx) -> np.ndarray:
        if idx >= len(self.point_list):
            raise IndexError("Index out of range")
        return self.point_list[idx]
    
    def __len__(self) -> int:
        return len(self.point_list)


class Frames():
    """
    frames -> frame -> key points (x, y, z)
    """

    def __init__(self, frame_list:list) -> None:
        self.frame_list = frame_list

    def __getitem__(self, idx) -> Frame:
        if idx >= len(self.frame_list):
            raise IndexError("Index out of range")
        return self.frame_list[idx]
    
    def __len__(self) -> int:
        return len(self.frame_list)
    
    
    # TODO furtuer functions need to be added


    

def extract_keyframe(video_path,save_path,frame_interval=0.5):
    # assert
    assert(os.path.exists(video_path)), f"{video_path} does not exits!"
    assert(os.path.exists(save_path)), f"{save_path} does not exits!"

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()
    

    # Initialize frame counter
    frame_count = 0

    # Loop through the frames
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if we have reached the end of the video
        if not ret:
            break

        # Increment frame counter
        frame_count += 1

        # Extract frames at the desired interval
        if frame_count % int(cap.get(cv2.CAP_PROP_FPS) * frame_interval) == 0:
            # Save the extracted frame as an image
            frame_filename = f"{save_path}/frame_{frame_count}.jpg"
            cv2.imwrite(frame_filename, frame)
            print(f"Frame {frame_count} saved as {frame_filename}")

    # Release the video file
    cap.release()
