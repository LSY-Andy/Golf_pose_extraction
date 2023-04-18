import cv2
import os
import numpy as np
import mediapipe as mp

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