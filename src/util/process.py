import argparse
import os
import cv2

def extract_keyframe(video_path, save_path, frame_interval=0.5):
    assert os.path.isfile(video_path), f"{video_path} does not exist!"
    assert os.path.isdir(save_path), f"{save_path} does not exist!"

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video file opened successfully
    if not cap.isOpened():
        raise ValueError("Error: Could not open video file")

    fps = int(cap.get(cv2.CAP_PROP_FPS))

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
        if frame_count % int(fps * frame_interval) == 0:
            # Save the extracted frame as an image
            frame_filename = f"{save_path}/frame_{frame_count}.jpg"
            cv2.imwrite(frame_filename, frame)
            print(f"Frame {frame_count} saved as {frame_filename}")

    # Release the video file
    cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract keyframes from a video.')
    parser.add_argument('--video_path', type=str, help='Path to the video file')
    parser.add_argument('--save_path', type=str, help='Directory to save extracted keyframes')
    parser.add_argument('--frame_interval', type=float, default=0.5, help='Interval between extracted frames (seconds)')

    args = parser.parse_args()

    extract_keyframe(args.video_path, args.save_path, args.frame_interval)
