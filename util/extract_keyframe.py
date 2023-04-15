import cv2

# Open the video file
video_path = "../data/standard/standard.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# Set the frame extraction interval (in seconds)
frame_interval = 0.5

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
        frame_filename = f"../data/frame_{frame_count}.jpg"
        cv2.imwrite(frame_filename, frame)
        print(f"Frame {frame_count} saved as {frame_filename}")

# Release the video file
cap.release()
