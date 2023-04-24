import cv2
import mediapipe as mp
import numpy as np
import os
from util.frame_data import Frames
from .base_detector import BaseDetector
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class MediaPipeDetector(BaseDetector):
    '''
    An extractor to read from a source and get all the pose estimations
    using MediaPipe as Detector
    '''
    def video_extract(self) -> Frames:
        # For webcam input:
        # TODO: check the validation of the path
        cap = cv2.VideoCapture(self.source)
        frames = []
        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                frames.append(results)

                if self.display:
                    # Draw the pose annotation on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    # Flip the image horizontally for a selfie-view display.
                    im = cv2.flip(image, 1)
                    im = cv2.resize(im, (540, 960))
                    cv2.imshow('MediaPipe Pose', im)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
        cap.release()
        return frames

    def image_extract(self) -> Frames:
        # For static images:
        # TODO: check the validation of the image file paths
        # TODO: extend the code to more types than jpg
        IMAGE_FILES = []

        # Get all files and folders within the specified folder
        all_items = os.listdir(self.source)

        # Filter only the JPEG files
        IMAGE_FILES.extend(
            [self.source + file for file in all_items if file.lower().endswith(".jpg")])
        
        keyframes = []

        BG_COLOR = (192, 192, 192)  # gray
        with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5) as pose:
            for idx, file in enumerate(IMAGE_FILES):
                image = cv2.imread(file)
                # Convert the BGR image to RGB before processing.
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if not results.pose_landmarks:
                    continue

                keyframes.append(results)

                if self.display:
                    annotated_image = image.copy()
                    # Draw segmentation on the image.
                    # To improve segmentation around boundaries, consider applying a joint
                    # bilateral filter to "results.segmentation_mask" with "image".
                    condition = np.stack(
                        (results.segmentation_mask,) * 3, axis=-1) > 0.1
                    bg_image = np.zeros(image.shape, dtype=np.uint8)
                    bg_image[:] = BG_COLOR
                    annotated_image = np.where(
                        condition, annotated_image, bg_image)
                    # Draw pose landmarks on the image.
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    cv2.imwrite(self.source + '../results/' +
                                str(idx) + '.png', annotated_image)
                    # Plot pose world landmarks.
                    mp_drawing.plot_landmarks(
                        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        return keyframes
