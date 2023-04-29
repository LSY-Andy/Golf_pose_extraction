# using YOLOv8
import ultralytics
from .base_detector import BaseDetector
ultralytics.checks()

from ultralytics import YOLO
import cv2
import os
import torch
from typing import List


class YOLODetector(BaseDetector):
    '''
    An extractor to read from a source and get all the pose estimations
    using YOLOv8 as Detector
    '''
    def __init__(self, source: str = '', display=False) -> None:
        super().__init__(source, display)
        
    def video_extract(self) -> List[List[torch.Tensor]]:
        # For webcam input:
        # TODO: check the validation of the path
        model = YOLO('models/Yolo/yolov8n-pose.pt')
        cap = cv2.VideoCapture(self.source)
        frames = []
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
            result = model(image)
            frames.append(result)

            # TODO: display and save the result
        cap.release()
        return frames

    def image_extract(self) -> List[torch.Tensor]:
        # For static images:
        # TODO: check the validation of the image file paths
        # TODO: extend the code to more types than jpg
        IMAGE_FILES = []
        # TODO: make the pretrained path an argument
        # TODO: fine-tuning the model using our data following https://github.com/airockchip/ultralytics_yolov8/blob/main/examples/tutorial.ipynb
        model = YOLO('models/Yolo/yolov8n-pose.pt')

        # Get all files and folders within the specified folder
        all_items = os.listdir(self.source)

        # Filter only the JPEG files
        IMAGE_FILES.extend(
            [self.source + file for file in all_items if file.lower().endswith(".jpg")])
        
        keyframes = []

        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            # Convert the BGR image to RGB before processing.
            result = model(image)
            
            # TODO: encapsulation
            # Notice that result is a list, the length demostrate the number of people in this image. So there might be multi people there
            # the list has type List(ultralytics.yolo.engine.results.Results), it has following attributes:
            # keypoints, a 3d tensor with shape (1, 17, 3), which is the format of coco keypoint dataset
            # the 17 represents the number of keypoints and 3 represents x, y, and visible
            keyframes.extend(result[0].keypoints)
            
        # TODO: display and save the results
        return keyframes