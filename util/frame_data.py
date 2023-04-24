from typing import List
import numpy as np
from dataclasses import dataclass

KEYPOINTS = [
    'NOSE',
    'LEFT_SHOULDER',
    'RIGHT_SHOULDER',
    'LEFT_ELBOW',
    'RIGHT_ELBOW',
    'LEFT_WRIST',
    'RIGHT_WRIST',
    'LEFT_PINKY',
    'RIGHT_PINKY',
    'LEFT_INDEX',
    'RIGHT_INDEX',
    'LEFT_THUMB',
    'RIGHT_THUMB',
    'LEFT_HIP',
    'RIGHT_HIP',
    'LEFT_KNEE',
    'RIGHT_KNEE',
    'LEFT_ANKLE',
    'RIGHT_ANKLE',
    'LEFT_HEEL',
    'RIGHT_HEEL',
    'LEFT_FOOT_INDEX',
    'RIGHT_FOOT_INDEX',
]


@dataclass
class Frame:
    """
    Frame is a container of points
    It is implemented in np.array
    Anyway, point does not have to be a single class
    """
    point_list: np.ndarray

    def get_point_by_position(self, position: str) -> np.ndarray:
        """
        Return the coordinates of a specific keypoint by its name.

        :param position: The name of the requested keypoint.
        :return: A numpy array containing the coordinates of the keypoint.
        :raises AssertionError: If the specified position is not a valid keypoint.
        """
        assert (position in KEYPOINTS), f"Invalid position: {position}"
        return self.point_list[KEYPOINTS.index(position)]

    def get_point_by_idx(self, idx: int) -> np.ndarray:
        """
        Return the coordinates of a specific keypoint by its index.

        :param idx: The index of the requested keypoint (0-based).
        :return: A numpy array containing the coordinates of the keypoint.
        :raises IndexError: If the index is out of range.
        """
        return self[idx]

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Get the coordinates of a specific keypoint by its index.

        :param idx: The index of the requested keypoint (0-based).
        :return: A numpy array containing the coordinates of the keypoint.
        :raises IndexError: If the index is out of range.
        """
        if idx >= len(self.point_list):
            raise IndexError("Index out of range")
        return self.point_list[idx]

    def __len__(self) -> int:
        """
        Get the number of keypoints in the Frame.

        :return: The number of keypoints in the Frame.
        """
        return len(self.point_list)

    
    def get_distance(self, pos1: str, pos2: str) -> float:
        """
        Calculate Euclidean distance between two positions.

        :param pos1: The name of the first position.
        :param pos2: The name of the second position.
        :return: The Euclidean distance between the two positions.
        """
        p1 = self.get_point_by_position(pos1)
        p2 = self.get_point_by_position(pos2)
        return np.linalg.norm(p1 - p2)
    
    def to_relative_position(self) -> 'Frame':
        """
        Move the nose to (0, 0, 0) since the nose isn't supposed to move
        when playing the golf

        :return: A new frame whose nose position is (0,0,0)
        """
        # move nose to (0,0,0)
        nose = self.get_point_by_position('NOSE')
        centralized_position = self.point_list - nose
        return Frame(centralized_position)

@dataclass
class Frames:
    """
    Frames is a container of frame
    """
    frame_list: List[Frame]

    def __getitem__(self, idx: int) -> Frame:
        """
        Get a specific frame from the container by its index.

        :param idx: The index of the requested frame (0-based).
        :return: The requested Frame object.
        :raises IndexError: If the index is out of range.
        """
        if idx >= len(self.frame_list):
            raise IndexError("Index out of range")
        return self.frame_list[idx]

    def __len__(self) -> int:
        """
        Get the number of frames in the container.

        :return: The number of frames in the container.
        """
        return len(self.frame_list)


    def filter_by_position(self, positions: List[str]) -> 'Frames':
        """
        Filter the frames by position.

        :param positions: A list of position names to keep.
        :return: A new Frames object containing only the selected positions.
        """
        assert all(pos in KEYPOINTS for pos in positions), "Invalid position names"
        new_frames = []
        for f in self.frame_list:
            point_list = [f.get_point_by_position(pos) for pos in positions]
            new_frames.append(Frame(np.array(point_list)))
        return Frames(new_frames)
    
    def average_frame(self) -> Frame:
        """
        Calculate the average frame from all frames in the container
        
        :return: The average frame as a Frame object
        """
        if not self.frame_list:
            raise ValueError("No frames in container")

        num_frames = len(self.frame_list)
        if num_frames == 1:
            return self[0]

        avg_points = np.mean([f.point_list for f in self.frame_list], axis=0)
        return Frame(avg_points)
    
    def to_relative_position(self) -> 'Frames':
        return Frames([frame.to_relative_position() for frame in self.frame_list])