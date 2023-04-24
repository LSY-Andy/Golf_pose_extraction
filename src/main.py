import argparse
from util.process import extract_keyframe


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract keyframes from a video.')
    parser.add_argument('--video_path', type=str, help='Path to the video file')
    parser.add_argument('--save_path', type=str, help='Directory to save extracted keyframes')
    parser.add_argument('--frame_interval', type=float, default=0.5, help='Interval between extracted frames (seconds)')

    args = parser.parse_args()

    extract_keyframe(args.video_path, args.save_path, args.frame_interval)