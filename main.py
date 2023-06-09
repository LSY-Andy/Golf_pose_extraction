import argparse
from util.process import extract_keyframe
from src.matchers.matcher_factory import matcher_factory
from src.visualize.base_visualization import generate_video
import yaml


def parser() -> argparse.ArgumentParser:
    '''
    init parser
    '''
    parser = argparse.ArgumentParser(conflict_handler='resolve', description='Extract keyframes from a video.')
    parser.add_argument('--config_file', type=argparse.FileType("r"), help="Path to configuration file", required=False)
    parser.add_argument('--backbone', type=str, help='The model for detector and matcher', required=False)
    parser.add_argument('--frame_interval', type=float, default=0.5, help='Interval between extracted frames (seconds)', required=False)
    parser.add_argument('--display', default=False, action='store_true', help='Whether or not to display the visualization', required=False)
    parser.add_argument('--standard_video_path', type=str, help='Path to the video file', required=False)
    parser.add_argument('--practice_video_path', type=str, help='Path to the video file', required=False)
    parser.add_argument('--standard_path', type=str, help='Directory to save extracted keyframes', required=False)
    return parser

def main() -> None:
    # define the parser and get config file
    parser_ = parser()
    args = parser_.parse_args()
    parsing_list = []

    # load the config file
    if args.config_file is not None:
        config = yaml.safe_load(args.config_file.read())
        for key, value in config.items():
            if key == "display":
                if value or vars(args)[key]:
                    parsing_list.extend(["--display"])
            elif value is not None:
                ext_val = ["--" + key, str(value)]
                parsing_list.extend(ext_val)
        args = parser_.parse_args(parsing_list)
    else:
        args = parser_.parse_args()

    extract_keyframe(args.standard_video_path, args.standard_path, args.frame_interval)
    matcher = matcher_factory[args.backbone](args.standard_path, args.display)
    match_learner = matcher.match(args.practice_video_path)
    generate_video(args.practice_video_path,match_learner)
    

if __name__ == '__main__':
    main()
