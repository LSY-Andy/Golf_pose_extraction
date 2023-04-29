from src.matchers.base_matcher import BaseMatcher
import numpy as np
import cv2
from util.process import extract_keyframe
from src.matchers.matcher_factory import matcher_factory


def visualize_matching(learner_path,matcher_learner) -> list :     
    # 获取匹配到的帧的索引、损失值和关键点信息
    matched_frames, losses, kp_extract = matcher_learner
    frames = []
    
    # 迭代所有匹配到的帧
    for i,frame_idx in enumerate(matched_frames):
        # load frames
        cap = cv2.VideoCapture(learner_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, frame = cap.read()
        cap.release()

        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.resize(frame, (600,950))

        # Define the presentation sidebar
        display_width = 300
        display_height = frame.shape[0]
        display = np.zeros((display_height, display_width, 3), dtype=np.uint8)

        # Draw the text and color
        loss_text = f'Loss: {losses[i]:.2f}'
        cv2.putText(display, loss_text, (20, 80), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        # other information
        # TODO add the other loss details later
        cv2.putText(display, 'Detail loss', (20, 180), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, 'Feature 1', (20, 280), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, 'Feature 2', (20, 340), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, 'Feature 3', (20, 400), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, 'Feature 4', (20, 460), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        # combine the display with golf frames
        cv2.rectangle(display, (0, 0), (display_width, display_height), (255, 255, 255), 3)
        combined = np.concatenate((display, frame),axis=1)

        # show the visualization frames
        frames.append(combined)
        # 将所有帧的图像连接在一起
    return frames
    #返回学习者关键帧(loss值标注图像)

def generate_video(learner_path,match_learner):
    frames_return = visualize_matching(learner_path,match_learner)
    height,width,_ = frames_return[0].shape

    fps = 1

    # 使用cv2.VideoWriter创建一个新视频文件 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 设置编解码器
    out = cv2.VideoWriter('Learner_feedback_video.mp4', fourcc, fps,(width,height))

    # 遍历frames列表中的所有元素，并将每个元素写入新视频文件中
    for frame in frames_return:
        # 将numpy数组转换为OpenCV的图像格式
        frame = frame.astype(np.uint8)
        # 将帧写入视频中
        out.write(frame)
    # 关闭新视频文件并释放所有资源
    out.release()

    cv2.destroyAllWindows()

