import math

import numpy as np
import os
import pandas as pd
import cv2
import scipy.io as scio

def extract_frame(videos_dir, video_name, save_folder):
    filename = os.path.join(videos_dir, video_name)
    video_name_str = video_name[:-4]
    video_capture = cv2.VideoCapture()
    video_capture.open(filename)
    cap=cv2.VideoCapture(filename)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # the heigh of frames
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # the width of frames

        
    dim = (video_width, video_height)
    video_read_index = 0
    frame_idx = 0
    number = math.floor(video_length/8)
    print(video_length)
    print(number)

    for i in range(video_length):
        if (i % number == 0 and video_read_index <= 7):
            has_frames, frame = video_capture.read()
            if has_frames:
                # key frame
                read_frame = cv2.resize(frame, dim)
                exit_folder(os.path.join(save_folder, video_name_str))
                cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                         '{:03d}'.format(video_read_index) + '.png'), read_frame)
                video_read_index += 1
    return
            
def exit_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)    
        
    return


videos_dir = 'LLVE-QA'
save_folder = 'key_frames'
for video_name in os.listdir(videos_dir):
    extract_frame(videos_dir, video_name, save_folder)
