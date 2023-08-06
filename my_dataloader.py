import os

import pandas as pd
from PIL import Image

import torch
from torch.utils import data
import numpy as np
import scipy.io as scio
import cv2



class VideoDataset_images(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, filename_path, transform, database_name):
        super(VideoDataset_images, self).__init__()


        if database_name == 'my_train':
            column_names = ['name', 't1','t2','mos']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'my_test':
            column_names = ['name', 't1','t2','mos']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()


        self.videos_dir = data_dir
        self.transform = transform
        self.length = len(self.video_names)
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if self.database_name == 'my_train' or self.database_name == 'my_test':
            video_name = self.video_names[idx]
            video_name_str = video_name.split('.')[0]

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))
        path_name = os.path.join(self.videos_dir, video_name_str)
        video_channel = 3
        video_height_crop = 720
        video_width_crop = 1280
        video_length_read = 8


        key_frames = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])

        for i in range(video_length_read):
            imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            key_frames[i] = read_frame


        return key_frames, video_score, video_name



class VideoDataset_temporal_feature(data.Dataset):
    def __init__(self,  temporal_feature, filename_path, database_name, feature_type):
        super(VideoDataset_temporal_feature, self).__init__()
        if database_name == 'my_train':
            column_names = ['name', 't1','t2','mos']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'my_test':
            column_names = ['name', 't1','t2','mos']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        self.temporal_feature = temporal_feature
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database_name == 'my_train' or self.database_name == 'my_test':
            video_name = self.video_names[idx]
            video_name_str = video_name.split('.')[0]

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))
        video_length_read = 8

        # read temporal features

        if self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.temporal_feature, video_name_str)
            temporal_feature = torch.zeros([video_length_read, 2048 + 256])
            for i in range(video_length_read):
                i_index = i
                feature_3D_slow = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                temporal_feature[i] = feature_3D

        return  temporal_feature, video_score, video_name


class VideoDataset_extract_temporal_feature(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, filename_path, transform, resize):
        super(VideoDataset_extract_temporal_feature, self).__init__()
        column_names = ['name', 't1', 't2', 'mos']

        dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                               encoding="utf-8-sig")

        self.video_names = dataInfo['name']
        self.score = dataInfo['mos']
        self.videos_dir = data_dir
        self.transform = transform
        self.resize = resize
        self.length = len(self.video_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names.iloc[idx]
        video_score = torch.FloatTensor(np.array(float(self.score.iloc[idx]))) / 20

        filename = os.path.join(self.videos_dir, video_name)

        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap = cv2.VideoCapture(filename)

        video_channel = 3

        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

        if video_frame_rate == 0:
            video_clip = 10
        else:
            video_clip = int(video_length / video_frame_rate)



        video_clip_min = 8

        video_length_clip = 32

        transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])

        transformed_video_all = []

        video_read_index = 0
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                read_frame = self.transform(read_frame)
                transformed_frame_all[video_read_index] = read_frame
                video_read_index += 1

        if video_read_index < video_length:
            for i in range(video_read_index, video_length):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]

        video_capture.release()

        for i in range(video_clip):
            transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            if (i * video_frame_rate + video_length_clip) <= video_length:
                transformed_video = transformed_frame_all[
                                    i * video_frame_rate: (i * video_frame_rate + video_length_clip)]
            else:
                transformed_video[:(video_length - i * video_frame_rate)] = transformed_frame_all[i * video_frame_rate:]
                for j in range((video_length - i * video_frame_rate), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i * video_frame_rate - 1]
            transformed_video_all.append(transformed_video)

        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                transformed_video_all.append(transformed_video_all[video_clip - 1])

        return transformed_video_all, video_score, video_name



class VideoDataset_images_with_temporal_features(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, data_dir_3D, filename_path, transform, database_name, feature_type):
        super(VideoDataset_images_with_temporal_features, self).__init__()


        if database_name == 'my_train':
            column_names = ['name', 't1','t2','mos']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'my_test':
            column_names = ['name', 't1','t2','mos']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()


        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.transform = transform
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if self.database_name == 'my_train' or self.database_name == 'my_test':
            video_name = self.video_names[idx]
            video_name_str = video_name.split('.')[0]

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name_str)

        video_channel = 3
        video_height_crop = 720
        video_width_crop = 1280
        video_length_read = 8


        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])

        for i in range(video_length_read):
            imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame

        # read temporal features

        if self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048 + 256])
            for i in range(video_length_read):
                i_index = i
                feature_3D_slow = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                transformed_feature[i] = feature_3D

        return transformed_video, transformed_feature, video_score, video_name


class VideoDataset_spatio_temporal_brightness(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, data_dir_3D, brightness, filename_path, transform, database_name, feature_type):
        super(VideoDataset_spatio_temporal_brightness, self).__init__()


        if database_name == 'my_train':
            column_names = ['name', 't1','t2','mos']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'my_test':
            column_names = ['name', 't1','t2','mos']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()


        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.transform = transform
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.database_name = database_name
        self.brightness = brightness

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if self.database_name == 'my_train' or self.database_name == 'my_test':
            video_name = self.video_names[idx]
            video_name_str = video_name.split('.')[0]

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name_str)

        video_channel = 3
        video_height_crop = 512
        video_width_crop = 960
        video_length_read = 8


        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])
        j = 0
        for img in os.listdir(path_name):
            imge_name = os.path.join(path_name, img)
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[j] = read_frame
            j += 1

        # read temporal features

        if self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            brightness_folder_name = os.path.join(self.brightness, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048 + 256 + 144])
            for i in range(video_length_read):
                i_index = i

                feature_3D_slow = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()

                feature_3D_fast = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()

                brightness_consistency = np.load(
                    os.path.join(brightness_folder_name, 'brightness_consistency' + str(i_index) + '.npy')
                )
                brightness_consistency = torch.from_numpy(brightness_consistency)
                brightness_consistency = brightness_consistency.squeeze()
                brightness_consistency *= 10

                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast, brightness_consistency])
                transformed_feature[i] = feature_3D

        return transformed_video, transformed_feature, video_score, video_name
