# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from utils import performance_fit
from utils import L1RankLoss
import torch.nn as nn

from my_dataloader import VideoDataset_spatio_temporal_brightness
from final_fusion_model import swin_small_patch4_window7_224 as create_model

from torchvision import transforms
import time
import xlsxwriter as xw

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)

    weights = 'ckpts/last2_SI+TI_epoch_16_SRCC_0.937369.pth'
    weights_dict = torch.load(weights, map_location=device)
    print(model.load_state_dict(weights_dict))

    videos_dir =  'key_frames'
    data_dir_3D =  'temporal_feature'
    brightness =  'brightness_consistency'
    # datainfo_train = 'data/my_train.csv'
    datainfo_test = 'data/my_train.csv'
    # transformations_train = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transformations_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # trainset = VideoDataset_spatio_temporal_brightness(videos_dir, data_dir_3D, brightness, datainfo_train, transformations_train, 'my_train', 'SlowFast')
    testset = VideoDataset_spatio_temporal_brightness(videos_dir, data_dir_3D, brightness, datainfo_test, transformations_test, 'my_test', 'SlowFast')

    ## dataloader
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size,
    #                                            shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=args.num_workers)
    workbook = xw.Workbook('data.xlsx')
    worksheet = workbook.add_worksheet()
    with torch.no_grad():
        model.eval()
        label = np.zeros([len(testset)])
        y_output = np.zeros([len(testset)])
        for i, (video, tem_f, mos, name) in enumerate(test_loader):
            print(name)
            video = video.to(device)
            tem_f = tem_f.to(device)
            video = torch.reshape(video, [video.shape[0] * video.shape[1], 3, 512, 960])
            tem_f = torch.reshape(tem_f, [tem_f.shape[0] * tem_f.shape[1], 2304 + 144])
            label[i] = mos.item()
            outputs = model(video, tem_f)
            y_output[i] = torch.mean(outputs).item()
            print(y_output[i])
            worksheet.write(i, 1, y_output[i])

        workbook.close()

        # test_PLCC, test_SRCC, test_KRCC, test_RMSE = performance_fit(label, y_output)
        #
        # print(
        #     'The result on the test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
        #         test_SRCC, test_KRCC, test_PLCC, test_RMSE))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--database', type=str)
    parser.add_argument('--model_name', type=str)
    # training parameters
    parser.add_argument('--conv_base_lr', type=float, default=1e-5)
    parser.add_argument('--decay_ratio', type=float, default=0.95)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--n_trial', type=int, default=0)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--exp_version', type=int)
    parser.add_argument('--print_samples', type=int, default=1000)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    # misc
    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--loss_type', type=str, default='L1RankLoss')

    parser.add_argument('--weights', type=str, default='swin_small_patch4_window7_224.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=True)

    args = parser.parse_args()
    main(args)