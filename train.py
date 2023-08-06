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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))


    # weights = 'ckpts/last2_SI+TI_epoch_10_SRCC_0.930263.pth'
    # weights_dict = torch.load(weights, map_location=device)
    # print(model.load_state_dict(weights_dict))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=0.00002, weight_decay=0.0000001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)

    if args.loss_type == 'L1RankLoss':
        criterion = L1RankLoss(batchsize=args.train_batch_size)

    param_num = 0
    for param in pg:
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    videos_dir = 'key_frames'
    data_dir_3D = 'temporal_feature'
    brightness = 'brightness_consistency'
    datainfo_train = 'data/my_train.csv'
    datainfo_test = 'data/my_val.csv'
    transformations_train = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transformations_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainset = VideoDataset_spatio_temporal_brightness(videos_dir, data_dir_3D, brightness, datainfo_train, transformations_train, 'my_train', 'SlowFast')
    testset = VideoDataset_spatio_temporal_brightness(videos_dir, data_dir_3D, brightness, datainfo_test, transformations_test, 'my_test', 'SlowFast')

    ## dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size,
                                               shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=args.num_workers)


    best_test_criterion = -1  # SROCC min
    best_test = []

    print('Starting training:')

    old_save_name = None

    for epoch in range(args.epochs):
        model.train()
        batch_losses = []
        batch_losses_each_disp = []
        session_start_time = time.time()
        for i, (video, tem_f, mos, _) in enumerate(train_loader):

            video = video.to(device)
            tem_f = tem_f.to(device)
            video = torch.reshape(video, [video.shape[0] * video.shape[1], 3, 720, 1280])
            tem_f = torch.reshape(tem_f, [tem_f.shape[0] * tem_f.shape[1], 2304 + 144])
            labels = mos.to(device).float()
            outputs = model(video, tem_f)
            optimizer.zero_grad()

            loss = criterion(outputs, labels)
            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())
            loss.backward()

            optimizer.step()

            if (i + 1) % (args.print_samples // args.train_batch_size) == 0:
                session_end_time = time.time()
                avg_loss_epoch = sum(batch_losses_each_disp) / (args.print_samples // args.train_batch_size)
                print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' % \
                      (epoch + 1, args.epochs, i + 1, len(trainset) // args.train_batch_size, \
                       avg_loss_epoch))
                batch_losses_each_disp = []
                print('CostTime: {:.4f}'.format(session_end_time - session_start_time))
                session_start_time = time.time()

        avg_loss = sum(batch_losses) / (len(trainset) // args.train_batch_size)
        print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

        scheduler.step()
        lr = scheduler.get_last_lr()
        print('The current learning rate is {:.06f}'.format(lr[0]))

        # do validation after each epoch
        with torch.no_grad():
            model.eval()
            label = np.zeros([len(testset)])
            y_output = np.zeros([len(testset)])
            for i, (video, tem_f, mos, _) in enumerate(test_loader):
                video = video.to(device)
                tem_f = tem_f.to(device)
                video = torch.reshape(video, [video.shape[0] * video.shape[1], 3, 720, 1280])
                tem_f = torch.reshape(tem_f, [tem_f.shape[0] * tem_f.shape[1], 2304 + 144])
                label[i] = mos.item()
                outputs = model(video, tem_f)
                y_output[i] = torch.mean(outputs).item()

            test_PLCC, test_SRCC, test_KRCC, test_RMSE = performance_fit(label, y_output)

            print(
                'Epoch {} completed. The result on the test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                    epoch + 1, \
                    test_SRCC, test_KRCC, test_PLCC, test_RMSE))


            if test_SRCC > best_test_criterion:
                print("Update best model using best_test_criterion in epoch {}".format(epoch + 1))
                best_test_criterion = test_SRCC
                best_test = [test_SRCC, test_KRCC, test_PLCC, test_RMSE]
                print('Saving model...')
                if not os.path.exists(args.ckpt_path):
                    os.makedirs(args.ckpt_path)

                if epoch > 0:
                    if os.path.exists(old_save_name):
                        os.remove(old_save_name)

                # save_model_name = os.path.join(config.ckpt_path, config.model_name + '_' + \
                #                                config.database + '_' + config.loss_type + '_NR_v' + str(
                #     config.exp_version) \
                #                                + '_epoch_%d_SRCC_%f.pth' % (epoch + 1, test_SRCC))
                save_model_name = args.ckpt_path + '/' + 'last2_SI+TI_epoch_%d_SRCC_%f.pth' % (epoch + 1, test_SRCC)
                torch.save(model.state_dict(), save_model_name)
                old_save_name = save_model_name

    print('Training completed.')
    print(
        'The best training result on the test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            best_test[0], best_test[1], best_test[2], best_test[3]))


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
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
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