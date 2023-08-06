
import math
import os
import numpy
from torchvision import transforms
import torch
from PIL import Image


imgs_path = 'key_frames'
save_folder = 'brightness_consistency'
transform= transforms.Compose(
        [transforms.Resize([720, 1280]), transforms.ToTensor(), transforms.Normalize(mean=[0.485], std=[0.229])])
count = 0
for imgs in os.listdir(imgs_path):
    img_path = os.path.join(imgs_path, imgs)
    print(img_path)
    length = len(os.listdir(img_path))
    number = math.floor(length/8)
    print(number * 8)
    brightness = torch.zeros([number * 8, 144])
    i = 0
    for img in os.listdir(img_path):
        imge_name = os.path.join(img_path, img)
        read_frame = Image.open(imge_name)
        read_frame = read_frame.convert('L')
        read_frame = transform(read_frame)
        read_frame = torch.reshape(read_frame, [144, 80, 80])
        read_frame = torch.flatten(read_frame, start_dim=1)
        brightness_mean = torch.mean(read_frame, dim=1)
        brightness[i] = brightness_mean
        i += 1
        if i >= number * 8:
            break

    brightness = torch.reshape(brightness, [8, number, 144])
    brightness_consistency = torch.var(brightness, dim=1)
    save_path = os.path.join(save_folder, imgs)
    os.makedirs(save_path)

    for i in range(8):
        b = brightness_consistency[i].numpy()
        b = b.reshape(1, 144, 1, 1, 1)
        numpy.save(os.path.join(save_path, "brightness_consistency{}.npy".format(i)), b)



