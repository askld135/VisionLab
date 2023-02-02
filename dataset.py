import numpy as np
import sys
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2


root = "./RGBW_training_dataset_fullres/"
sys.path.append(root)
sys.path.append("./RGBW_training_dataset_fullres/rgbw_rmsc_start_here/simple_ISP/")
from rgbw_rmsc_start_here.data_scripts import process_function as pf

crop_size = 256
 
def denormalize(x):
    rgb_mean = torch.tensor([0.4931, 0.4950, 0.4936]).cuda().view(1, 3, 1, 1)
    rgb_std = torch.tensor([0.2774, 0.2877, 0.3138]).cuda().view(1, 3, 1, 1)
    y = x * rgb_std + rgb_mean
    return y

class Raw2RgbDataset(Dataset):
    def __init__(self, raw_dir, gt_dir, db_type='train'):
        self.raw_dir = raw_dir
        self.raw_list = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir)]
        self.raw_list.sort()
        
        self.gt_dir = gt_dir
        self.gt_list = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir)]
        self.gt_list.sort()
        self.to_tensor = transforms.ToTensor()
        self.crop_size = 256
        self.db_type = db_type

    def __len__(self):
        return len(self.raw_dir)

    def __getitem__(self, idx):
        raw_img_path = self.raw_list[idx]
        raw_img = pf.read_bin_file(raw_img_path)
        raw_tensor = self.to_tensor(raw_img) / 255.

        gt_img_path = self.gt_list[idx]
        gt_img = pf.read_bin_file(gt_img_path)
        gt_tensor = self.to_tensor(gt_img) / 255.
        
        # if self.db_type == 'train':
            
        #     raw_tensor, gt_tensor = self.random_crop(raw_tensor, gt_tensor)
        
            
        return raw_tensor, gt_tensor
    
    
    
    def random_crop(self, input, target):
        h = input.size(-2)
        w = input.size(-1)
        
        
        rand_w = torch.randint((w - self.crop_size) // 4, [1, 1]) * 4
        rand_h = torch.randint((h - self.crop_size) // 4, [1, 1]) * 4
        
        
        input = input[:, rand_h : rand_h + self.crop_size, rand_w : rand_w + self.crop_size]
        target = target[:, rand_h : rand_h + self. crop_size, rand_w : rand_w + self.crop_size]
        
        return input, target


#make binary file list
input_0dB_dir = os.path.join(root, "input/train_RGBW_full_input_0dB")
input_0dB_list = [os.path.join(input_0dB_dir, f) for f in os.listdir(input_0dB_dir)]




input_24dB_dir = os.path.join(root, "input/train_RGBW_full_input_24dB")
input_24dB_list = [os.path.join(input_24dB_dir, f) for f in os.listdir(input_24dB_dir)]

input_42dB_dir = os.path.join(root, "input/train_RGBW_full_input_42dB")
input_42dB_list = [os.path.join(input_42dB_dir, f) for f in os.listdir(input_42dB_dir)]

input_0dB_list.sort()
input_24dB_list.sort()
input_42dB_list.sort()

gt_dir = os.path.join(root, "GT_bayer/train_bayer_full_gt")
gt_list = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir)]

gt_list.sort()


# for i in range(len(gt_list)):
#     image_0dB = pf.read_bin_file(input_0dB_list[i])
#     image_24dB = pf.read_bin_file(input_24dB_list[i]) 
#     image_42dB = pf.read_bin_file(input_42dB_list[i])
#     gt = pf.read_bin_file(gt_list[i])
#     for j in range(1, 101):
#         h = image_0dB.shape[-2]
#         w = image_0dB.shape[-1]
#         if h <= crop_size or w <= crop_size:
#             rand_h = 0
#             rand_w = 0
#         else:
#             rand_h = np.random.randint(0, (h - crop_size) // 4) * 4
#             rand_w = np.random.randint(0, (w - crop_size) // 4) * 4
        
#         image_0dB = image_0dB[rand_h : rand_h + crop_size, rand_w : rand_w + crop_size]
#         image_24dB = image_24dB[rand_h : rand_h + crop_size, rand_w : rand_w + crop_size]
#         image_42dB = image_42dB[rand_h : rand_h + crop_size, rand_w : rand_w + crop_size]
#         gt = gt[rand_h : rand_h + crop_size, rand_w : rand_w + crop_size]
       
#         pf.save_bin(os.path.join(root, "input/crops/image_0dB/image0%d_%d"%(i+1,j)),image_0dB)
#         pf.save_bin(os.path.join(root, "input/crops/image_24dB/image0%d_%d"%(i+1,j)),image_24dB)
#         pf.save_bin(os.path.join(root, "input/crops/image_42dB/image0%d_%d"%(i+1,j)),image_42dB)
#         pf.save_bin(os.path.join(root, "input/crops/image0%d_%d"%(i+1,j)),gt)
    
        
        
    
            



















