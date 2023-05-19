import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
from torchvision import transforms
from torchvision.transforms import Normalize


def replace_nan(a):
    if torch.any(torch.isnan(a)):
        a = torch.where(torch.isnan(a), torch.full_like(a, 0), a)
    if torch.any(torch.isinf(a)):
        a = torch.where(torch.isinf(a), torch.full_like(a, 0), a)
    a = torch.clamp(a, 1e-10, 1e10)
    return a

def get_amplitude(aa):
    amp = replace_nan(aa.real ** 2 + aa.imag ** 2).sqrt().log10()
    return amp

def imageTransform(temp_image, model='attention'):
    if 'attention' or 'Attention' in model:
        temp_image = temp_image.unsqueeze(0)
    else:
        temp_image = temp_image[temp_image.shape[0] // 2:].unsqueeze(0)
    return temp_image


class Dataset_dict(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir=None, rir_dir=None, transform = None,start_freq = 0,end_freq = 29, failed_file='YouWillGetErrorIfNotPassingIt', random_choose_slice=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.freq_start = start_freq
        self.freq_end = end_freq
        self.dict_root = root_dir
        self.transform = transform
        self.failed_file = failed_file
        self.random_choose = random_choose_slice
        self.dict_data = []
        self.normalize = Normalize(mean=[0.5], std=[0.5])
        self.transform = transforms.Compose([transforms.Resize([224, 224])])
        self.rir_data = {}

        print("root dir:",root_dir)

        for (root, dirs, files) in os.walk(root_dir):
            if 'Cas' in root:
                continue
            print("root:",root,"dirs:",dirs,"len of files:",len(files))    
            self.dict_data =self.dict_data + [os.path.join(root,filen) for filen in files]
            self.root = root

        for (root, dirs, files) in os.walk(rir_dir):
            for file in files:
                temp = torch.load(os.path.join(root,file))
                for _, value in temp.items():
                    break
                assert len(value[0]) == 4
                lst = [self._Norm(x) for x in value[0]]
                lst = torch.cat(lst, dim=0).unsqueeze(0)
                self.rir_data[file.split('.')[0]] = lst
        print('len of total rir files:', len(self.rir_data))

    def _Norm(self, img):
        img = self.transform(get_amplitude(img).unsqueeze(0)) # [1, 224, 224]
        img = (img - img.min()) / (img.max() - img.min())   # to [0, 1]
        img = self.normalize(img)                # to mean=[0.5], std=[0.5]
        return img


    def __len__(self):
        return len(self.dict_data)

    def __getitem__(self, idx):
        images_list = []
        rirs_list = []
        ddr_list = []
        t60_list = []
        MeanT60_list = []
        name_list = []
        dict_data_name = self.dict_data[idx]
        audio_image_dict = dict()

        try:
            audio_image_dict = torch.load(dict_data_name)
        except:
            print("failed read file name:",dict_data_name)
            audio_image_dict = torch.load(self.failed_file)

        dict_keys = list(audio_image_dict.keys())[0]
        dict_data = audio_image_dict[dict_keys]
        valid_info_count = 0
        for list_c in range(len(dict_data)):
            if dict_data[list_c] ==0 :
                continue
            else:
                valid_info_count+=1

                if "image" in dict_data[list_c].keys():
                    temp_images = dict_data[list_c]['image']
                    temp_images = [self._Norm(x) for x in temp_images] # [tensor:(1, 224, 224) * 4]
                    temp_images = torch.cat(temp_images, dim=0).unsqueeze(0) # [1, 4, 224, 224]
                    images_list.append(temp_images)
                    t60_list.append(torch.unsqueeze(dict_data[list_c]['t60'][self.freq_start:self.freq_end],0)) # tensor:(1, 30)
                    ddr_list.append(torch.unsqueeze(dict_data[list_c]['ddr'][self.freq_start:self.freq_end],0)) 
                    MeanT60_list.append(torch.unsqueeze(dict_data[list_c]['MeanT60'],0))
                    name_list.append(dict_data_name)
                    rirs_list.append(self.rir_data[dict_data_name.split('/')[-1].split(dict_data_name.split('/')[-2])[1].strip('_')])

        images = torch.cat(images_list,dim=0) # [b, 4, 224, 224]
        ddr = torch.cat(ddr_list,dim=0) # [b, 29]
        t60 = torch.cat(t60_list,dim=0) # [b, 29]
        MeanT60 = torch.cat(MeanT60_list,dim=0) # [b, 30]
        rirs = torch.cat(rirs_list, dim=0)
        # print('shape:',images.shape, rirs.shape, t60.shape)

        sample = {'image': images, 'rir':rirs,'ddr': ddr, 't60':t60, "MeanT60":MeanT60,"validlen":valid_info_count,'name':name_list}


        return sample

padding_keys = []
stacking_keys = ['MeanT60','ddr','image','t60', 'rir']

def collate_fn(batch):
        keys = batch[0].keys()

        out = {k: [] for k in keys}

        for data in batch:
            for k,v in data.items():
                out[k].append(v)

        for k in stacking_keys:
            #print("key:",k)
            out[k] = torch.cat(out[k],dim=0) #torch.stack(out[k],dim=0)

        for k in padding_keys:
            out[k] = pad_sequence(out[k],batch_first = True)



        return out









