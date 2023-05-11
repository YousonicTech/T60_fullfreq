# -*- coding: utf-8 -*-
"""
@file      :  get_figs.py
@Time      :  2022/9/15 16:22
@Software  :  PyCharm
@summary   :
@Author    :  Bajian Xiang
"""

import torch
import matplotlib.pyplot as plt
import os

epoch = 63


def paint(image, clean, dereverb, savename, savedir):
    # 都是[224, 224]的图
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.imshow(image, origin='lower')
    plt.title('Reverb')

    plt.subplot(132)
    plt.imshow(clean, origin='lower')
    plt.title('Clean')

    plt.subplot(133)
    plt.imshow(dereverb, origin='lower')
    plt.title('Dereverb')

    plt.tight_layout()

    # save_name = 's2r4_3_粤语女声_1_TIMIT_a079_30_40_0dB-0_ssim75_slice0'

    save_path = os.path.join(savedir, savename) + '.jpg'
    plt.savefig(save_path)

    print('--save:', save_path)



if __name__ == "__main__":
    path = './image_result_epoch' + str(epoch) + '.pt'
    x = torch.load(path)
    for key, value in x.items():
        # key = '/data/xbj/0902_1000hz_with_clean/val/arthur-sykes-rymer-auditorium-university-york/arthur-sykes-rymer-auditorium-university-york_s2r4_3_arthur-sykes-rymer-auditorium-university-york_粤语女声_1_TIMIT_a079_30_40_0dB-0.pt'
        config = key.split('/')[-2]                                             # 'arthur-sykes-rymer-auditorium-university-york'
        config_room = key.split('/')[-1]
        room = config_room.split(config)[-2].strip('_')                         # s2r4_3
        utterance = config_room.split(config)[-1].strip('_').strip('.pt')       # 粤语女声_1_TIMIT_a079_30_40_0dB-0

        savedir = os.path.join('.', 'figs', config, room+'_'+utterance)                 #'./figs/arthur-sykes-rymer-auditorium-university-york/s2r4_3_粤语女声_1_TIMIT_a079_30_40_0dB-0'
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        # 准备存图
        # dict_keys(['clean', 'image', 'dereverb', 't60_gt', 't60_pred', 'ssim'])

        ssim = value['ssim']
        save_name = room + '_' + utterance + '_ssim' + str(float(ssim))[2:4] # 's2r4_3_粤语女声_1_TIMIT_a079_30_40_0dB-0_ssim75'
        # 画图
        for slice in range(value['image'].shape[0]):
            paint(value['image'][slice][0].cpu(), value['clean'][slice][0].cpu(), value['dereverb'][slice][0].cpu(), save_name + '_slice' + str(slice), savedir)



