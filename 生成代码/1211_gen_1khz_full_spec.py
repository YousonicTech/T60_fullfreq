# -*- coding: utf-8 -*-
"""
@file      :  1211_gen_1khz_full_spec.py
@Time      :  2022/12/11 18:31
@Software  :  PyCharm
@summary   :
@Author    :  Bajian Xiang
"""


import numpy as np
from scipy.signal import lfilter
import wave
import glob
import os
import torch
import pandas as pd
import argparse
from gen_specgram import PipeLineNew


def get_config_rir_name(wav_file_name):
    room_name = wav_file_name.split('/')[-1].split('.')[0]
    config_name = wav_file_name.split('/')[-2]
    rir_name = room_name.split(config_name)[1].strip('_')
    return config_name, rir_name


##configuration area
chunk_length = 4
chunk_overlap = 0.5
# TODO 需要改变csv,numpydir/csv_save_path这些路径


parser = argparse.ArgumentParser(description='manual to this script')
# 切分用倍频程
parser.add_argument('--csv_file', type=str,
                    default="/mnt/sda/xbj/1000Hz.csv")
parser.add_argument('--dir_str', type=str,
                    default="/data/2000_Wav/1130_RIR_1khz/koli-national-park-winter")
parser.add_argument('--save_dir', type=str,
                    default="/data/xbj/1130_1000Hz_new/rir/koli-national-park-winter")
parser.add_argument('--rir_dir', type=str, default="/data/xbj/1130_RIR_1khz")

##functions
def SPLCal(x):
    Leng = len(x)
    pa = np.sqrt(np.sum(np.power(x, 2)) / Leng)
    p0 = 2e-5
    spl = 20 * np.log10(pa / p0)
    return spl


class Totensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, ddr, t60, meanT60 = sample['image'], sample['ddr'], sample['t60'], sample['MeanT60']

        # image, ddr, t60 = sample['image'], sample['ddr'], sample['t60']
        ddr = ddr.astype(float)
        t60 = t60.astype(float)
        meanT60 = meanT60.astype(float)
        # image = image.transpose((2, 0, 1))
        return {'image': image,
                'ddr': torch.from_numpy(ddr),
                't60': torch.from_numpy(t60),
                "MeanT60": torch.from_numpy(meanT60)
                }
        #


if __name__ == "__main__":

    args = parser.parse_args()
    save_dir = args.save_dir
    save_dir = args.save_dir
    dir_str = args.dir_str
    csv_file = args.csv_file

    csv_data = pd.read_csv(csv_file)
    print(csv_data)

    if not os.path.exists(args.save_dir):
        os.makedirs(save_dir)

    pip = PipeLineNew()

    for file_name in glob.glob(dir_str + r"/*.wav"):
        f = wave.open(file_name, "rb")
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        print(file_name)

        print(nchannels, sampwidth, framerate, nframes / framerate)

        str_data = f.readframes(nframes)
        f.close()
        wave_data = np.frombuffer(str_data, dtype=np.int16)
        wave_data.shape = -1, nchannels
        wave_data = wave_data.T
        print('wave_data.shape:', wave_data.shape)
        audio_time = nframes / framerate
        if audio_time <= 4:
            wave_data = np.append(wave_data, torch.zeros(16001*4-len(wave_data)))
        print('wave_data.shape after append:', wave_data.shape)
        chan_num = 0
        count = 0
        new_file_name = (file_name.split("\\")[-1]).split(".")[0]
        new_file_name = new_file_name.split("/")[-1]

        ## process each channel of audio
        for audio_samples_np in wave_data:
            whole_audio_SPL = SPLCal(audio_samples_np)

            available_part_num = (audio_time - chunk_overlap) // (
                    chunk_length - chunk_overlap)  # 4*x - (x-1)*0.5 <= audio_time    x为available_part_num

            if available_part_num == 1:
                cut_parameters = [chunk_length]
            else:
                cut_parameters = np.arange(chunk_length,
                                           (chunk_length - chunk_overlap) * available_part_num + chunk_overlap,
                                           chunk_length)

            start_time = int(0)  # 开始时间设为0
            count = 0
            # 开始存储pt文件
            dict = {}
            save_data = []
            for t in cut_parameters:
                stop_time = int(t)  # pydub以毫秒为单位工作
                start = int(start_time * framerate)
                end = int((start_time + chunk_length) * framerate)
                audio_chunk = audio_samples_np[start:end]  # 音频切割按开始时间到结束时间切割

                ##ingore chunks with no audio
                # chunk_spl = SPLCal(audio_chunk)
                # if whole_audio_SPL - chunk_spl >= 20:
                #     continue

                ##file naming

                count += 1

                ##A weighting
                # chunk_a_weighting = splweighting.weight_signal(audio_chunk, framerate)

                ##gammatone
                chunk_result = pip(audio_chunk, framerate)

                chan = chan_num + 1
                config = new_file_name.split("_")[0]  # +"_" + new_file_name.split("_")[1]
                if config == "dirac":
                    config = new_file_name.split("_")[0]  # +"_" + new_file_name.split("_")[1]
                    room = new_file_name.split(config)[1][1:-1]
                else:
                    config = new_file_name.split("_")[0]  # +"_" + new_file_name.split("_")[1]
                    room = new_file_name.split(config)[1][1:-1]
                print(new_file_name)

                a = (csv_data['Room:'] == room).values
                b = (csv_data['Room Config:'] == config).values

                data = csv_data[a & b]
                T60_data = data.loc[:, ['T60:']]
                FB_T60_data = data.loc[:, ['FB T60:']]
                FB_T60_M_data = data.loc[:, ['FB T60 Mean (Ch):']]
                DDR_each_band = np.array([0 for i in range(30)])
                T60_each_band = (T60_data.values).reshape(-1)
                MeanT60_each_band = np.array([FB_T60_data, FB_T60_M_data])
                image = chunk_result
                print('-- image shape:', [x.shape for x in image], ' --')
                sample = {'image': image, 'ddr': None, 't60': None, "MeanT60": None}
                transform = Totensor()
                sample = transform(sample)

                save_data.append(sample)
                start_time = start_time + chunk_length - chunk_overlap  # 开始时间变为结束时间前1s---------也就是叠加上一段音频末尾的4s
            if len(save_data) != 0:
                pt_file_name = os.path.join(save_dir, new_file_name + '-' + str(chan_num) + '.pt')
                dict[new_file_name + '-' + str(chan_num)] = save_data
                torch.save(dict, pt_file_name)
            chan_num = chan_num + 1
        print('----------------finish----------------')
