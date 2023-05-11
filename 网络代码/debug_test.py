import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

import torchvision.transforms

# from workspace_utils import active_session
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
torch.multiprocessing.set_sharing_strategy('file_system')
import valdata_meant60
from valdata_meant60 import Val_meanT60
from torch.utils.tensorboard import SummaryWriter
# 决定使用哪块GPU进行训练
import csv
import os
import glob
from new_data_load_original import Dataset_dict, collate_fn

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device_ids = [0, 1]
## TODO: Once you've define the network, you can instantiate it
# one example conv layer has been provided for you
from resnet101 import ResNet101

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor
import torch.optim as optim
import datetime
from valdata_meant60 import ValDataset, Val_meanT60
import argparse

torch.backends.cudnn.benchmark = True


def parse_args():
    # Test路径见下面 val_dict_root
    parser = argparse.ArgumentParser(description='Evaluate the mmTransformer')
    parser.add_argument('--trained_epoch', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='save_model/New_attention_2022-7-12-afternoon')
    # parser.add_argument('--model_path', type=str, default='/Users/bajianxiang/Desktop/internship/new_dataset_t60_attention/save_model/resnet_1000Hz/t60_predict_model_0_fullT60_rir_timit_noise.pt')
    parser.add_argument('--model_path', type=str,default="/mnt/sda/xbj/t60_resnet101/save_model/resnet_2000Hz/") # 记得改成对应的路径，最后要带/
    parser.add_argument('--epoch_for_save', type=int, default=0)  # 记得改epoch数
    parser.add_argument('--load_pretrain', type=bool, default=True)
    # 125Hz:(0, 7); 250Hz:(1, 10); 500Hz:(2, 13); 1kHz:(3, 16); 2kHz:(4, 19)
    parser.add_argument('--which_freq', type=int, default=4)
    parser.add_argument('--freq_loc', type=int, default=19)    # 记得改成对应的倍频程
    parser.add_argument('--start_freq', type=int, default=0)   # 记得改成对应的倍频程
    parser.add_argument('--end_freq', type=int, default=29)
    parser.add_argument('--ln_out', type=int, default=1)
    parser.add_argument('--SERVER', type=bool, default=True)  # 在服务器上运行记得改成True
    parser.add_argument('--outputresult_dir', type=str, default="resnet101_test_debug") # 记得改输出路径，不用带/应该

    # parser.add_argument('--epoch',type = int,default = 170)
    args = parser.parse_args()
    return args


def load_checkpoint(checkpoint_path=None, trained_epoch=None, model=None, device=None):
    save_model = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(save_model['model'])
    trained_epoch_load = save_model['epoch']
    # trained_epoch = state['epoch']
    print('model loaded from %s' % checkpoint_path)
    return_epoch = 0
    if not trained_epoch is None:
        return_epoch = trained_epoch
    else:
        return_epoch = trained_epoch_load

    return model, return_epoch


def net_sample_output():
    for i, sample in enumerate(val_loader):

        images = sample['image']
        ddr = sample['ddr']
        t60 = sample['t60']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = torch.tensor(images.clone().detach(), dtype=torch.float32, device=device)

        # forward pass to get net output
        t60_predict = net(images)

        # break after first image is tested
        if i == 0:
            return images, t60_predict, t60


def val_net(net, epoch, val_loader, writer):
    with torch.no_grad():
        net.eval()
        total_mean_loss = 0
        total_mean_bias = 0
        progress_bar = tqdm(val_loader)
        for j, datas in enumerate(progress_bar):
            images_reshape = datas['image'].to(torch.float32).to(device)  # 服务器上这个to(device)改了，得在CPU上运算
            valid_len = datas['validlen']
            gt_t60_reshape = datas['t60'].to(torch.float32).to(device).unsqueeze(1)

            gt_t60_reshape = gt_t60_reshape.T[args.freq_loc].T  # [total_slices, 1]
            output_pts = net(images_reshape)

            loss = criterion(output_pts, gt_t60_reshape)
            bias = torch.sum((gt_t60_reshape - output_pts)) / output_pts.shape[0]

            total_mean_loss += loss.item()
            total_mean_bias += bias.item()

        mean_loss = total_mean_loss / len(val_loader)
        mean_bias = float(total_mean_bias) / len(val_loader)
        if not writer is None:
            writer.add_scalar('val/mean_loss', mean_loss, epoch)
            writer.add_scalar('val/mean_bias', mean_bias, epoch)

        print("Epoch %d Evaluation: mean loss:%f,mean bias:%f" % (epoch, mean_loss, mean_bias))


def output_result_analysis(result_dict, output_dir, test_path):
    test_dataset = test_path.split("/")[-2]
    if not test_dataset:
        test_dataset = "hahahah"
    csv_file = os.path.join(output_dir, test_dataset + ".csv")

    f = open(csv_file, "w")
    csv_writer_normal = csv.writer(f)
    bias_lst = []
    for key, value in result_dict.items():

        freq_name = 125 * 2 ** args.which_freq
        if freq_name < 1000:
            freq_name = str(freq_name) + 'Hz'
        else:
            freq_name = str(int(freq_name / 1000)) + 'k Hz'
        csv_writer = csv_writer_normal
        csv_writer.writerow([0, freq_name])
        csv_writer.writerow([str(key)])

        for k in range(len(result_dict[key]["output_list"])):
            csv_writer.writerow(["output%d" % (k)] + result_dict[key]["output_list"][k].cpu().numpy().tolist())
        csv_writer.writerow(["mean_output"] + result_dict[key]["mean_output"].cpu().numpy().tolist())
        csv_writer.writerow(["gt"] + result_dict[key]["gt"].cpu().numpy().tolist())
        csv_writer.writerow(["mean_bias"] + result_dict[key]["mean_bias"].cpu().numpy().tolist())
        bias_lst.extend(result_dict[key]["mean_bias"].cpu().numpy().tolist())
        csv_writer.writerow(["mean_mse"] + (result_dict[key]["mean_bias"] ** 2).cpu().numpy().tolist())
        csv_writer.writerow([" "])
    return bias_lst


def test_net(net, epoch, val_loader, writer, batch_size):
    result_dict = dict()
    with torch.no_grad():
        net.eval()

        total_mean_loss = torch.zeros((1, args.ln_out))
        total_mean_bias = torch.zeros((1, args.ln_out))

        progress_bar = tqdm(val_loader)
        useless_count = 0
        for j, datas in enumerate(progress_bar):

            images_reshape = datas['image'].to(torch.float32).to(device)

            gt_t60_reshape = datas['t60'].to(torch.float32).to(device).unsqueeze(1)
            valid_len = datas['validlen']
            names = datas['name']
            print('images_reshape.shape = ', images_reshape.shape)
            # for i in range(images_reshape.shape[0]):
            #     temp_img = images_reshape[i].squeeze()
            #     try:
            #         print('successfully save ')
            #         plt.imshow(temp_img.cpu())
            #         plt.savefig('/mnt/sda/xbj/check_img/image'+ str(i) + '.jpg')
            #     except:
            #         print('unsuccessfully save ')
            #         temp_img = temp_img[0]
            #         plt.imshow(temp_img.cpu())
            #         plt.savefig('/mnt/sda/xbj/check_img/image' + str(i)  + '.jpg')

            gt_t60_reshape = gt_t60_reshape.T[args.freq_loc].T  # [total_slices, 1]
            output_pts = net(images_reshape)  # [29,1]

            bias = gt_t60_reshape - output_pts
            rsquare_error = torch.sqrt(bias ** 2)
            abs_bias = torch.abs(gt_t60_reshape - output_pts)

            if not torch.isnan(rsquare_error).all():
                total_mean_loss += torch.mean(torch.mean(rsquare_error, dim=0), dim=0).cpu().detach()
                total_mean_bias += torch.mean(torch.mean(bias, dim=0), dim=0).cpu().detach()

            else:
                useless_count += 1

            for i in range(len(valid_len)):
                start_num = 0
                if i > 0:
                    start_num = sum(valid_len[0:i])

                output_list = [output_pts[k] for k in range(start_num, start_num + valid_len[i])]
                mean_output = torch.mean(output_pts[start_num:start_num + valid_len[i]], dim=0)
                mean_abs_bias = torch.mean(abs_bias[start_num:start_num + valid_len[i]], dim=0)
                mean_bias = torch.mean(bias[start_num:start_num + valid_len[i]], dim=0)
                mean_rsquare_error = torch.mean(rsquare_error[start_num:start_num + valid_len[i]], dim=0)
                mean_gt = torch.mean(gt_t60_reshape[start_num:start_num + valid_len[i]], dim=0)
                result_dict[names[i][0]] = {"mean_output": mean_output, "mean_bias": mean_bias, "gt": mean_gt,
                                            "square_error": mean_rsquare_error, "output_list": output_list}

        mean_loss = total_mean_loss / (len(val_loader) - batch_size * useless_count)
        mean_bias = total_mean_bias / (len(val_loader) - batch_size * useless_count)
        if not writer is None:
            writer.add_scalar('val/mean_loss', mean_loss, epoch)
            writer.add_scalar('val/mean_bias', mean_bias, epoch)

        print("Mean loss:", mean_loss)
        print("Mean bias:", mean_bias)
        # print("Epoch %d Evaluation: mean loss:%f,mean bias:%f" %(epoch,mean_loss,mean_bias))

        return result_dict


class StepLR:
    def __init__(self, lr, lr_epochs):
        assert len(lr) - len(lr_epochs) == 1
        self.warmup = False
        self.warmup_epochs = 10
        self.min_lr = min(lr)
        self.max_lr = max(lr)
        # self.lr = lr
        # self.lr_epochs = lr_epochs

        # self.lr_warmup = lambda epoch_in : min(self.lr)+0.5*(max(self.lr) - min(self.lr))*(1+np.cos((epoch_in-self.warmup_epochs)*PI/(2*self.warmup_epochs)))
        self.lr_warmup = lambda epoch_in: self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + np.cos((epoch_in - self.warmup_epochs) * PI / (2 * self.warmup_epochs)))
        if self.warmup == True:
            self.lr_epochs = [self.warmup_epochs] + [i + self.warmup_epochs for i in lr_epochs]
            self.lr = [self.lr_warmup] + lr
        else:
            self.lr_epochs = lr_epochs
            self.lr = lr

    def __call__(self, epoch):
        idx = 0
        for lr_epoch in self.lr_epochs:
            if epoch < lr_epoch:
                break
            idx += 1
        if self.warmup == True and idx == 0:
            return self.lr[idx](epoch)
        else:
            return self.lr[idx]


if __name__ == "__main__":

    args = parse_args()

    DEBUG = args.SERVER
    LOAD_PRETRAIN = args.load_pretrain

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = args.model_path + 't60_predict_model_' + str(args.epoch_for_save) + '_fullT60_rir_timit_noise.pt'

    net = ResNet101(ln_out=args.ln_out)
    if LOAD_PRETRAIN == True:
        start_time = time.time()
        net, trained_epoch = load_checkpoint(model_path, 99, net, device)
        print('Successfully Loaded model: {}'.format(model_path))
        print('Finished Initialization in {:.3f}s!!!'.format(
            time.time() - start_time))
    else:
        trained_epoch = 0
    net.to(device)

    # print(net)

    criterion = torch.nn.MSELoss()

    # optimizer = optim.Adam([{'params':net.parameters(),'initial_lr':0.0001}], lr=0.0001,weight_decay=0.0001)
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_transform = transforms.Resize([224, 224])


    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {"Total": total_num, "Trainable": trainable_num}


    print(get_parameter_number(net))

    if DEBUG == False:
        val_dict_root = "/Users/bajianxiang/Desktop/internship/filter_down_spec_dataset"
        val_batch_size = 8
        failed_file = "/Users/bajianxiang/Desktop/internship/filter_down_spec_dataset/111/koli-national-park-winter_koli_snow_site4_1way_bformat_1_koli-national-park-winter_东北话男声_1_TIMIT_a005_100_110_0dB-0.pt"
    else:
        val_dict_root = "/data/xbj/0815_GEN_DATASET/0815_GEN_DATASET/val"
        val_batch_size = 3
        failed_file = "/data/xbj/0815_GEN_DATASET/0815_GEN_DATASET/val/hoffmann-lime-kiln-langcliffeuk/hoffmann-lime-kiln-langcliffeuk_ir_p2_0_1_hoffmann-lime-kiln-langcliffeuk_东北话男声_1_TIMIT_a053_110_120_0dB-0.pt"

    val_transformed_dataset = Dataset_dict(root_dir=val_dict_root, transform=data_transform, start_freq=args.start_freq,
                                           end_freq=args.end_freq, which_freq=args.which_freq, failed_file=failed_file)
    print("len of val dataset:", len(val_transformed_dataset))
    # print('Number of images: ', len(transformed_dataset))
    if DEBUG == False:

        val_loader = torch.utils.data.DataLoader(val_transformed_dataset, shuffle=False, num_workers=0,
                                                 batch_size=val_batch_size, drop_last=True,
                                                 collate_fn=collate_fn)
    else:
        val_loader = torch.utils.data.DataLoader(val_transformed_dataset,
                                                 shuffle=True, num_workers=4,
                                                 batch_size=val_batch_size, drop_last=True, prefetch_factor=100,
                                                 collate_fn=collate_fn)
    print("after train loader init")
    trained_epoch = args.trained_epoch
    # train_net(trained_epoch,n_epochs, train_loader,val_loader,batch_size,args)
    test_output_result = test_net(net, trained_epoch, val_loader, None, val_batch_size)
    outputresult_dir = args.outputresult_dir
    if not os.path.exists(outputresult_dir):
        os.makedirs(outputresult_dir)

    freq_name = 125 * 2 ** args.which_freq
    if freq_name < 1000:
        freq_name = str(freq_name) + 'Hz'
    else:
        freq_name = str(int(freq_name / 1000)) + 'kHz'

    bias_lst = output_result_analysis(test_output_result, outputresult_dir, val_dict_root)

    fig = plt.figure(figsize=(10, 5))  # 设置画布的尺寸
    fig_name = 'Test_' + freq_name + '_epoch' + str(args.epoch_for_save)
    plt.title(fig_name, fontsize=20)  # 标题，并设定字号大小
    labels = ['mean_bias']
    plt.boxplot(bias_lst, labels=labels)
    plt.show()  # 存储图像
    plt.savefig(outputresult_dir + '/' + fig_name+'.png', dpi=fig.dpi)
    # TODO 训练时要改dict，权重名字，验证时保存的文件名和路径
