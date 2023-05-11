# -*- coding: utf-8 -*-
"""
@file      :  cnn_split_freq_xbj.py
@Time      :  2022/7/19 16:05
@Software  :  PyCharm
@summary   :  New model: Split 7 frequencies in case of interference
@Author    :  Bajian Xiang
"""

import torch
# from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F
from torchsummary import summary
import torch.nn as nn


def cal_count_len(valid_len):
    count_len = [0]
    for i in valid_len:
        count_len.append(count_len[-1] + i)
    return count_len


class Net(nn.Module):

    def __init__(self, ln_out=7):
        super(Net, self).__init__()
        batch_size = 2

        self.ln_in = 300
        self.ln_out = ln_out
        self.DEBUG = 0
        if self.DEBUG == 1:
            self.bias = False
        else:
            self.bias = True

        # 第一层
        self.conv1 = nn.Conv2d(1, 5, (1, 10), stride=(2, 2), padding=(0, 5), bias=self.bias)
        self.bn1 = nn.BatchNorm2d(5)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(5, 5, (1, 10), stride=(1, 1), padding=(0, 4), bias=self.bias)
        self.bn2 = nn.BatchNorm2d(5)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(5, 5, (1, 10), stride=(1, 1), padding=(0, 5), bias=self.bias)

        self.bn3 = nn.BatchNorm2d(5)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(5, 5, (1, 10), stride=(1, 1), bias=self.bias)
        self.bn4 = nn.BatchNorm2d(5)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(5, 5, (1, 10), stride=(1, 1), padding=(0, 9), bias=self.bias)
        self.bn5 = nn.BatchNorm2d(5)
        self.relu5 = nn.ReLU(inplace=True)

        # 第二层
        self.conv6 = nn.Conv2d(5, 10, (1, 1), stride=(2, 2), bias=self.bias)

        self.conv7 = nn.Conv2d(5, 10, (1, 10), stride=(2, 2), padding=(0, 1), bias=self.bias)
        self.bn7 = nn.BatchNorm2d(10)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(10, 10, (1, 10), stride=(1, 1), padding=(0, 6), bias=self.bias)
        self.bn8 = nn.BatchNorm2d(10)
        self.relu8 = nn.ReLU(inplace=True)

        self.conv9 = nn.Conv2d(10, 10, (1, 10), stride=(1, 1), padding=(0, 3), bias=self.bias)
        self.bn9 = nn.BatchNorm2d(10)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(10, 10, (1, 10), stride=(1, 1), padding=(0, 6), bias=self.bias)
        self.bn10 = nn.BatchNorm2d(10)
        self.relu10 = nn.ReLU(inplace=True)

        # 第三层
        self.conv11 = nn.Conv2d(10, 15, (1, 1), stride=(2, 2), bias=self.bias)

        self.conv12 = nn.Conv2d(10, 15, (1, 10), stride=(2, 2), padding=(0, 1), bias=self.bias)
        self.bn12 = nn.BatchNorm2d(15)
        self.relu12 = nn.ReLU(inplace=True)

        self.conv13 = nn.Conv2d(15, 15, (1, 10), stride=(1, 1), padding=(0, 6), bias=self.bias)
        self.bn13 = nn.BatchNorm2d(15)
        self.relu13 = nn.ReLU(inplace=True)

        self.conv14 = nn.Conv2d(15, 15, (1, 10), stride=(1, 1), padding=(0, 3), bias=self.bias)
        self.bn14 = nn.BatchNorm2d(15)
        self.relu14 = nn.ReLU(inplace=True)
        self.conv15 = nn.Conv2d(15, 15, (1, 10), stride=(1, 1), padding=(0, 6), bias=self.bias)
        self.bn15 = nn.BatchNorm2d(15)
        self.relu15 = nn.ReLU(inplace=True)

        # 第四层
        self.conv16 = nn.Conv2d(15, self.ln_out, (1, 1), stride=(2, 2), bias=self.bias)

        self.conv17 = nn.Conv2d(15, self.ln_out, (1, 10), stride=(2, 2), padding=(0, 1), bias=self.bias)
        self.bn17 = nn.BatchNorm2d(self.ln_out)
        self.relu17 = nn.ReLU(inplace=True)
        self.conv18 = nn.Conv2d(self.ln_out, self.ln_out, (1, 10), stride=(1, 1), padding=(0, 6), bias=self.bias)
        self.bn18 = nn.BatchNorm2d(self.ln_out)
        self.relu18 = nn.ReLU(inplace=True)

        self.conv19 = nn.Conv2d(self.ln_out, self.ln_out, (1, 10), stride=(1, 1), padding=(0, 2), bias=self.bias)
        self.bn19 = nn.BatchNorm2d(self.ln_out)
        self.relu19 = nn.ReLU(inplace=True)
        self.conv20 = nn.Conv2d(self.ln_out, self.ln_out, (1, 10), stride=(1, 1), padding=(0, 7), bias=self.bias)
        self.bn20 = nn.BatchNorm2d(self.ln_out)
        self.relu20 = nn.ReLU(inplace=True)

        self.bn_test = nn.BatchNorm2d(num_features=batch_size, eps=0, affine=False, track_running_stats=False)

        # 之后加上lstm+maxpooling+dropout+fc+relu
        """
        lstm input: (batch, 250 * 7 = 1750) 
        lstm out  : (batch, 7) 
        """
        # self.lstm = RNN(input_size=250 * self.ln_out)
        # 这个lstm_new 好像没有用到
        # self.lstm_new = torch.nn.LSTM(input_size=250 * self.ln_out, hidden_size=self.ln_out, batch_first=True)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=250, num_heads=5, batch_first=True) # heads必须要能被embed_dim整除
        self.layernorm = nn.LayerNorm(normalized_shape=250, eps=0, elementwise_affine=False) # LayerNorm最后一个维度，也就是embed_dim
        self.feed_forward = PositionalWiseFeedForward(d_model=250, d_hidden=512, d_out=1)
        """
        embed_dim == input_dim ，即query，key，value的embedding_size必须等于embed_dim, 是输入的embedding size，即query输入形状(L, N, E)的E数值，
        embed_dim是每一个单词本来的词向量长度；num_heads是我们MultiheadAttention的head的数量。
        """

        self.maxpooling = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.4)
        self.drop2 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(196, ln_out, bias=False)
        self.fc2 = nn.Linear(64, self.ln_out, bias=self.bias)
        self.fc1_relu = nn.ReLU(inplace=True)
        self.fc2_relu = nn.ReLU(inplace=True)
        self.fc_extra = nn.Linear(self.ln_out, self.ln_out)
        self.relu_extra = nn.ReLU(inplace=True)



    def forward(self, x, valid_len, batch_size):
        """
        1. example of the params: x: [9, 1, 21, 1999], valid_len: [4, 5], batch_size: 2
        2. to split the frequencies, x -> [9:(batch), 7:(freq), 3:(embedding/feature), 1999(time)]
        """

        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        identity1 = x

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = x + identity1
        x = self.bn3(x)
        x = self.relu3(x)

        identity2 = x

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = identity2 + x
        x = self.bn5(x)
        x = self.relu5(x)

        identity3 = self.conv6(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)
        x = x + identity3
        x = self.bn8(x)
        x = self.relu8(x)

        a = x

        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu9(x)
        x = self.conv10(x)
        x = a + x
        x = self.bn10(x)
        x = self.relu10(x)

        identity4 = self.conv11(x)

        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu12(x)
        x = self.conv13(x)
        x = identity4 + x
        x = self.bn14(x)
        x = self.relu14(x)

        x = self.conv14(x)
        x = self.conv15(x)

        identity5 = self.conv16(x)

        x = self.conv17(x)
        x = self.bn17(x)
        x = self.relu17(x)
        x = self.conv18(x)
        x = identity5 + x
        x = self.bn18(x)
        x = self.relu18(x)

        b = x
        x = self.conv19(x)
        x = self.bn19(x)
        x = self.relu19(x)
        x = self.conv20(x)
        x = b + x
        x = self.bn20(x)
        x = self.relu20(x)

        feature_list = []
        """提取CNN Feature"""
        for i in range(batch_size):
            if i > 0:
                start_num = 0
                for j in range(0, i - 1):
                    start_num += valid_len[j]
            else:
                start_num = 0
            # print("feature shape:",x[start_num:start_num+valid_len[i],:,:,:].shape)
            # print("valid len:",valid_len[i])
            cnn_feature = x[start_num:start_num + valid_len[i], :, :, :].view(valid_len[i], -1)

            feature_list.append(cnn_feature)

        cnn_concat_feature = torch.cat(feature_list, dim=0) # Example: (9, 1750)
        packed_transformer_input = cnn_concat_feature.view(cnn_concat_feature.size(0), 7, -1)  # Example: [9, 7, 250]


        """开始attention 部分"""
        # Example: valid_len = [4, 5]
        # count_len = [0, 4, 9] 切分的索引,表示第一段语音[0:4], 第二段[4:9]，
        count_len = cal_count_len(valid_len)
        attention_feature = []

        for index in range(len(count_len) - 1):
            temp_feature = packed_transformer_input[count_len[index]: count_len[index + 1]].transpose(1, 0)  # [4, 7, 250] -> [7, 4, 250] 前两维度互换
            attention_mask = torch.zeros([temp_feature.shape[1], temp_feature.shape[1]]).bool()  # [4, 4]

            attention_mask = attention_mask.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            attn_output, attn_output_weights = self.multihead_attn(query=temp_feature, key=temp_feature, value=temp_feature, attn_mask=attention_mask)
            attn_output = attn_output.transpose(1, 0) # [4, 7, 250]
            attention_feature.append(attn_output)

        attention_feature = torch.cat(attention_feature, dim=0)
        attention_feature = self.layernorm(attention_feature + packed_transformer_input)  # Residual+LayerNorm

        x = self.feed_forward(attention_feature) # batch, 7

        """attention 结束"""

        x = self.relu_extra(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        # input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        # input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)
        # q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  # tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out = out.view(*input.shape)

        return self.gamma * out + input


# Feed-Forward
class PositionalWiseFeedForward(nn.Module):

    def __init__(self, d_model, d_hidden, d_out, dropout=0.1):
        super(PositionalWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.w_2 = nn.Linear(d_hidden, d_out)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x)))).squeeze()

if __name__ == '__main__':
    net = Net()
