import torch
#from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence,pad_sequence

import torch.nn as nn
class Net(nn.Module):

    def __init__(self,ln_out=20):
        super(Net, self).__init__()
        batch_size=2

        ## TODO: Define all the layers of this CNN, the only requirements are:

        self.ln_in = 300
        self.ln_out = ln_out
        self.DEBUG = 0
        if self.DEBUG == 1:
            self.bias = False
        else:
            self.bias = True

        self.conv1 = nn.Conv2d(1, 5, (1, 10), stride=(1, 2),bias=self.bias)
        self.bn1 = nn.BatchNorm2d(5)
        #self.ln1 =
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(5, 5, (1, 10), stride=(1, 2),bias = self.bias)
        self.bn2 = nn.BatchNorm2d(5)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(5, 5, (1, 11), stride=(1, 2),bias = self.bias)
        self.bn3 = nn.BatchNorm2d(5)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(5, 5, (1, 11), stride=(1, 2),bias = self.bias)
        self.bn4 = nn.BatchNorm2d(5)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(5, 5, (3, 8), stride=(2, 2),bias = self.bias)
        self.bn5 = nn.BatchNorm2d(5)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(5, 5, (4, 7), stride=(2, 1),bias = self.bias)
        self.bn6 = nn.BatchNorm2d(5)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(5,1,(1,1),stride=(1,1),bias = self.bias)
        self.bn7 = nn.BatchNorm2d(1)
        self.relu7 = nn.ReLU(inplace=True)

        self.bn_test = nn.BatchNorm2d(num_features=batch_size, eps=0, affine=False, track_running_stats=False)

        #之后加上lstm+maxpooling+dropout+fc+relu
        self.lstm = RNN(input_size=980)
        self.lstm_new = torch.nn.LSTM(input_size=980, hidden_size=20, batch_first=True)


        self.maxpooling = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.4)
        self.drop2 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(196,ln_out,bias = False)
        self.fc2 = nn.Linear(64,self.ln_out,bias = self.bias)
        self.fc1_relu =  nn.ReLU(inplace=True)
        self.fc2_relu = nn.ReLU(inplace=True)
        self.fc_extra = nn.Linear(20,20)
        self.relu_extra = nn.ReLU(inplace=True)




    def forward(self, x,h_n,h_c,valid_len,batch_size,max_image_num):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        feature_list = []
        for i in range(batch_size):
            if i>0:
                start_num = 0
                for j in range(0,i-1):
                    start_num+=valid_len[j]
            else:
                start_num = 0
            # print("feature shape:",x[start_num:start_num+valid_len[i],:,:,:].shape)
            # print("valid len:",valid_len[i])
            cnn_feature  = x[start_num:start_num+valid_len[i],:,:,:].view(valid_len[i],-1)

            feature_list.append(cnn_feature)

        cnn_concat_feature = torch.cat(feature_list,dim=0)
        #batch,time,channel*freq
        pad_cnn_sequence = pad_sequence(feature_list, batch_first=True)
        pad_cnn_sequence_bn = self.bn_test(pad_cnn_sequence.unsqueeze(0))

        pad_cnn_sequence = torch.squeeze(pad_cnn_sequence_bn,0)
        packed_lstm_input = pack_padded_sequence(input=pad_cnn_sequence,enforce_sorted=False, lengths=valid_len, batch_first=True)
        packed_out, _,_ = self.lstm(packed_lstm_input,h_n,h_c)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        rnn_feature_list = []
        for i in range(batch_size):
            rnn_feature_list.append(out[i,0:valid_len[i],:])
        rnn_feature = torch.cat(rnn_feature_list,dim=0)
        #x = self.conv7(x)
        #x = self.bn7(x)
        # x = self.relu7(x)



        #or lstm,x's dimension is 3
        #so squeeze dim 1
        # b,c,h,w = x.shape
       # x = x.reshape(x.shape[0],98,10)
        #self.lstm.to(device=x.device)
        # x, h_n = self.lstm(x)


      #  cnn_features = []

        # for i in range(batch_size):
        #     split_cnn_feature = x[max_image_num * i:max_image_num * i + valid_len[i]]
        #     cnn_features.append(torch.squeeze(split_cnn_feature,dim=1).view(-1,split_cnn_feature.shape[-1]*split_cnn_feature.shape[-2]))
        #
        # valid_len = torch.tensor(valid_len)
        # pad_cnn_sequence = pad_sequence(cnn_features,batch_first = True)
        # packed_lstm_input = pack_padded_sequence(input=pad_cnn_sequence,enforce_sorted=False, lengths=valid_len, batch_first=True)
        # packed_out, _ = self.lstm_new(packed_lstm_input)
        # out, _ = pad_packed_sequence(packed_out, batch_first=True)
        #


        #x = self.bn7(x)
        #x = x.view(x.shape[0],-1)
        x = self.fc_extra(rnn_feature)
        x = self.relu_extra(x)



        # x = self.drop1(out)
        # x = self.fc1(x)
        # x = self.fc1_relu(x)

        # x = self.drop2(x)
        #
        # x = self.fc2(x)
        # x = self.fc2_relu(x)


        return x

class RNN(nn.Module):
    def __init__(self,input_size=4):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=20,
            num_layers=1,
            batch_first=True
        )
        # num_layers = 1,
        # batch_first = True


    def forward(self, x,h_n,h_c):
        r_out, (h_n, h_c) = self.rnn(x)  # None 表示 hidden state 会用全0的 state
        return r_out,h_n.clone(),h_c.clone()
        # r_out, (h_n, h_c) = self.rnn(x, (h_n, h_c))  # None 表示 hidden state 会用全0的 state
        # return r_out,h_n,h_c
