from torch import nn
import json
from models.transformer import TrajPredTransformer, TrajPredTransformerV1
from utils.config import read_config, print_config, load_to_dict_config
from data.load_data import load_ETCEn_data, load_ETCEn_dataV1
from utils.metrics import Accuracy, time_station_Accuracy, TimeAccuracy, mae_loss, unnormalize
import torch, os, tqdm, numpy as np, random
from torch.utils.tensorboard import SummaryWriter


class TrajLSTM(nn.Module):
    def __init__(self, conf):
        super(TrajLSTM, self).__init__()

        self.vocab_size = int(conf['vocab_size'])
        self.inp_size = int(conf['inp_size'])
        hidden_size = int(conf['hidden_size'])
        out_size = int(conf['out_size'])
        num_layers = int(conf['num_layers'])
        
        self.lstm1 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.reg = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, out_size),
        )  # regression
        
        self.classfier =  nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.vocab_size),
        )  # classfier
        
        self.stations_embeding = nn.Embedding(self.vocab_size, hidden_size)
        self.intervals_embeding = self.liner = nn.Linear(1, hidden_size)
        
        
        self.crossEntropy = nn.CrossEntropyLoss()
        
    
    def des_loss(self, output, target):
        target = target.contiguous().reshape(-1)
        output = output.reshape(-1, self.vocab_size)
        mask = (target != 0).to(target.device)
        target = target[mask]
        output = output[mask]
        loss = self.crossEntropy(output.reshape(-1, self.vocab_size), target)
        return loss
    
    def mae_loss(self, predicted, observed, null_val=0.0):
        mask = (observed != null_val)
        mask = mask.float()
        mask /=  torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(predicted - observed)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

    def loss(self, output, target, intervals_output, intervals_target):
        desloss = self.des_loss(output, target)
        intervalsloss = self.mae_loss(intervals_output, intervals_target)

        return 0.5 * intervalsloss + 0.5 * desloss, desloss, intervalsloss
    
    def forward(self, src_staions, src_intervals):
        src_staions = self.stations_embeding(src_staions)
        src_intervals = self.intervals_embeding(src_intervals)
        tgt_stations = self.lstm1(src_staions)[0]  # y, (h, c) = self.lstm(x)
        batch_size, seq_len, hid_dim = tgt_stations.shape
        tgt_stations = tgt_stations.reshape(-1, hid_dim)
        tgt_stations = self.classfier(tgt_stations)
        tgt_stations = tgt_stations.reshape(batch_size, seq_len, -1)
        tgt_stations = tgt_stations[:,-1,:]
                      
        tgt_intervals = self.lstm2(src_intervals)[0]  # y, (h, c) = self.lstm(x)
        batch_size, seq_len, hid_dim = tgt_intervals.shape
        tgt_intervals = tgt_intervals.reshape(-1, hid_dim)
        tgt_intervals = self.reg(tgt_intervals)
        tgt_intervals = tgt_intervals.reshape(batch_size, seq_len, -1)
        tgt_intervals = tgt_intervals[:,-1,:]
        return tgt_stations, tgt_intervals

    """
    PyCharm Crtl+click nn.LSTM() jump to code of PyTorch:
    Examples::
        >>> lstm = nn.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)     # 5个时间步，也就是每个时间序列的长度是5,3表示一共有3个时间序列，10表示每个序列在每个时间步的维度是10
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = lstm(input, (h0, c0))
    """

    def output_y_hc(self, x, hc):

        y, hc = self.lstm(x, hc)  # y, (h, c) = self.lstm(x)

        seq_len, batch_size, hid_dim = y.size()
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y, hc