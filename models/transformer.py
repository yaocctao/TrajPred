import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
import numpy as np
from models.fusion import fusionGate
from visualizer import get_local
from utils.metrics import unnormalize, MinMaxUnnormalize
from utils.standardization import exp_unnormalize
from torch_geometric.nn import GCNConv


# Set device to GPU
device = torch.device('cuda:0')

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output, attn_probs

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output, attn_probs

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @get_local("self_attn")
    def forward(self, x, mask):
        attn_output, self_attn = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    # @get_local("self_attn")
    @get_local("cross_attn")
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output, self_attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output, cross_attn = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
class trajDecoderLayer(nn.Module):
    
    def __init__(self, d_model, num_heads, d_ff, dropout, adj = None):
        super(trajDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        # self.feed_forward2 = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.adj = adj
        # if adj != None:
        #     self.dynamicGAT = MultiHeadAttention(d_model, num_heads)
            # self.fc = FC(d_model, d_model, 1804, dropout)
            # self.gcn = GCN(d_model, d_model, dropout)
    
    @get_local("self_attn")
    def forward(self, x, tgt_mask, station_index=None, station_emb=None, last_output=None):
        output = None
        attn_output, self_attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        # if self.adj != None:
        #     #后面再加一个gat层,考虑图结构信息，见草稿里的图
        #     adj = self.adj[station_index].unsqueeze(1)
        #     dynamicGAT_output, _ = self.dynamicGAT(x, station_emb, station_emb, adj)
        #     x = self.norm3(x + self.dropout(dynamicGAT_output))
        #     ff_output = self.feed_forward(x)
        #     x = self.norm4(x + self.dropout(ff_output))
        #     if last_output:
        #         _output = self.fc(x).permute(0,2,1)
        #         output = self.norm3((_output + self.dropout(last_output)).permute(0,2,1)).permute(0,2,1)
        #         output = self.gcn(output, self.adj)
        #         output = self.norm4((_output + self.dropout(output)).permute(0,2,1)).permute(0,2,1)
        #     else:
        #         _output = self.fc(x).permute(0,2,1)
        #         output = self.gcn(_output, self.adj)
        #         output = self.norm3((_output + self.dropout(output)).permute(0,2,1)).permute(0,2,1)
            
        
        return x, output

class StaionsAttensionLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(StaionsAttensionLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # self.enc_output_self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.feed_forward2 = PositionWiseFeedForward(d_model, d_ff)
        self.sigmoid = nn.Sigmoid()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def make_combined_att_mask(self, attention_mask):
        """Combined attention mask and layer mask.

        Args:
            attention_mask (tensor): mask for attention operation, [batch_size, seq_len, seq_len]

        Returns:
            tensor: [batch_size, seq_len * 2, seq_len * 2]
        """
        # [batch_size, seq_len, seq_len * 2]
        combined_mask = torch.cat([attention_mask, attention_mask], dim=-1)
        # [batch_size, seq_len, seq_len * 2]
        contextual_mask = torch.cat([attention_mask, attention_mask], dim=-1)
        # [batch_size, seq_len * 2, seq_len * 2]
        combined_mask = torch.cat([contextual_mask, combined_mask], dim=-2)
        return combined_mask
    
    # @get_local("self_attn")
    @get_local("cross_attn")
    def forward(self, x, enc_output, intervals, last_intervals_attn_output = None, src_mask = None, cross_mask = None):
        
        attn_output, self_attn = self.self_attn(x, enc_output, enc_output, src_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        # _x = torch.cat((x, intervals), dim=1)
        # _enc_output = torch.cat((enc_output, intervals), dim=1)
        # _intervals = torch.cat((x, intervals), dim=1)
        # _cross_mask = self.make_combined_att_mask(cross_mask)
        
        # intervals_attn_output, cross_attn = self.cross_attn(_x, _enc_output, _intervals, _cross_mask)
        # intervals_attn_output = intervals_attn_output[:, :intervals.size(1), :]
        intervals_attn_output, cross_attn = self.cross_attn(x, enc_output, intervals, cross_mask)
        if last_intervals_attn_output != None:
            intervals_attn_output = self.norm3(last_intervals_attn_output + self.dropout(intervals_attn_output))
            ff_output = self.feed_forward2(intervals_attn_output)
            new_intervals_attn_output = self.norm4(intervals_attn_output + self.dropout(ff_output))
            
        else:
            new_intervals_attn_output = intervals_attn_output
            
        return x, new_intervals_attn_output

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(src.device)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(src.device)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=src.device), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask 
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
    
class IntervalsEncoder(nn.Module):
    def __init__(self, max_len, hidden_size, dropout):
        super(IntervalsEncoder, self).__init__()
        # self.DoW_Emb = nn.Embedding(8, hidden_size)
        # self.HoD_Emb = nn.Embedding(25, hidden_size)
        # self.time_encoder = nn.Conv2d(max_len, hidden_size, (2,1))
        # self.intervals_encoder = nn.Conv1d(1, hidden_size, 1)
        # self.intervals_encoder = nn.Linear(1, hidden_size)
        # self.fc = FC(hidden_size, hidden_size, dropout)
        self.liner = nn.Linear(1, hidden_size)
        
        # self.encoder = nn.Conv2d(2, hidden_size, 1)
        # self.encoder = nn.Linear(2, hidden_size, 1)
        
    def forward(self, HoD, DoW, intervals):
        # DoW = self.DoW_Emb(DoW)
        # HoD = self.HoD_Emb(HoD)
        # time_emb = self.time_encoder(torch.stack((HoD, DoW), dim=-2)).squeeze(-2)
        intervals_emb = self.liner(intervals)
        # intervals_emb = self.fc(intervals)
        
        # output = self.encoder(torch.stack((time_emb, intervals_emb), dim=1)).squeeze(-2)
        # output = time_emb + intervals_emb
        output = intervals_emb
        return output

class Intervals_FC(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(Intervals_FC, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(output_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.Layer_norms1 = nn.LayerNorm(output_size)
        # self.Layer_norms2 = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.Layer_norms1(x + self.dropout(x))
        x = self.relu(x)
        # x = self.Layer_norms2(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class FC(nn.Module):
    def __init__(self, input_size, output_size, vocab_size, dropout):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(output_size, vocab_size)
        self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()
        self.Layer_norms1 = nn.LayerNorm(output_size)
        self.Layer_norms2 = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.Layer_norms1(x + self.dropout(x))
        x = self.sigmoid(x) * x
        x = self.Layer_norms2(x + self.dropout(x))
        x = self.fc2(x)
        # x = self.Layer_norms2(x + self.dropout(x))
        # x = self.relu(x)
        return x
class StationEmb(nn.Module):
    
    def __init__(self, vocab_size, max_len, hidden_size):
        super(StationEmb, self).__init__()
        self.stationEmbedding = nn.Embedding(vocab_size, hidden_size)
        self.positionEmbedding = nn.Embedding(max_len, hidden_size)
        self.DoW_Emb = nn.Embedding(8, hidden_size)
        self.HoD_Emb = nn.Embedding(25, hidden_size)
        self.pos_emb = nn.Parameter(torch.randn(1,max_len,hidden_size))
        self.fusion =  nn.Conv2d(hidden_size, hidden_size, (2,1))
        
    
    def forward(self, stations, position):
        enc_sta_emb =  self.stationEmbedding(stations)
        enc_pos_emb =  self.positionEmbedding(position)
        # enc_dow_emb = self.DoW_Emb(DoW)
        # enc_hod_emb = self.HoD_Emb(HoD)
        
        # fusion_emb = self.fusion(torch.stack([enc_sta_emb, enc_pos_emb], dim=1).permute(0,3,1,2)).permute(0,2,3,1).squeeze(1)
        
        return enc_sta_emb + enc_pos_emb + self.pos_emb.data

   
class GCN(nn.Module):
    def __init__(self, n_features, hidden_dim, dropout):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(n_features, hidden_dim, dropout, bias=True)
        # self.gc2 = GCNConv(hidden_dim, n_classes, bias=False)
        self.relu = nn.ReLU()
        # self.register_buffer('supports', supports)

    def forward(self, inputs, adj):
        x = inputs
        x = self.relu(self.gc1(x, adj))+x
        # x = self.gc2(x, adj)+x
        return x
   
    
class EncoderEmb(nn.Module):
    
    def __init__(self, vocab_size, hidden_size):
        super(EncoderEmb, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.DoW_Emb = nn.Embedding(8, hidden_size)
        self.HoD_Emb = nn.Embedding(25, hidden_size)
        
    def forward(self, enc_src, DoW, HoD):
        enc_src_emb =  self.embedding(enc_src)
        enc_dow_emb = self.DoW_Emb(DoW)
        enc_hod_emb = self.HoD_Emb(HoD)
        
        return enc_src_emb + enc_dow_emb + enc_hod_emb
   
class TrajPredTransformer(nn.Module):
    def __init__(self, conf):
        super(TrajPredTransformer, self).__init__()
        self.vocab_size = int(conf["vocab_size"])
        # self.tgt_intervals_cov = nn.Conv2d(int(conf["max_len"]), int(conf["max_len"]), (2,1))
        self.tgt_intervals_cov = nn.Conv2d(2, 1, (1,1))
        self.decoder_embedding = nn.Embedding(self.vocab_size, int(conf["hidden_size"]))
        self.positional_encoding = PositionalEncoding(int(conf["hidden_size"]), int(conf["max_len"]))
        self.intervalsEncoder = IntervalsEncoder(int(conf["max_len"]), int(conf["hidden_size"]))

        self.decoder_layers = nn.ModuleList([trajDecoderLayer(int(conf["hidden_size"]), int(conf["num_heads"]), int(conf["d_ff"]), float(conf["dropout"])) for _ in range(int(conf["num_layers"]))])
        self.cross_decoder_layers = nn.ModuleList([DecoderLayer(int(conf["hidden_size"]), int(conf["num_heads"]), int(conf["d_ff"]), float(conf["dropout"])) for _ in range(int(conf["num_layers"]))])

        self.fc = nn.Linear(int(conf["hidden_size"]), self.vocab_size)
        self.intervals_fc = nn.Linear(int(conf["hidden_size"]), 1)
        self.dropout = nn.Dropout(float(conf["dropout"]))
        self.crossEntropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        
        # 不确定性权重参数（可训练）
        self.log_sigma1 = nn.Parameter(torch.tensor(1.0))
        self.log_sigma2 = nn.Parameter(torch.tensor(1.0))
        
    def generate_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(3).to(src.device)
        seq_length = src.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=src.device), diagonal=1)).bool()
        src_mask = src_mask & nopeak_mask
        return  src_mask
    
    def random_mask():
        pass
    
    def loss(self, output, target, intervals_output, intervals_target):
        desloss = self.des_loss(output, target)
        # intervalsloss = 0.5*self.intervals_loss(intervals_output, intervals_target) + 0.5*self.intervals_loss(src_intervals_output, src_intervals_tgt)
        intervalsloss = self.intervals_loss(intervals_output, intervals_target)
        print("self.log_sigma1",self.log_sigma1)
        print("self.log_sigma2",self.log_sigma2)
        return (1/(2*self.log_sigma1**2))*desloss + (1/(2*self.log_sigma2**2))*intervalsloss / 60 + torch.log(self.log_sigma1*self.log_sigma2)
    
    def des_loss(self, output, target):
        target = target.contiguous().reshape(-1)
        output = output.reshape(-1, self.vocab_size)
        mask = (target != 0).to(target.device)
        target = target[mask]
        output = output[mask]
        loss = self.crossEntropy(output.reshape(-1, self.vocab_size), target)
        return loss
    
    def intervals_loss(self, intervals_output, target):
        mask = (target != 0).to(target.device)
        intervals_output = intervals_output[mask]
        target = target[mask]
        length = len(target)
        loss = self.mse(intervals_output, target)/length
        return loss
    
    def DesTimeLoss(self, output, target, intervals_output, intervals_target):
        mask = (target != 0).to(target.device)
        intervals_mask = (intervals_target != 0).to(intervals_target.device)
        output = torch.argmax(output, dim=-1)
        output = output[mask]
        target = target[mask]
        station_corrects = (output == target).flatten()
        total = len(target)
        
        time_corrects = (torch.abs(intervals_output - intervals_target) < 60)
        time_corrects = time_corrects[intervals_mask]
        
        #station_corrects和time_corrects都为True的有多少个
        corrects = (station_corrects & time_corrects).sum().float()
        
        return - torch.log((corrects + 1)/(total + 1))
    
    def mae_loss(self, predicted, observed, null_val=0.0):
        mask = (observed != null_val)
        mask = mask.float()
        mask /=  torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(predicted - observed)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

    def forward(self, src, HoD, DoW, intervals, tgt = None):
        intervals_emb = self.intervalsEncoder(HoD, DoW, intervals)
        src_mask = self.generate_mask(src)
        intervals_mask = self.generate_mask(intervals.squeeze(-1))
        src_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(src)))
        
        if tgt is not None:
            tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
            dec_output = self.tgt_intervals_cov(torch.stack((src_embedded, tgt_embedded), dim=1)).squeeze(1)
            src_intervals_output = F.pad(self.intervals_fc(dec_output)[:,1:], (0,0,1,0), value=0)
        
        dec_output = src_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, src_mask)
            
        output = self.fc(dec_output)
        
        intervals_dec_output = intervals_emb
        for cross_dec_layer in self.cross_decoder_layers:
            intervals_dec_output = cross_dec_layer(intervals_dec_output, src_embedded, intervals_mask, intervals_mask)
        
        dec_output = self.tgt_intervals_cov(torch.stack((src_embedded, dec_output), dim=1)).squeeze(1)
        intervals_dec_output = fusionGate(intervals_dec_output, dec_output)

        intervals_output = self.intervals_fc(intervals_dec_output)

        if tgt is not None:
            return output, src_intervals_output, intervals_output
        
        return output, intervals_output
    
class TrajPredTransformerV1(nn.Module):
    def __init__(self, conf):
        super(TrajPredTransformerV1, self).__init__()
        self.num_layers = int(conf["num_layers"])
        self.vocab_size = int(conf["vocab_size"])
        # self.tgt_intervals_cov = nn.Conv2d(int(conf["max_len"]), int(conf["max_len"]), (2,1))
        self.station_cov = nn.Conv1d(int(conf["hidden_size"]), int(conf["hidden_size"]), 2)
        self.enc_station_cov = nn.Conv1d(int(conf["hidden_size"]), int(conf["hidden_size"]), 2)
        # self.tgt_intervals_cov = nn.Conv2d(2, 1, (1,1))
        self.tgt_intervals_cov = nn.Conv2d(int(conf["hidden_size"]), int(conf["hidden_size"]), (2,1))
        self.tgt_intervals_cov1 = nn.Conv2d(int(conf["hidden_size"]), int(conf["hidden_size"]), (2,1))
        self.decoder_embedding = nn.Embedding(self.vocab_size, int(conf["hidden_size"]))
        # self.decoder_embedding = StationEmb(self.vocab_size, int(conf["max_len"]), int(conf["hidden_size"]))
        self.encoder_embedding = EncoderEmb(self.vocab_size, int(conf["hidden_size"]))
        self.positional_encoding = PositionalEncoding(int(conf["hidden_size"]), int(conf["max_len"]))
        self.start_end_embedding = nn.Embedding(2, int(conf["hidden_size"]))
        self.intervalsEncoder = IntervalsEncoder(int(conf["max_len"]), int(conf["hidden_size"]), float(conf["dropout"]))
        self.enc_intervalsEncoder = IntervalsEncoder(int(conf["max_len"]), int(conf["hidden_size"]), float(conf["dropout"]))

        self.encoder_layers = nn.ModuleList([EncoderLayer(int(conf["hidden_size"]), int(conf["num_heads"]), int(conf["d_ff"]), float(conf["dropout"])) for _ in range(self.num_layers)])
        # self.adj = self.get_supports(conf['adj_path'])
        self.adj = self.adjecent(conf['adj_path'])
        #放入其他网络相同的设备
        self.adj = self.adj.to(device)
        self.decoder_layers = nn.ModuleList([trajDecoderLayer(int(conf["hidden_size"]), int(conf["num_heads"]), int(conf["d_ff"]), float(conf["dropout"])) for _ in range(self.num_layers)])
        self.staion_attn_layers = nn.ModuleList([trajDecoderLayer(int(conf["hidden_size"]), int(conf["num_heads"]), int(conf["d_ff"]), float(conf["dropout"])) for _ in range(self.num_layers)])
        self.cross_decoder_layers = nn.ModuleList([StaionsAttensionLayer(int(conf["hidden_size"]), int(conf["num_heads"]), int(conf["d_ff"]), float(conf["dropout"])) for _ in range(self.num_layers)])

        # self.fc = nn.Linear(int(conf["hidden_size"]), self.vocab_size)
        self.fc = FC(int(conf["hidden_size"]), int(conf["hidden_size"]), self.vocab_size, float(conf["dropout"]))
        # self.intervals_fc = nn.Linear(int(conf["hidden_size"]), 1)
        self.intervals_fc = Intervals_FC(int(conf["hidden_size"]), int(conf["hidden_size"]), float(conf["dropout"]))
        self.dropout = nn.Dropout(float(conf["dropout"]))
        self.crossEntropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        
        # 不确定性权重参数（可训练）
        self.log_sigma1 = nn.Parameter(torch.tensor(1.0))
        self.log_sigma2 = nn.Parameter(torch.tensor(1.0))
        
        
    def adjecent(self, file_adj):
        '''
        :return: adj matrix
        '''
        data = pd.read_csv(filepath_or_buffer=file_adj)
        adj = np.zeros(shape=[self.vocab_size, self.vocab_size], dtype=np.int32)
        for line in data[['from_id', 'to_id']].values:
            adj[int(line[0])][int(line[1])] = 1
        #adj 标准化
        # adj = self.normalization(adj)
        adj = adj + np.eye(self.vocab_size)
        return torch.from_numpy(adj)
    
    def get_supports(self, file_adj):
        data = pd.read_csv(filepath_or_buffer=file_adj)
        edge_index = [[line[0],line[1]]for line in data[['from_id', 'to_id']].values]
        edge_index.extend([[i,i]for i in range(self.vocab_size)])
        edge_index = torch.tensor(edge_index,dtype=torch.long)
        return edge_index.t().contiguous()

    def generate_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(3).to(src.device)
        seq_length = src.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=src.device), diagonal=1)).bool()
        src_mask = src_mask & nopeak_mask
        return  src_mask
    
    def enc_src_mask(self, src):
        return (src != 0).unsqueeze(1).unsqueeze(2).to(src.device)
    
    def station_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(3).to(src.device)
        seq_length = src.size(1)
        nopeak_mask = (torch.tril(torch.ones(1, seq_length, seq_length, device=src.device), diagonal=-1)).bool()
        src_mask = src_mask & nopeak_mask
        return  src_mask
        
    
    def loss(self, output, target, intervals_output, intervals_target, mean, std):
        desloss = self.des_loss(output, target)
        mask = (intervals_target != 0.0).to(intervals_target.device)
        intervals_output = intervals_output[mask]
        intervals_target = intervals_target[mask]
        # intervals_output = unnormalize(intervals_output, mean, std)
        # intervals_target = exp_unnormalize(intervals_target, torch.tensor(0.011146042964972321).cuda(), mean, std)
        # intervals_target = unnormalize(intervals_target, mean, std)
        # intervals_output = exp_unnormalize(intervals_output, torch.tensor(0.011146042964972321).cuda(), mean, std)
        intervalsloss = self.mae_loss(intervals_output, intervals_target)
        # return (1/(2*self.log_sigma1**2))*desloss + (1/(2*self.log_sigma2**2))*intervalsloss / 60 + torch.log(self.log_sigma1*self.log_sigma2)
        return 0.5 * intervalsloss + 0.5 * desloss, desloss, intervalsloss
    
    def des_loss(self, output, target):
        target = target.contiguous().reshape(-1)
        output = output.reshape(-1, self.vocab_size)
        mask = (target != 0).to(target.device)
        target = target[mask]
        output = output[mask]
        loss = self.crossEntropy(output.reshape(-1, self.vocab_size), target)
        return loss
    
    def intervals_loss(self, intervals_output, target):
        mask = (target != 0).to(target.device)
        intervals_output = intervals_output[mask]
        target = target[mask]
        loss = self.mse(intervals_output, target)
        return loss
    
    def DesTimeLoss(self, output, target, intervals_output, intervals_target):
        mask = (target != 0).to(target.device)
        intervals_mask = (intervals_target != 0).to(intervals_target.device)
        output = torch.argmax(output, dim=-1)
        output = output[mask]
        target = target[mask]
        station_corrects = (output == target).flatten()
        total = len(target)
        
        time_corrects = (torch.abs(intervals_output - intervals_target) < 60)
        time_corrects = time_corrects[intervals_mask]
        
        #station_corrects和time_corrects都为True的有多少个
        corrects = (station_corrects & time_corrects).sum().float()
        
        return - torch.log((corrects + 1)/(total + 1))
    
    def mae_loss(self, predicted, observed, null_val=0.0):
        mask = (observed != null_val)
        mask = mask.float()
        mask /=  torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(predicted - observed)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

    def forward(self, enc_src, enc_intervals, src, enc_HoD, enc_DoW, intervals, travel_index, tgt = None):
        
        intervals_emb = self.intervalsEncoder(enc_HoD, enc_DoW, intervals)
        # enc_intervals_emb = self.enc_intervalsEncoder(enc_HoD, enc_DoW, enc_intervals)
        # enc_src_mask = self.enc_src_mask(enc_src)
        src_mask = self.generate_mask(src)
        station_mask = self.station_mask(src)
        # intervals_mask = self.generate_mask(intervals.squeeze(-1))
        # enc_src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(enc_src, enc_DoW, enc_HoD)))
        src_embedded = self.decoder_embedding(src)
        # src_embedded = self.decoder_embedding(src)
        
        # enc_src_embedded = self.enc_station_cov(F.pad(enc_src_embedded, (0,0,1,0), value=0).permute(0,2,1)).permute(0,2,1)
        # enc_output = enc_src_embedded + enc_intervals_emb
        # for enc_layer in self.encoder_layers:
        #     enc_output = enc_layer(enc_output, enc_src_mask)
            
        # src_embedded[:, 0, :] = enc_output[:, 0, :]

        
        dec_output = src_embedded
        stationEmbedding = self.decoder_embedding.weight.reshape(1, self.vocab_size, -1)
        for i, dec_layer in enumerate(self.decoder_layers):
            if i == 0:
                # dec_output, output = dec_layer(dec_output, src_mask, src, stationEmbedding, None)
                dec_output, output = dec_layer(dec_output, src_mask, src, stationEmbedding, None)
            else:
                # dec_output, output = dec_layer(dec_output, src_mask, src, None, output)
                dec_output, output = dec_layer(dec_output, src_mask, src, stationEmbedding, output)
    
        # output = output.permute(0,2,1)
        output = self.fc(dec_output)
        # if self.adj is not None:
        #     adj = self.adj[src]
        #     output = output.masked_fill(adj == 0, 0.0)
        
        if tgt is None:
            tgt = torch.argmax(output, -1) 
        dec_output = self.decoder_embedding(tgt)
        # dec_output = self.decoder_embedding(tgt)
        # src_embedded = self.decoder_embedding(src)
        
        start_src_embedded = src_embedded + self.start_end_embedding(torch.zeros(src_embedded.shape[0],src_embedded.shape[1], dtype=torch.int32).to(device))
        end_src_embedded = F.pad(src_embedded[:,1:,:], (0,0,0,1), value=0) + self.start_end_embedding(torch.ones(src_embedded.shape[0],src_embedded.shape[1], dtype=torch.int32).to(device))
        end_dec_output = dec_output + self.start_end_embedding(torch.ones(dec_output.shape[0],dec_output.shape[1], dtype=torch.int32).to(device))

        staion_cov_emb = self.tgt_intervals_cov1(torch.stack((start_src_embedded, end_src_embedded), dim=1).permute(0,3,1,2)).permute(0,2,3,1).squeeze(1)
        tgt_sections_output = self.tgt_intervals_cov1(torch.stack((start_src_embedded, end_dec_output), dim=1).permute(0,3,1,2)).permute(0,2,3,1).squeeze(1)
        
        
        for i, cross_dec_layer in enumerate(self.cross_decoder_layers):
            if i == 0:
                tgt_sections_output, intervals_attn_output = cross_dec_layer(tgt_sections_output, staion_cov_emb, intervals_emb, None, station_mask, station_mask)
            else:
                tgt_sections_output, intervals_attn_output = cross_dec_layer(tgt_sections_output, staion_cov_emb, intervals_emb, intervals_attn_output, station_mask, station_mask)
        
        
        # intervals_dec_output = fusionGate(tgt_sections_output, dec_output)
        intervals_dec_output = intervals_attn_output
        intervals_output = self.intervals_fc(intervals_dec_output)

        return output, intervals_output

class TrajPredTransformerV2(nn.Module):
    def __init__(self, conf):
        super(TrajPredTransformerV2, self).__init__()
        self.num_layers = int(conf["num_layers"])
        self.vocab_size = int(conf["vocab_size"])
        # self.tgt_intervals_cov = nn.Conv2d(int(conf["max_len"]), int(conf["max_len"]), (2,1))
        self.station_cov = nn.Conv1d(int(conf["hidden_size"]), int(conf["hidden_size"]), 2)
        self.enc_station_cov = nn.Conv1d(int(conf["hidden_size"]), int(conf["hidden_size"]), 2)
        # self.tgt_intervals_cov = nn.Conv2d(2, 1, (1,1))
        self.tgt_intervals_cov = nn.Conv2d(int(conf["hidden_size"]), int(conf["hidden_size"]), (2,1))
        self.tgt_intervals_cov1 = nn.Conv2d(int(conf["hidden_size"]), int(conf["hidden_size"]), (2,1))
        self.station_embedding = nn.Embedding(self.vocab_size, int(conf["hidden_size"]))
        self.encoder_embedding = EncoderEmb(self.vocab_size, int(conf["hidden_size"]))
        self.positional_encoding = PositionalEncoding(int(conf["hidden_size"]), int(conf["max_len"]))
        self.start_end_embedding = nn.Embedding(2, int(conf["hidden_size"]))
        self.intervalsEncoder = IntervalsEncoder(int(conf["max_len"]), int(conf["hidden_size"]), float(conf["dropout"]))
        self.enc_intervalsEncoder = IntervalsEncoder(int(conf["max_len"]), int(conf["hidden_size"]), float(conf["dropout"]))
        self.fc2 = FC(int(conf["hidden_size"]), int(conf["hidden_size"]), float(conf["dropout"]))
        self.fc3 = FC(int(conf["hidden_size"]), int(conf["hidden_size"]), float(conf["dropout"]))

        self.encoder_layers = nn.ModuleList([EncoderLayer(int(conf["hidden_size"]), int(conf["num_heads"]), int(conf["d_ff"]), float(conf["dropout"])) for _ in range(self.num_layers)])
        self.trajEncoderLayers = nn.ModuleList([trajDecoderLayer(int(conf["hidden_size"]), int(conf["num_heads"]), int(conf["d_ff"]), float(conf["dropout"])) for _ in range(self.num_layers)])
        self.staion_attn_layers = nn.ModuleList([trajDecoderLayer(int(conf["hidden_size"]), int(conf["num_heads"]), int(conf["d_ff"]), float(conf["dropout"])) for _ in range(self.num_layers)])
        self.cross_decoder_layers = nn.ModuleList([StaionsAttensionLayer(int(conf["hidden_size"]), int(conf["num_heads"]), int(conf["d_ff"]), float(conf["dropout"])) for _ in range(self.num_layers)])

        self.fc = nn.Linear(int(conf["hidden_size"]), self.vocab_size)
        self.intervals_fc = nn.Linear(int(conf["hidden_size"]), 1)
        # self.intervals_fc = Intervals_FC(int(conf["hidden_size"]), int(conf["hidden_size"]), 1)
        self.dropout = nn.Dropout(float(conf["dropout"]))
        self.crossEntropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        
        # 不确定性权重参数（可训练）
        self.log_sigma1 = nn.Parameter(torch.tensor(1.0))
        self.log_sigma2 = nn.Parameter(torch.tensor(1.0))

    def generate_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(3).to(src.device)
        seq_length = src.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=src.device), diagonal=1)).bool()
        src_mask = src_mask & nopeak_mask
        return  src_mask
    
    def enc_src_mask(self, src):
        return (src != 0).unsqueeze(1).unsqueeze(2).to(src.device)
    
    def station_mask(self, src, travel_mask):
        travel_mask = travel_mask.bool()
        src_mask = (src != 0).unsqueeze(1).unsqueeze(3).to(src.device)
        seq_length = src.size(1)
        
        nopeak_mask = (torch.tril(torch.ones(1, seq_length, seq_length, device=src.device), diagonal=-1)).bool()

        true_counts = travel_mask.sum(dim=1)
        expanded_matrix = travel_mask.repeat_interleave(true_counts, dim=0)

        travel_mask_temp = travel_mask.unsqueeze(-1) | ~travel_mask.unsqueeze(-2).expand(-1, seq_length, -1)
        travel_mask_temp[travel_mask] = expanded_matrix
        
        src_mask = src_mask & nopeak_mask
        src_mask = src_mask & travel_mask_temp.unsqueeze(1)
        return  src_mask
        
    
    def loss(self, output, target, intervals_output, intervals_target, mean ,std):
        desloss = self.des_loss(output, target)
        mask = (intervals_target != 0.0).to(intervals_target.device)
        intervals_output = intervals_output[mask]
        intervals_target = intervals_target[mask]
        # intervals_target = unnormalize(intervals_target, mean, std)
        intervals_target = exp_unnormalize(intervals_target, 0.011146042964972321, mean, std)
        # intervals_target = MinMaxUnnormalize(intervals_target, mean, std)
        # intervals_output = unnormalize(intervals_output, mean, std)
        intervals_output = exp_unnormalize(intervals_output, 0.011146042964972321, mean, std)
        # intervals_output = MinMaxUnnormalize(intervals_output, mean, std)
        intervalsloss = self.mae_loss(intervals_output, intervals_target)
        # return (1/(2*self.log_sigma1**2))*desloss + (1/(2*self.log_sigma2**2))*intervalsloss / 60 + torch.log(self.log_sigma1*self.log_sigma2)
        return 0.5 * intervalsloss + 0.5 * desloss, desloss, intervalsloss
    
    def des_loss(self, output, target):
        target = target.contiguous().reshape(-1)
        output = output.reshape(-1, self.vocab_size)
        mask = (target != 0).to(target.device)
        target = target[mask]
        output = output[mask]
        loss = self.crossEntropy(output.reshape(-1, self.vocab_size), target)
        return loss
    
    def intervals_loss(self, intervals_output, target):
        mask = (target != 0).to(target.device)
        intervals_output = intervals_output[mask]
        target = target[mask]
        loss = self.mse(intervals_output, target)
        return loss
    
    def DesTimeLoss(self, output, target, intervals_output, intervals_target):
        mask = (target != 0).to(target.device)
        intervals_mask = (intervals_target != 0).to(intervals_target.device)
        output = torch.argmax(output, dim=-1)
        output = output[mask]
        target = target[mask]
        station_corrects = (output == target).flatten()
        total = len(target)
        
        time_corrects = (torch.abs(intervals_output - intervals_target) < 60)
        time_corrects = time_corrects[intervals_mask]
        
        #station_corrects和time_corrects都为True的有多少个
        corrects = (station_corrects & time_corrects).sum().float()
        
        return - torch.log((corrects + 1)/(total + 1))
    
    def mae_loss(self, predicted, observed, null_val=0.0):
        mask = (observed != null_val)
        mask = mask.float()
        mask /=  torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(predicted - observed)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

    def forward(self, enc_src, enc_intervals, src, enc_HoD, enc_DoW, travel_index, travel_mask, intervals, tgt = None):
        
        intervals_emb = self.intervalsEncoder(enc_HoD, enc_DoW, intervals)
        # enc_intervals_emb = self.enc_intervalsEncoder(enc_HoD, enc_DoW, enc_intervals)
        # enc_src_mask = self.enc_src_mask(enc_src)
        src_mask = self.generate_mask(src)
        station_mask = self.station_mask(src, travel_mask)
        # intervals_mask = self.generate_mask(intervals.squeeze(-1))
        # enc_src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(enc_src, enc_DoW, enc_HoD)))
        src_embedded = self.station_embedding(src)
        
        # enc_src_embedded = self.enc_station_cov(F.pad(enc_src_embedded, (0,0,1,0), value=0).permute(0,2,1)).permute(0,2,1)
        # enc_output = enc_src_embedded + enc_intervals_emb
        # for enc_layer in self.encoder_layers:
        #     enc_output = enc_layer(enc_output, enc_src_mask)
            
        # src_embedded[:, 0, :] = enc_output[:, 0, :]

        
        enc_output = src_embedded
        for enc_layer in self.trajEncoderLayers:
            enc_output = enc_layer(enc_output, src_mask)
    
        
        output = self.fc(enc_output)
        
        if tgt is None:
            tgt = torch.argmax(output, -1) 
        dec_output = self.station_embedding(tgt)
        # src_embedded = self.station_embedding(src)
        
        start_src_embedded = src_embedded + self.start_end_embedding(torch.zeros(src_embedded.shape[0],src_embedded.shape[1], dtype=torch.int32).to(device))
        end_src_embedded = F.pad(src_embedded[:,1:,:], (0,0,0,1), value=0) + self.start_end_embedding(torch.ones(src_embedded.shape[0],src_embedded.shape[1], dtype=torch.int32).to(device))
        end_dec_output = dec_output + self.start_end_embedding(torch.ones(dec_output.shape[0],dec_output.shape[1], dtype=torch.int32).to(device))

        staion_cov_emb = self.tgt_intervals_cov1(torch.stack((start_src_embedded, end_src_embedded), dim=1).permute(0,3,1,2)).permute(0,2,3,1).squeeze(1)
        tgt_sections_output = self.tgt_intervals_cov1(torch.stack((start_src_embedded, end_dec_output), dim=1).permute(0,3,1,2)).permute(0,2,3,1).squeeze(1)
        
        
        for i, cross_dec_layer in enumerate(self.cross_decoder_layers):
            if i == 0:
                tgt_sections_output, intervals_attn_output = cross_dec_layer(tgt_sections_output, staion_cov_emb, intervals_emb, None, station_mask, station_mask)
            else:
                tgt_sections_output, intervals_attn_output = cross_dec_layer(tgt_sections_output, staion_cov_emb, intervals_emb, intervals_attn_output, station_mask, station_mask)
        
        
        # intervals_dec_output = fusionGate(tgt_sections_output, dec_output)
        intervals_dec_output = intervals_attn_output
        intervals_output = self.intervals_fc(intervals_dec_output)

        return output, intervals_output

class TrajPredTransformerV3(nn.Module):
    def __init__(self, conf):
        super(TrajPredTransformerV3, self).__init__()
        self.num_layers = int(conf["num_layers"])
        self.vocab_size = int(conf["vocab_size"])
        # self.tgt_intervals_cov = nn.Conv2d(int(conf["max_len"]), int(conf["max_len"]), (2,1))
        self.station_cov = nn.Conv1d(int(conf["hidden_size"]), int(conf["hidden_size"]), 2)
        self.enc_station_cov = nn.Conv1d(int(conf["hidden_size"]), int(conf["hidden_size"]), 2)
        # self.tgt_intervals_cov = nn.Conv2d(2, 1, (1,1))
        self.tgt_intervals_cov = nn.Conv2d(int(conf["hidden_size"]), int(conf["hidden_size"]), (2,1))
        self.tgt_intervals_cov1 = nn.Conv2d(int(conf["hidden_size"]), int(conf["hidden_size"]), (2,1))
        self.decoder_embedding = nn.Embedding(self.vocab_size, int(conf["hidden_size"]))
        # self.decoder_embedding = StationEmb(self.vocab_size, int(conf["max_len"]), int(conf["hidden_size"]))
        self.encoder_embedding = EncoderEmb(self.vocab_size, int(conf["hidden_size"]))
        self.positional_encoding = PositionalEncoding(int(conf["hidden_size"]), int(conf["max_len"]))
        self.start_end_embedding = nn.Embedding(2, int(conf["hidden_size"]))
        self.intervalsEncoder = IntervalsEncoder(int(conf["max_len"]), int(conf["hidden_size"]), float(conf["dropout"]))
        self.enc_intervalsEncoder = IntervalsEncoder(int(conf["max_len"]), int(conf["hidden_size"]), float(conf["dropout"]))

        self.encoder_layers = nn.ModuleList([EncoderLayer(int(conf["hidden_size"]), int(conf["num_heads"]), int(conf["d_ff"]), float(conf["dropout"])) for _ in range(self.num_layers)])
        # self.adj = self.get_supports(conf['adj_path'])
        self.adj = self.adjecent(conf['adj_path'])
        #放入其他网络相同的设备
        self.adj = self.adj.to(device)
        self.decoder_layers = nn.ModuleList([trajDecoderLayer(int(conf["hidden_size"]), int(conf["num_heads"]), int(conf["d_ff"]), float(conf["dropout"])) for _ in range(self.num_layers)])
        self.staion_attn_layers = nn.ModuleList([trajDecoderLayer(int(conf["hidden_size"]), int(conf["num_heads"]), int(conf["d_ff"]), float(conf["dropout"])) for _ in range(self.num_layers)])
        self.cross_decoder_layers = nn.ModuleList([StaionsAttensionLayer(int(conf["hidden_size"]), int(conf["num_heads"]), int(conf["d_ff"]), float(conf["dropout"])) for _ in range(self.num_layers)])

        # self.fc = nn.Linear(int(conf["hidden_size"]), self.vocab_size)
        self.fc = FC(int(conf["hidden_size"]) * self.num_layers, int(conf["hidden_size"]), self.vocab_size, float(conf["dropout"]))
        # self.intervals_fc = nn.Linear(int(conf["hidden_size"]), 1)
        self.intervals_fc = Intervals_FC(int(conf["hidden_size"]) * self.num_layers, int(conf["hidden_size"]), float(conf["dropout"]))
        self.dropout = nn.Dropout(float(conf["dropout"]))
        self.crossEntropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        
        # 不确定性权重参数（可训练）
        self.log_sigma1 = nn.Parameter(torch.tensor(1.0))
        self.log_sigma2 = nn.Parameter(torch.tensor(1.0))
        
        
    def adjecent(self, file_adj):
        '''
        :return: adj matrix
        '''
        data = pd.read_csv(filepath_or_buffer=file_adj)
        adj = np.zeros(shape=[self.vocab_size, self.vocab_size], dtype=np.int32)
        for line in data[['from_id', 'to_id']].values:
            adj[int(line[0])][int(line[1])] = 1
        #adj 标准化
        # adj = self.normalization(adj)
        adj = adj + np.eye(self.vocab_size)
        return torch.from_numpy(adj)
    
    def get_supports(self, file_adj):
        data = pd.read_csv(filepath_or_buffer=file_adj)
        edge_index = [[line[0],line[1]]for line in data[['from_id', 'to_id']].values]
        edge_index.extend([[i,i]for i in range(self.vocab_size)])
        edge_index = torch.tensor(edge_index,dtype=torch.long)
        return edge_index.t().contiguous()

    def generate_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(3).to(src.device)
        seq_length = src.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=src.device), diagonal=1)).bool()
        src_mask = src_mask & nopeak_mask
        return  src_mask
    
    def enc_src_mask(self, src):
        return (src != 0).unsqueeze(1).unsqueeze(2).to(src.device)
    
    def station_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(3).to(src.device)
        seq_length = src.size(1)
        nopeak_mask = (torch.tril(torch.ones(1, seq_length, seq_length, device=src.device), diagonal=-1)).bool()
        src_mask = src_mask & nopeak_mask
        return  src_mask
        
    
    def loss(self, output, target, intervals_output, intervals_target, mean, std):
        desloss = self.des_loss(output, target)
        mask = (intervals_target != 0.0).to(intervals_target.device)
        intervals_output = intervals_output[mask]
        intervals_target = intervals_target[mask]
        # intervals_output = unnormalize(intervals_output, mean, std)
        # intervals_target = exp_unnormalize(intervals_target, torch.tensor(0.011146042964972321).cuda(), mean, std)
        # intervals_target = unnormalize(intervals_target, mean, std)
        # intervals_output = exp_unnormalize(intervals_output, torch.tensor(0.011146042964972321).cuda(), mean, std)
        intervalsloss = self.mae_loss(intervals_output, intervals_target)
        return (1/(2*self.log_sigma1**2))*desloss + (1/(2*self.log_sigma2**2))*intervalsloss + torch.log(self.log_sigma1*self.log_sigma2), desloss, intervalsloss
        # return 0.7 * intervalsloss + 0.3 * desloss, desloss, intervalsloss
    
    def des_loss(self, output, target):
        target = target.contiguous().reshape(-1)
        output = output.reshape(-1, self.vocab_size)
        mask = (target != 0).to(target.device)
        target = target[mask]
        output = output[mask]
        loss = self.crossEntropy(output.reshape(-1, self.vocab_size), target)
        return loss
    
    def intervals_loss(self, intervals_output, target):
        mask = (target != 0).to(target.device)
        intervals_output = intervals_output[mask]
        target = target[mask]
        loss = self.mse(intervals_output, target)
        return loss
    
    def DesTimeLoss(self, output, target, intervals_output, intervals_target):
        mask = (target != 0).to(target.device)
        intervals_mask = (intervals_target != 0).to(intervals_target.device)
        output = torch.argmax(output, dim=-1)
        output = output[mask]
        target = target[mask]
        station_corrects = (output == target).flatten()
        total = len(target)
        
        time_corrects = (torch.abs(intervals_output - intervals_target) < 60)
        time_corrects = time_corrects[intervals_mask]
        
        #station_corrects和time_corrects都为True的有多少个
        corrects = (station_corrects & time_corrects).sum().float()
        
        return - torch.log((corrects + 1)/(total + 1))
    
    def mae_loss(self, predicted, observed, null_val=0.0):
        mask = (observed != null_val)
        mask = mask.float()
        mask /=  torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(predicted - observed)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

    def forward(self, enc_src, enc_intervals, src, enc_HoD, enc_DoW, intervals, travel_index, tgt = None):
        
        intervals_emb = self.intervalsEncoder(enc_HoD, enc_DoW, intervals)
        # enc_intervals_emb = self.enc_intervalsEncoder(enc_HoD, enc_DoW, enc_intervals)
        # enc_src_mask = self.enc_src_mask(enc_src)
        src_mask = self.generate_mask(src)
        station_mask = self.station_mask(src)
        # intervals_mask = self.generate_mask(intervals.squeeze(-1))
        # enc_src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(enc_src, enc_DoW, enc_HoD)))
        src_embedded = self.decoder_embedding(src)
        # src_embedded = self.decoder_embedding(src)
        
        # enc_src_embedded = self.enc_station_cov(F.pad(enc_src_embedded, (0,0,1,0), value=0).permute(0,2,1)).permute(0,2,1)
        # enc_output = enc_src_embedded + enc_intervals_emb
        # for enc_layer in self.encoder_layers:
        #     enc_output = enc_layer(enc_output, enc_src_mask)
            
        # src_embedded[:, 0, :] = enc_output[:, 0, :]

        
        dec_output = src_embedded
        stationEmbedding = self.decoder_embedding.weight.reshape(1, self.vocab_size, -1)
        dec_outputs = []
        intervals_attn_outputs = []
        for i, dec_layer in enumerate(self.decoder_layers):
            if i == 0:
                # dec_output, output = dec_layer(dec_output, src_mask, src, stationEmbedding, None)
                dec_output, output = dec_layer(dec_output, src_mask, src, stationEmbedding, None)
            else:
                # dec_output, output = dec_layer(dec_output, src_mask, src, None, output)
                dec_output, output = dec_layer(dec_output, src_mask, src, stationEmbedding, output)
            
            dec_outputs.append(dec_output)
    
            # output = output.permute(0,2,1)
            
            # if self.adj is not None:
            #     adj = self.adj[src]
            #     output = output.masked_fill(adj == 0, 0.0)
            
            # if tgt is None:
            #     tgt = torch.argmax(output, -1) 
            # dec_output = self.decoder_embedding(tgt)
            # dec_output = self.decoder_embedding(tgt)
            # src_embedded = self.decoder_embedding(src)
            
            start_src_embedded = src_embedded + self.start_end_embedding(torch.zeros(src_embedded.shape[0],src_embedded.shape[1], dtype=torch.int32).to(device))
            end_src_embedded = F.pad(src_embedded[:,1:,:], (0,0,0,1), value=0) + self.start_end_embedding(torch.ones(src_embedded.shape[0],src_embedded.shape[1], dtype=torch.int32).to(device))
            end_dec_output = dec_output + self.start_end_embedding(torch.ones(dec_output.shape[0],dec_output.shape[1], dtype=torch.int32).to(device))

            staion_cov_emb = self.tgt_intervals_cov1(torch.stack((start_src_embedded, end_src_embedded), dim=1).permute(0,3,1,2)).permute(0,2,3,1).squeeze(1)
            tgt_sections_output = self.tgt_intervals_cov1(torch.stack((start_src_embedded, end_dec_output), dim=1).permute(0,3,1,2)).permute(0,2,3,1).squeeze(1)
            
            
            # for i, cross_dec_layer in enumerate(self.cross_decoder_layers):
            cross_dec_layer = self.cross_decoder_layers[i]
            if i == 0:
                tgt_sections_output, intervals_attn_output = cross_dec_layer(tgt_sections_output, staion_cov_emb, intervals_emb, None, station_mask, station_mask)
            else:
                tgt_sections_output, intervals_attn_output = cross_dec_layer(tgt_sections_output, staion_cov_emb, intervals_emb, intervals_attn_output, station_mask, station_mask)

            intervals_attn_outputs.append(intervals_attn_output)
            
        
        # intervals_dec_output = fusionGate(tgt_sections_output, dec_output)
        # intervals_dec_output = intervals_attn_output
        intervals_dec_output = torch.cat(intervals_attn_outputs, dim=-1)
        intervals_output = self.intervals_fc(intervals_dec_output)
        
        dec_output = torch.cat(dec_outputs, dim=-1)
        output = self.fc(dec_output)
        return output, intervals_output
   
if __name__ == '__main__':
    label = torch.randint(0, 100, (32, 10)).to(device)
    tgt = F.pad(label[:, 1:], (0, 1), value=0)
    src = F.pad(label[:, :-1], (0, 1), value=0)
    model = Transformer(100, 512, 8, 6, 2048, 100, 0.1).to(device)
    output = model(src, tgt)

