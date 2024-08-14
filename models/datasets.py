from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
from utils.metrics import normalize

class ETCEnDataSet(Dataset):
    def __init__(self, data, tokenizer, device, is_base_history_eval=False):
        self.data = data
        self.device =device
        self.tokenizer = tokenizer
        self.is_base_history_eval =is_base_history_eval

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_base_history_eval:
            src_stations = torch.tensor(self.data[idx]["stationId"]).to(self.device)
            src_DoW = torch.tensor(self.data[idx]["DoW"]).to(self.device)
            src_HoD = torch.tensor(self.data[idx]["HoD"]).to(self.device)
            src_intervals = torch.tensor(self.data[idx]["intervals"]).to(self.device)
        else:
            src_stations = torch.tensor(self.data[idx]["stationId"][:-1]).to(self.device)
            src_DoW = torch.tensor(self.data[idx]["DoW"][:-1]).to(self.device)
            src_HoD = torch.tensor(self.data[idx]["HoD"][:-1]).to(self.device)
            src_intervals = torch.tensor(self.data[idx]["intervals"][:-1]).to(self.device)
            
        mask = (src_intervals != 0.0)
        src_intervals = src_intervals[mask]
        src_intervals = F.pad(src_intervals, (1, 0), value=0)
        src_stations = src_stations[F.pad(mask, (1,0), value=True)]
        src_DoW = src_DoW[F.pad(mask, (1,0), value=True)]
        src_HoD = src_HoD[F.pad(mask, (1,0), value=True)]
        
        
        src = src_stations[:-1]
        DoW = src_DoW[:-1]
        HoD = src_HoD[:-1]
        intervals = src_intervals[:-1]
        src = self.tokenizer.tokenize(src)
        DoW = self.tokenizer.time_tokenize(DoW, 24)
        HoD = self.tokenizer.time_tokenize(HoD, 7)
        intervals = self.tokenizer.time_tokenize(intervals, self.tokenizer.mask_index, True)
            
            
        tgt = self.tokenizer.tokenize(src_stations)[1:]
        tgt = F.pad(tgt, (0, 1), value = 0)
        intervals_tgt = F.pad(src_intervals[2:], (1, 1), value=0)
        intervals_tgt = self.tokenizer.time_tokenize(intervals_tgt, self.tokenizer.mask_index, True)
        res = dict()
        res["src"] = src
        res["tgt"] = tgt
        res["DoW"] = DoW
        res["HoD"] = HoD
        res["intervals"] = intervals
        res["intervals_tgt"] = intervals_tgt
        return res
    
class ETCEnDataSetV1(Dataset):
    def __init__(self, data, tokenizer, device, is_base_history_eval=False):
        self.data = data
        self.device =device
        self.tokenizer = tokenizer
        self.is_base_history_eval =is_base_history_eval
        self.mean = 465.5223178538118
        self.std = 959.7372163727321
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_base_history_eval:
            src_stations = torch.tensor(self.data[idx]["stationId"]).to(self.device)
            src_DoW = torch.tensor(self.data[idx]["DoW"]).to(self.device)
            src_HoD = torch.tensor(self.data[idx]["HoD"]).to(self.device)
            src_intervals = normalize(torch.tensor(self.data[idx]["intervals"]).to(self.device))
        else:
            src_stations = torch.tensor(self.data[idx]["stationId"][:-1]).to(self.device)
            src_DoW = torch.tensor(self.data[idx]["DoW"][:-1]).to(self.device)
            src_HoD = torch.tensor(self.data[idx]["HoD"][:-1]).to(self.device)
            src_intervals = normalize(torch.tensor(self.data[idx]["intervals"][:-1]).to(self.device))
            
        mask = (src_intervals != 0.0)
        src_intervals = src_intervals[mask]
        # src_intervals = F.pad(src_intervals, (1, 0), value=0)
        src_stations = src_stations[F.pad(mask, (1,0), value=True)]
        src_DoW = src_DoW[F.pad(mask, (1,0), value=True)]
        src_HoD = src_HoD[F.pad(mask, (1,0), value=True)]
        
        
        src = src_stations[10:-1]
        enc_src = F.pad(src_stations[:10], (0,1), value=self.tokenizer.eos_index)
        # DoW = src_DoW[10:-1]
        enc_DoW = src_DoW[:10]
        # HoD = src_HoD[10:-1]
        enc_HoD = src_HoD[:10]
        intervals = src_intervals[10:]
        enc_intervals = src_intervals[:10]
        src = self.tokenizer.tokenize(src)
        enc_src = self.tokenizer.tokenize(enc_src)
        # DoW = self.tokenizer.time_tokenize(DoW, 24)
        enc_DoW = self.tokenizer.time_tokenize(enc_DoW, 7)
        # HoD = self.tokenizer.time_tokenize(HoD, 7)
        enc_HoD = self.tokenizer.time_tokenize(enc_HoD, 24)
        intervals = self.tokenizer.time_tokenize(intervals, self.tokenizer.mask_index, True)
        enc_intervals = self.tokenizer.time_tokenize(enc_intervals, self.tokenizer.mask_index, True)
            
            
        tgt = self.tokenizer.tokenize(src_stations[10:])[1:]
        tgt = F.pad(tgt, (0, 1), value = 0)
        res = dict()
        res["enc_src"] = enc_src
        res["src"] = src
        res["tgt"] = tgt
        # res["DoW"] = DoW
        res["enc_DoW"] = enc_DoW
        # res["HoD"] = HoD
        res["enc_HoD"] = enc_HoD
        res["intervals"] = intervals
        res["enc_intervals"] = enc_intervals
        return res