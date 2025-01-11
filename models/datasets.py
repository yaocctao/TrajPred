import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
from utils.metrics import normalize, MinMaxNormalize
from utils.standardization import exp_normalize
from tqdm import tqdm

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
            src_intervals = torch.tensor(self.data[idx]["intervals"]).to(self.device)
        else:
            src_stations = torch.tensor(self.data[idx]["stationId"][:-1]).to(self.device)
            src_DoW = torch.tensor(self.data[idx]["DoW"][:-1]).to(self.device)
            src_HoD = torch.tensor(self.data[idx]["HoD"][:-1]).to(self.device)
            src_intervals = torch.tensor(self.data[idx]["intervals"][:-1]).to(self.device)
            
        mask = (src_intervals != 0.0)
        src_intervals = src_intervals[mask]
        src_intervals = normalize(src_intervals)
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
        res["src"] = src
        pre_index = F.pad(mask[mask == True][10:], (1,0), value=True)
        if self.is_base_history_eval:
            pre_index = torch.concat([~pre_index[:-1], torch.tensor([True]).to(self.device)], 0)
            
            pre_index = self.tokenizer.time_tokenize(pre_index, False)[1:]
            pre_index = F.pad(pre_index, (0, 1), value = False)
        else:
            pre_index = self.tokenizer.time_tokenize(pre_index, False)[1:]
            pre_index = F.pad(pre_index, (0, 1), value = False)
        res["enc_src"] = enc_src
        res["pre_index"] = pre_index
        res["tgt"] = tgt
        # res["DoW"] = DoW
        res["enc_DoW"] = enc_DoW
        # res["HoD"] = HoD
        res["enc_HoD"] = enc_HoD
        res["intervals"] = intervals
        res["enc_intervals"] = enc_intervals
        return res
    
    
class ETCTrajDataSetV1ForLSTM(Dataset):
    def __init__(self, data, tokenizer, mean, std, device, is_base_history_eval=False, is_eval=False):
        self.device =device
        self.tokenizer = tokenizer
        self.is_base_history_eval = is_base_history_eval
        self.is_eval = is_eval
        self.mean = mean
        self.std = std
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def preprocess_data(self, data):
        new_data = []
        for d in tqdm(data):
            if self.is_base_history_eval or self.is_eval:
                if len(d["stationId"]) == 1 and not self.is_eval:
                    return dict()
                else:
                    src_stations = torch.tensor(self.list_squeeze(d["stationId"]), dtype=torch.int64).to(self.device)
                    # travel_index = torch.tensor(self.list_squeeze(d["travel_index"]), dtype=torch.int32).to(self.device)
                    # travel_mask = torch.tensor(self.list_squeeze(d["travel_mask"]), dtype=torch.int32).to(self.device)
                    # src_DoW = torch.tensor(self.list_squeeze(d["DoW"]), dtype=torch.int32).to(self.device)
                    # src_HoD = torch.tensor(self.list_squeeze(d["HoD"]), dtype=torch.int32).to(self.device)
                    src_intervals = torch.tensor(self.list_squeeze(d["intervals"]), dtype=torch.float32).to(self.device)
                    src_intervals[0] = -1
                    
                    next_travel_len =  len(src_stations) - len(torch.tensor(self.list_squeeze(d["stationId"][:-1]), dtype=torch.int64).to(self.device))

            else:
                if len(d["stationId"]) == 1:
                    src_stations = torch.tensor(self.list_squeeze(d["stationId"]), dtype=torch.int64).to(self.device)
                    # travel_index = torch.tensor(self.list_squeeze(d["travel_index"]), dtype=torch.int32).to(self.device)
                    # travel_mask = torch.tensor(self.list_squeeze(d["travel_mask"]), dtype=torch.int32).to(self.device)
                    # src_DoW = torch.tensor(self.list_squeeze(d["DoW"]), dtype=torch.int32).to(self.device)
                    # src_HoD = torch.tensor(self.list_squeeze(d["HoD"]), dtype=torch.int32).to(self.device)
                    src_intervals = torch.tensor(self.list_squeeze(d["intervals"]), dtype=torch.float32).to(self.device)
                    src_intervals[0] = -1
                else:
                    src_stations = torch.tensor(self.list_squeeze(d["stationId"][:-1]), dtype=torch.int64).to(self.device)
                    # travel_index = torch.tensor(self.list_squeeze(d["travel_index"][:-1]), dtype=torch.int32).to(self.device)
                    # travel_mask = torch.tensor(self.list_squeeze(d["travel_mask"][:-1]), dtype=torch.int32).to(self.device)
                    # src_DoW = torch.tensor(self.list_squeeze(d["DoW"][:-1]), dtype=torch.int32).to(self.device)
                    # src_HoD = torch.tensor(self.list_squeeze(d["HoD"][:-1]), dtype=torch.int32).to(self.device)
                    src_intervals = torch.tensor(self.list_squeeze(d["intervals"][:-1]), dtype=torch.float32).to(self.device)
                    src_intervals[0] = -1
                
            mask = (src_intervals != 0.0)
            src_intervals = src_intervals[mask]
            # src_intervals = normalize(src_intervals, self.mean, self.std)
            src_intervals[0] = torch.tensor(0.0, dtype=torch.float32).to(self.device)
            src_stations = src_stations[mask]
            # src_DoW = src_DoW[mask]
            # src_HoD = src_HoD[mask]
            
            
            # src = self.tokenizer.tokenize(src_stations[:-1])
            # travel_index = self.tokenizer.tokenize(travel_index[:-1])
            # travel_index[0] = torch.tensor(0).to(self.device)
            
            # src_travel_mask = self.tokenizer.tokenize(travel_mask[:-1])
            # src_travel_mask[0] = torch.tensor(0).to(self.device)
            
            # tgt_travel_mask = self.tokenizer.tokenize(travel_mask)[1:]
            # tgt_travel_mask = F.pad(tgt_travel_mask, (0, 1), value = 0)
            
            # travel_mask = src_travel_mask & tgt_travel_mask
            
            # intervals = self.tokenizer.time_tokenize(src_intervals[:-1], self.tokenizer.mask_index, True)
            # tgt_intervals = self.tokenizer.time_tokenize(src_intervals[1:], self.tokenizer.mask_index, True)
            
            # tgt = self.tokenizer.tokenize(src_stations)[1:]
            # tgt = F.pad(tgt, (0, 1), value = 0)
            if self.is_base_history_eval:
                next_travel = mask[-next_travel_len:]
                next_travel = next_travel[next_travel == True]
                
                pre_travel = mask[:-next_travel_len]
                pre_travel = pre_travel[pre_travel == True]
                
                pre_index = torch.concat([~pre_travel, next_travel], 0)
                # pre_index = self.tokenizer.time_tokenize(pre_index, False)[1:]
                # pre_index = F.pad(pre_index, (0, 1), value = False)
            else:
                pre_index = mask[mask == True]
                # pre_index = self.tokenizer.time_tokenize(pre_index, False)[1:]
                # pre_index = F.pad(pre_index, (0, 1), value = False)
            
            #获取为true的索引位置
            index = torch.nonzero(pre_index == True).flatten()
            for j in index[:255]:
                res = dict()
                
                res["src"] = np.array(src_stations[j-30:j])
                res["tgt"] = np.array(src_stations[j])
                res["intervals"] = np.array(F.pad(src_intervals[j-29:j], (1, 0), value = 0.0).unsqueeze(-1))
                res["tgt_intervals"] = src_intervals[j].unsqueeze(-1)
                
                res["enc_src"] = torch.tensor(0).to(self.device)
                res["enc_DoW"] = torch.tensor(0).to(self.device)
                res["enc_HoD"] = torch.tensor(0).to(self.device)
                res["enc_intervals"] = torch.tensor(0).to(self.device)
                new_data.append(res)
        return new_data
    
    def list_squeeze(self, lists):
        return [e for l in lists for e in l]

    def __getitem__(self, idx):
        res = dict()
        res["src"] = torch.tensor(self.data[idx]["src"], dtype=torch.int64).to(self.device)
        res["tgt"] = torch.tensor(self.data[idx]["tgt"], dtype=torch.int64).to(self.device)
        intervals = torch.tensor(self.data[idx]["intervals"], dtype=torch.float32).to(self.device)
        intervals = normalize(intervals, self.mean, self.std)
        # intervals = exp_normalize(intervals)
        intervals[0] = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        res["intervals"] = intervals
        res["tgt_intervals"] =  normalize(torch.tensor(self.data[idx]["tgt_intervals"], dtype=torch.float32).to(self.device), self.mean, self.std)
        # res["tgt_intervals"] =  exp_normalize(torch.tensor(self.data[idx]["tgt_intervals"], dtype=torch.float32).to(self.device))
        
        res["enc_src"] = torch.tensor(self.data[idx]['enc_src'], dtype=torch.float32).to(self.device)
        res["enc_DoW"] = torch.tensor(self.data[idx]['enc_DoW'], dtype=torch.float32).to(self.device)
        res["enc_HoD"] = torch.tensor(self.data[idx]['enc_HoD'], dtype=torch.float32).to(self.device)
        res["enc_intervals"] = torch.tensor(self.data[idx]['enc_intervals'], dtype=torch.float32).to(self.device)
        return res
    
class ETCTrajDataSetV1(Dataset):
    def __init__(self, data, tokenizer, device, mean, std, is_base_history_eval=False, is_eval=False):
        self.data = data
        self.device =device
        self.tokenizer = tokenizer
        self.is_base_history_eval = is_base_history_eval
        self.is_eval = is_eval
        self.mean = mean
        self.std = std
        
    def __len__(self):
        return len(self.data)
    
    def list_squeeze(self, lists):
        return [e for l in lists for e in l]

    def __getitem__(self, idx):
        if self.is_base_history_eval or self.is_eval:
            if len(self.data[idx]["stationId"]) == 1 and not self.is_eval:
                return dict()
            else:
                src_stations = torch.tensor(self.list_squeeze(self.data[idx]["stationId"]), dtype=torch.int64).to(self.device)
                travel_index = torch.tensor(self.list_squeeze(self.data[idx]["travel_index"]), dtype=torch.int32).to(self.device)
                travel_mask = torch.tensor(self.list_squeeze(self.data[idx]["travel_mask"]), dtype=torch.int32).to(self.device)
                src_DoW = torch.tensor(self.list_squeeze(self.data[idx]["DoW"]), dtype=torch.int32).to(self.device)
                src_HoD = torch.tensor(self.list_squeeze(self.data[idx]["HoD"]), dtype=torch.int32).to(self.device)
                src_intervals = torch.tensor(self.list_squeeze(self.data[idx]["intervals"]), dtype=torch.float32).to(self.device)
                src_intervals[0] = -1
                
                next_travel_len =  len(src_stations) - len(torch.tensor(self.list_squeeze(self.data[idx]["stationId"][:-1]), dtype=torch.int64).to(self.device))

        else:
            if len(self.data[idx]["stationId"]) == 1:
                src_stations = torch.tensor(self.list_squeeze(self.data[idx]["stationId"]), dtype=torch.int64).to(self.device)
                travel_index = torch.tensor(self.list_squeeze(self.data[idx]["travel_index"]), dtype=torch.int32).to(self.device)
                travel_mask = torch.tensor(self.list_squeeze(self.data[idx]["travel_mask"]), dtype=torch.int32).to(self.device)
                src_DoW = torch.tensor(self.list_squeeze(self.data[idx]["DoW"]), dtype=torch.int32).to(self.device)
                src_HoD = torch.tensor(self.list_squeeze(self.data[idx]["HoD"]), dtype=torch.int32).to(self.device)
                src_intervals = torch.tensor(self.list_squeeze(self.data[idx]["intervals"]), dtype=torch.float32).to(self.device)
                src_intervals[0] = -1
            else:
                src_stations = torch.tensor(self.list_squeeze(self.data[idx]["stationId"][:-1]), dtype=torch.int64).to(self.device)
                travel_index = torch.tensor(self.list_squeeze(self.data[idx]["travel_index"][:-1]), dtype=torch.int32).to(self.device)
                travel_mask = torch.tensor(self.list_squeeze(self.data[idx]["travel_mask"][:-1]), dtype=torch.int32).to(self.device)
                src_DoW = torch.tensor(self.list_squeeze(self.data[idx]["DoW"][:-1]), dtype=torch.int32).to(self.device)
                src_HoD = torch.tensor(self.list_squeeze(self.data[idx]["HoD"][:-1]), dtype=torch.int32).to(self.device)
                src_intervals = torch.tensor(self.list_squeeze(self.data[idx]["intervals"][:-1]), dtype=torch.float32).to(self.device)
                src_intervals[0] = -1
             
        mask = (src_intervals != 0.0)
        src_intervals = src_intervals[mask]
        # src_intervals = normalize(src_intervals, self.mean, self.std)
        src_intervals = exp_normalize(src_intervals, 0.0055730214824861605, self.mean, self.std)
        src_intervals[0] = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        src_stations = src_stations[mask]
        travel_mask = travel_mask[mask]
        travel_index = travel_index[mask]
        # src_DoW = src_DoW[mask]
        # src_HoD = src_HoD[mask]
        
        
        src = self.tokenizer.tokenize(src_stations[:-1])
        # src_DoW = self.tokenizer.tokenize(src_DoW[:-1], 24)
        travel_index = self.tokenizer.tokenize(travel_index[:-1])
        travel_index[0] = torch.tensor(0).to(self.device)
        
        src_travel_mask = self.tokenizer.tokenize(travel_mask[:-1])
        src_travel_mask[0] = torch.tensor(0).to(self.device)
        
        tgt_travel_mask = self.tokenizer.tokenize(travel_mask)[1:]
        tgt_travel_mask = F.pad(tgt_travel_mask, (0, 1), value = 0)
        
        travel_mask = src_travel_mask & tgt_travel_mask
        
        intervals = self.tokenizer.time_tokenize(src_intervals, self.tokenizer.mask_index, True)
        
        # tgt_intervals = self.tokenizer.time_tokenize(src_intervals[1:], self.tokenizer.mask_index, True)
            
            
        tgt = self.tokenizer.tokenize(src_stations)[1:]
        tgt = F.pad(tgt, (0, 1), value = 0)
        res = dict()
        res["src"] = src
        if self.is_base_history_eval:
            next_travel = mask[-next_travel_len:]
            next_travel = next_travel[next_travel == True]
            
            pre_travel = mask[:-next_travel_len]
            pre_travel = pre_travel[pre_travel == True]
            
            pre_index = torch.concat([~pre_travel, next_travel], 0)
            # pre_index = self.tokenizer.time_tokenize(pre_index, False)[1:]
            # pre_index = F.pad(pre_index, (0, 1), value = False)
            pre_index = self.tokenizer.time_tokenize(pre_index, False)
        else:
            pre_index = mask[mask == True]
            # pre_index = self.tokenizer.time_tokenize(pre_index, False)[1:]
            # pre_index = F.pad(pre_index, (0, 1), value = False)
            pre_index = self.tokenizer.time_tokenize(pre_index, False)
            
        res["tgt"] = tgt
        res["intervals"] = intervals
        res["tgt_intervals"] = intervals
        res["pre_index"] = pre_index
        
        res["travel_index"] = travel_index
        res["travel_mask"] = travel_mask
        
        res["enc_src"] = torch.tensor(0).to(self.device)
        res["enc_DoW"] = torch.tensor(0).to(self.device)
        res["enc_HoD"] = torch.tensor(0).to(self.device)
        res["enc_intervals"] = torch.tensor(0).to(self.device)
        return res
    

class ETCTrajDataSetV2(Dataset):
    def __init__(self, data, tokenizer, device, mean, std, is_base_history_eval=False):
        self.data = data
        self.device =device
        self.tokenizer = tokenizer
        self.is_base_history_eval = is_base_history_eval
        # self.mean = 465.5223178538118
        # self.mean = 96.1122
        self.mean = mean
        # self.std = 959.7372163727321
        # self.std = 366.9742
        self.std = std
        
    def __len__(self):
        return len(self.data)
    
    def list_squeeze(self, lists):
        return [e for l in lists for e in l]

    def __getitem__(self, idx):
        if self.is_base_history_eval:
            if len(self.data[idx]["stationId"]) == 1:
                return dict()
            else:
                src_stations = torch.tensor(self.list_squeeze(self.data[idx]["stationId"]), dtype=torch.int64).to(self.device)
                travel_index = torch.tensor(self.list_squeeze(self.data[idx]["travel_index"]), dtype=torch.int64).to(self.device)
                travel_mask = torch.tensor(self.list_squeeze(self.data[idx]["travel_mask"]), dtype=torch.int64).to(self.device)
                src_DoW = torch.tensor(self.list_squeeze(self.data[idx]["DoW"]), dtype=torch.int32).to(self.device)
                src_HoD = torch.tensor(self.list_squeeze(self.data[idx]["HoD"]), dtype=torch.int32).to(self.device)
                src_intervals = torch.tensor(self.list_squeeze(self.data[idx]["intervals"]), dtype=torch.float32).to(self.device)
                src_intervals[0] = -1
                
                next_travel_len =  len(src_stations) - len(torch.tensor(self.list_squeeze(self.data[idx]["stationId"][:-1]), dtype=torch.int64).to(self.device))

        else:
            if len(self.data[idx]["stationId"]) == 1:
                src_stations = torch.tensor(self.list_squeeze(self.data[idx]["stationId"]), dtype=torch.int64).to(self.device)
                travel_index = torch.tensor(self.list_squeeze(self.data[idx]["travel_index"]), dtype=torch.int64).to(self.device)
                travel_mask = torch.tensor(self.list_squeeze(self.data[idx]["travel_mask"]), dtype=torch.int64).to(self.device)
                src_DoW = torch.tensor(self.list_squeeze(self.data[idx]["DoW"]), dtype=torch.int32).to(self.device)
                src_HoD = torch.tensor(self.list_squeeze(self.data[idx]["HoD"]), dtype=torch.int32).to(self.device)
                src_intervals = torch.tensor(self.list_squeeze(self.data[idx]["intervals"]), dtype=torch.float32).to(self.device)
                src_intervals[0] = -1
            else:
                src_stations = torch.tensor(self.list_squeeze(self.data[idx]["stationId"][:-1]), dtype=torch.int64).to(self.device)
                travel_index = torch.tensor(self.list_squeeze(self.data[idx]["travel_index"][:-1]), dtype=torch.int64).to(self.device)
                travel_mask = torch.tensor(self.list_squeeze(self.data[idx]["travel_mask"][:-1]), dtype=torch.int64).to(self.device)
                src_DoW = torch.tensor(self.list_squeeze(self.data[idx]["DoW"][:-1]), dtype=torch.int32).to(self.device)
                src_HoD = torch.tensor(self.list_squeeze(self.data[idx]["HoD"][:-1]), dtype=torch.int32).to(self.device)
                src_intervals = torch.tensor(self.list_squeeze(self.data[idx]["intervals"][:-1]), dtype=torch.float32).to(self.device)
                src_intervals[0] = -1
             
        mask = (src_intervals != 0.0)
        src_intervals = src_intervals[mask]
        # src_intervals = exp_normalize(src_intervals, torch.tensor(0.011146042964972321).cuda(), self.mean, self.std)
        src_intervals = normalize(src_intervals, self.mean, self.std)
        # src_intervals = MinMaxNormalize(src_intervals, self.mean, self.std)
        src_intervals[0] = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        src_stations = src_stations[mask]
        travel_index = travel_index[mask]
        travel_mask = travel_mask[mask]
        # src_DoW = src_DoW[mask]
        # src_HoD = src_HoD[mask]
        
        
        src = self.tokenizer.tokenize(src_stations[:-1])
        travel_index = self.tokenizer.tokenize(travel_index[:-1])
        travel_index[0] = torch.tensor(0).to(self.device)
        
        src_travel_mask = self.tokenizer.tokenize(travel_mask[:-1])
        src_travel_mask[0] = torch.tensor(0).to(self.device)
        
        tgt_travel_mask = self.tokenizer.tokenize(travel_mask)[1:]
        tgt_travel_mask = F.pad(tgt_travel_mask, (0, 1), value = 0)
        
        travel_mask = src_travel_mask & tgt_travel_mask
        
        intervals = self.tokenizer.time_tokenize(src_intervals, self.tokenizer.mask_index, True)
            
            
        tgt = self.tokenizer.tokenize(src_stations)[1:]
        travel_index = self.tokenizer.tokenize(travel_index)[1:]
        
        tgt = F.pad(tgt, (0, 1), value = 0)
        travel_index = F.pad(travel_index, (0, 1), value = 0)
        res = dict()
        res["src"] = src
        if self.is_base_history_eval:
            next_travel = mask[-next_travel_len:]
            next_travel = next_travel[next_travel == True]
            
            pre_travel = mask[:-next_travel_len]
            pre_travel = pre_travel[pre_travel == True]
            
            pre_index = torch.concat([~pre_travel, next_travel], 0)
            pre_index = self.tokenizer.time_tokenize(pre_index, False)[1:]
            pre_index = F.pad(pre_index, (0, 1), value = False)
        else:
            pre_index = mask[mask == True]
            pre_index = self.tokenizer.time_tokenize(pre_index, False)[1:]
            pre_index = F.pad(pre_index, (0, 1), value = False)
            
        res["tgt"] = tgt
        res["intervals"] = intervals
        res["tgt_intervals"] = intervals
        res["pre_index"] = pre_index
        
        res["travel_index"] = travel_index
        res["travel_mask"] = travel_mask
        
        res["enc_src"] = torch.tensor(0).to(self.device)
        res["enc_DoW"] = torch.tensor(0).to(self.device)
        res["enc_HoD"] = torch.tensor(0).to(self.device)
        res["enc_intervals"] = torch.tensor(0).to(self.device)
        return res