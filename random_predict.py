import os
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm
from utils.config import read_config
from utils.metrics import Accuracy, TimeAccuracy, mae_loss, time_station_Accuracy


class RandomDataset():
    def __init__(self, data, device, is_base_history_eval=False):
        self.device =device
        self.is_base_history_eval = is_base_history_eval
        self.data = data
    
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
                src_intervals = torch.tensor(self.list_squeeze(self.data[idx]["intervals"]), dtype=torch.float32).to(self.device)
                src_intervals[0] = -1
                travel_mask = torch.tensor(self.list_squeeze(self.data[idx]["travel_mask"]), dtype=torch.int32).to(self.device)
                
                next_travel_len =  len(src_stations) - len(torch.tensor(self.list_squeeze(self.data[idx]["stationId"][:-1]), dtype=torch.int64).to(self.device))

        else:
            if len(self.data[idx]["stationId"]) == 1:
                src_stations = torch.tensor(self.list_squeeze(self.data[idx]["stationId"]), dtype=torch.int64).to(self.device)
                src_intervals = torch.tensor(self.list_squeeze(self.data[idx]["intervals"]), dtype=torch.float32).to(self.device)
                src_intervals[0] = -1
                travel_mask = torch.tensor(self.list_squeeze(self.data[idx]["travel_mask"]), dtype=torch.int32).to(self.device)
            else:
                src_stations = torch.tensor(self.list_squeeze(self.data[idx]["stationId"][:-1]), dtype=torch.int64).to(self.device)
                src_intervals = torch.tensor(self.list_squeeze(self.data[idx]["intervals"][:-1]), dtype=torch.float32).to(self.device)
                travel_mask = torch.tensor(self.list_squeeze(self.data[idx]["travel_mask"][:-1]), dtype=torch.int32).to(self.device)
                src_intervals[0] = -1
             
        mask = (src_intervals != 0.0)
        src_intervals = src_intervals[mask]
        src_intervals[0] = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        src_stations = src_stations[mask]
        travel_mask = travel_mask[mask]
        
        src = src_stations[:-1]
        src = F.pad(src, (1, 0), value = 0)
        
        intervals = src_intervals
        
        src_travel_mask = travel_mask[:-1]
        src_travel_mask = F.pad(src_travel_mask, (1, 0), value = 0)
        
        tgt_travel_mask = travel_mask
        
        travel_mask = src_travel_mask & tgt_travel_mask    
            
        tgt = src_stations
        res = dict()
        res["src"] = src.numpy().tolist()
        if self.is_base_history_eval:
            next_travel = mask[-next_travel_len:]
            next_travel = next_travel[next_travel == True]
            
            pre_travel = mask[:-next_travel_len]
            pre_travel = pre_travel[pre_travel == True]
            
            pre_index = torch.concat([~pre_travel, next_travel], 0)
        else:
            pre_index = mask[mask == True]
            
        res["tgt"] = tgt.numpy().tolist()
        res["intervals"] = intervals.numpy().tolist()
        res["tgt_intervals"] = intervals.numpy().tolist()
        res["pre_index"] = pre_index.numpy().tolist()
        res["travel_mask"] = travel_mask.numpy().tolist()
        
        return res
    
    

class Random():
    
    def __init__(self, adj) -> None:
        self.adj = adj.copy()
        self.adj.index = self.adj['from_id']
        #索引排序
        self.adj.sort_index(inplace=True)
        
        self._adj = adj
        self._adj.index = self._adj['to_id']
        self._adj.sort_index(inplace=True)
    

    
    def predict(self, src, intervals):
        station_predictions = []
        intervals_predictions = []
        
        for i in range(len(src)):
            if i < 2:
                station_predictions.append(src[i])
                intervals_predictions.append(intervals[i])
                continue
            last_station = src[i-1]
            last_intervals = intervals[i-1]
            current_station = src[i]
            if current_station not in self.adj.index:
                station_predictions.append(0)    
                intervals_predictions.append(0)
                continue
            predict = self.adj.loc[current_station]
            if len(predict.shape) != 1:
                predict = predict.sample(1).iloc[0]
                
            predict_station = predict['to_id']
            if last_station not in self._adj.index:
                station_predictions.append(predict_station)    
                intervals_predictions.append(0)
                continue
            last_info = self._adj.loc[last_station]
            if len(last_info.shape) != 1:
                last_info = last_info.sample(1).iloc[0]
                
            last_distance = last_info['distance']
            speed = last_distance/last_intervals
            distance = predict['distance']
            if speed == 0:
                interval = 0
            else:
                interval = distance/speed
            station_predictions.append(predict_station)
            intervals_predictions.append(interval)
        
        return station_predictions, intervals_predictions
    
def predict(adj_path,dataset):
    adj = pd.read_csv(adj_path)
    random = Random(adj)
    
    predictions = torch.tensor([]).to(dataset.device)
    labels = torch.tensor([]).to(dataset.device)
    predictions_intervals = torch.tensor([]).to(dataset.device)
    labels_intervals = torch.tensor([]).to(dataset.device)
    travel_masks = torch.tensor([], dtype=torch.bool).to(dataset.device)
    for data in tqdm(dataset):
        src = data['src']
        tgt = data['tgt']
        intervals = data['intervals']
        tgt_intervals = data['tgt_intervals']
        pred_index = data['pre_index']
        travel_mask = data['travel_mask']
        
        station_predictions, intervals_predictions = random.predict(src, intervals)
        
        predictions = torch.cat((predictions, torch.tensor(station_predictions)[pred_index]), 0)
        labels = torch.cat((labels, torch.tensor(tgt)[pred_index]), 0)
        predictions_intervals = torch.cat((predictions_intervals, torch.tensor(intervals_predictions)[pred_index]), 0)
        labels_intervals = torch.cat((labels_intervals, torch.tensor(tgt_intervals)[pred_index]), 0)
        travel_masks = torch.cat((travel_masks, torch.tensor(travel_mask)[pred_index]), 0)
    time_station_acc = time_station_Accuracy(predictions_intervals.flatten(), labels_intervals.flatten(), predictions, labels, 1.5)
    acc = Accuracy(predictions, labels)
    time_acc = TimeAccuracy(predictions_intervals.flatten(), labels_intervals.flatten(), 1.5)
    mae = mae_loss(predictions_intervals/60, labels_intervals/60)
    
    
    en_time_station_acc = time_station_Accuracy(predictions_intervals.flatten()[travel_masks], labels_intervals.flatten()[travel_masks], predictions[travel_masks], labels[travel_masks], 60)
    en_acc = Accuracy(predictions[travel_masks], labels[travel_masks])
    en_time_acc = TimeAccuracy(predictions_intervals.flatten()[travel_masks], labels_intervals.flatten()[travel_masks], 60)
    
    return acc, time_acc, time_station_acc, mae, en_time_station_acc, en_acc, en_time_acc
        
        

if __name__ == "__main__":
    path = "./config/LSTM.ini"
    conf = read_config(path)
    path = conf["DATASET"]["path"]
    device = conf["DATASET"]["device"]
    mean = float(conf["DATASET"]["mean"])
    std = float(conf["DATASET"]["std"])
    data_index_path = os.path.join(path,"train_dev_test.npz")
    data_path = os.path.join(path, "data.npz")
    data = np.load(data_path, allow_pickle=True)["data"]
    data_index = np.load(data_index_path)

    train_data = data[data_index["train"]][:1000]
    dev_data = data[data_index["dev"]][:1000]
    test_data = data[data_index["test"]][:1000]
    history_dataset = RandomDataset(train_data, "cpu", True)
    dev_dataset = RandomDataset(dev_data, "cpu", True)
    test_dataset = RandomDataset(test_data, "cpu", True)
    
    topo_path = os.path.join(path, 'topo.csv')
    
    eval_acc, eval_time_acc, eval_time_station_acc, eval_mae, eval_en_time_station_acc, eval_en_acc, eval_en_time_acc = predict(topo_path, dev_dataset)
    print(f"\nEval time_station_acc:{round(eval_time_station_acc * 100, 5)}, time_acc:{round(eval_time_acc * 100, 5)}, station_acc:{round(eval_acc * 100, 5)}, mae:{str(float(eval_mae))} \
            en_time_station_acc:{round(eval_en_time_station_acc * 100, 5)}, en_time_acc:{round(eval_en_time_acc * 100, 5)}, en_station_acc:{round(eval_en_acc * 100, 5)}")
    test_acc, test_time_acc, test_time_station_acc, test_mae, test_en_time_station_acc, test_en_acc, test_en_time_acc = predict(topo_path, test_dataset)
    history_acc, history_time_acc, history_time_station_acc, history_mae, history_en_time_station_acc, history_en_acc, history_en_time_acc = predict(topo_path, history_dataset)

    
    print(f"\nEval time_station_acc:{round(eval_time_station_acc * 100, 5)}, time_acc:{round(eval_time_acc * 100, 5)}, station_acc:{round(eval_acc * 100, 5)}, mae:{str(float(eval_mae))} \
            en_time_station_acc:{round(eval_en_time_station_acc * 100, 5)}, en_time_acc:{round(eval_en_time_acc * 100, 5)}, en_station_acc:{round(eval_en_acc * 100, 5)} \
            \nTest time_station_acc:{round(test_time_station_acc * 100, 5)}, time_acc:{round(test_time_acc * 100, 5)}, station_acc:{round(test_acc * 100, 5)}, mae:{str(float(test_mae))} \
            en_time_station_acc:{round(test_en_time_station_acc * 100, 5)}, en_time_acc:{round(test_en_time_acc * 100, 5)}, en_station_acc:{round(test_en_acc * 100, 5)} \
            \nhistory time_station_acc:{round(history_time_station_acc * 100, 5)}, time_acc:{round(history_time_acc * 100, 5)}, station_acc:{round(history_acc * 100, 5)}, mae:{str(float(history_mae))}\
            en_time_station_acc:{round(history_en_time_station_acc * 100, 5)}, en_time_acc:{round(history_en_time_acc * 100, 5)}, en_station_acc:{round(history_en_acc * 100, 5)}"
            )