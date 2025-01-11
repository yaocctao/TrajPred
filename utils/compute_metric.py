import json
import pickle
import os,sys
import pandas as pd
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import Accuracy, TimeAccuracy, mae_loss, mape, rmse,time_station_Accuracy
import numpy as np


def read_res(dir):
    predictions_path = os.path.join(dir, "predictions.npy")
    label_path = os.path.join(dir, "labels.npy")
    predictions = np.load(predictions_path, allow_pickle=True)
    travel_predictions = predictions.item()['travel_predictions']
    travel_intervals_predictions = predictions.item()['travel_intervals_predictions']
    
    labels = np.load(label_path, allow_pickle=True)
    travel_labels = labels.item()['travel_labels']
    travel_intervals_labels = labels.item()['travel_intervals_labels']

    
    return travel_predictions, travel_labels, travel_intervals_predictions, travel_intervals_labels

def read_id2station(dir = "data/trajFujianV2/stations2id.json"):
    with open(dir, 'r') as f:
        topo = json.load(f)
    stations = pd.read_csv("data/etc_map_topo.csv")
    stations = stations[stations['is_deleted'] == 0]
    filter_list = set()
    for _, row in stations.iterrows():
        if row['from_type'] == 3 and row['from_virtual'] == 0:
            filter_list.add(row['from_id'])
        if row['to_type'] == 3 and row['to_virtual'] == 0:
            filter_list.add(row['to_id'])
    enstationids = []
    for k, v in topo.items():
        if k.endswith('EN') or k in filter_list:
            enstationids.append(v)
    return enstationids



def filter_data(data, enstationids):
    enstationids = torch.tensor(enstationids)
    mask = torch.isin(data, enstationids)
    return mask

def compute_all_metrics(new_travel_predictions, new_travel_labels, new_travel_intervals_predictions, new_travel_intervals_labels, threshold = 1.5, is_en = False):
    print("len: ",len(new_travel_predictions))
    time_station_acc = time_station_Accuracy(new_travel_intervals_predictions, new_travel_intervals_labels, new_travel_predictions, new_travel_labels, threshold = threshold)
    acc = Accuracy(new_travel_predictions, new_travel_labels)
    time_acc = TimeAccuracy(new_travel_intervals_predictions, new_travel_intervals_labels, threshold = threshold)
    if is_en:
        divide = 60
    else:
        divide = 1
    mae = mae_loss(new_travel_intervals_predictions/divide, new_travel_intervals_labels/divide)
    mape_ = mape(new_travel_intervals_predictions/divide, new_travel_intervals_labels/divide)
    rmse_ = rmse(new_travel_intervals_predictions/divide, new_travel_intervals_labels/divide)
    # print("acc:", acc)
    # print("time_acc:", time_acc)
    # print("time_station_acc:", time_station_acc)
    # print("mae:", float(mae))
    # print("mape:", float(mape_))
    # print("rmse:", float(rmse_))
    
    return acc, time_acc, time_station_acc, mae, mape_, rmse_

def main(path = "data/predictions/TrajPredTransformerV1", type = "TrajPredTransformer"):
    #EasyTemporalPointProcess 在预测的时候返回的结果少一第一个位置的数据
    if type == "EasyTemporalPointProcess":
        with open(path, 'rb') as file:
            try:
                data = pickle.load(file, encoding='latin-1')
            except Exception:
                data = pickle.load(file)
        new_travel_predictions = []
        new_travel_labels = []
        new_travel_intervals_predictions = []
        new_travel_intervals_labels = []
        # for i in range(len(data['pred'])):
        mask = (data['label'][1][:,30:254] != 0.0)
        #优于our的方法最大长度256，为了保证结果一致，取最后的225个数据, 30:254后面用254是因为起点是30，不是后面的31
        new_travel_predictions.extend(data['pred'][1][:,30:254][mask])
        new_travel_labels.extend(data['label'][1][:,30:254][mask])
        new_travel_intervals_predictions.extend(data['pred'][0][:,30:254][mask])
        new_travel_intervals_labels.extend(data['label'][0][:,30:254][mask])
    elif type == "based-transformer":
        new_travel_predictions, new_travel_labels, new_travel_intervals_predictions, new_travel_intervals_labels = read_res(path)
    else:
        travel_predictions, travel_labels, travel_intervals_predictions, travel_intervals_labels = read_res(path)
        new_travel_predictions = []
        new_travel_labels = []
        new_travel_intervals_predictions = []
        new_travel_intervals_labels = []
        count = 0
        for i in range(len(travel_intervals_labels)):
            length = len(travel_intervals_labels[i])
            if length >= 31:
                new_travel_predictions.extend(travel_predictions[i][31:255])
                new_travel_labels.extend(travel_labels[i][31:255])
                new_travel_intervals_predictions.extend(travel_intervals_predictions[i][31:255])
                new_travel_intervals_labels.extend(travel_intervals_labels[i][31:255])
            else:
                count += 1
        # print("count:", count)

    new_travel_predictions = torch.tensor(new_travel_predictions)
    new_travel_labels = torch.tensor(new_travel_labels)
    new_travel_intervals_predictions = torch.tensor(new_travel_intervals_predictions).squeeze(-1)
    new_travel_intervals_labels = torch.tensor(new_travel_intervals_labels).squeeze(-1)
    #filter data
    enstationids = read_id2station()
    enstationids = filter_data(new_travel_labels, enstationids)
    
    print('-'*25, path, '-'*25)
    en_travel_predictions = new_travel_predictions[enstationids]
    en_travel_labels = new_travel_labels[enstationids]
    en_travel_intervals_predictions = new_travel_intervals_predictions[enstationids]
    en_travel_intervals_labels = new_travel_intervals_labels[enstationids]
    # print("en metrics:")
    acc, time_acc, time_station_acc, mae, mape_, rmse_ = compute_all_metrics(en_travel_predictions, en_travel_labels, en_travel_intervals_predictions, en_travel_intervals_labels, threshold = 60*5, is_en = True)
    en_res = {'station_acc': acc*100, 'time_acc': time_acc*100, 'time_station_acc': time_station_acc*100, 'mae': float(mae), 'mape': float(mape_), 'rmse':float( rmse_)}
    # print()
    
    # print("gantry metrics:")
    gantry_travel_predictions = new_travel_predictions[~enstationids]
    gantry_travel_labels = new_travel_labels[~enstationids]
    gantry_travel_intervals_predictions = new_travel_intervals_predictions[~enstationids]
    gantry_travel_intervals_labels = new_travel_intervals_labels[~enstationids]
    acc, time_acc, time_station_acc, mae, mape_, rmse_ = compute_all_metrics(gantry_travel_predictions, gantry_travel_labels, gantry_travel_intervals_predictions, gantry_travel_intervals_labels)
    gantry_res = {'station_acc': acc*100, 'time_acc': time_acc*100, 'time_station_acc': time_station_acc*100, 'mae': float(mae), 'mape': float(mape_), 'rmse':float( rmse_)}
    # print('-'*50)
    
    return en_res, gantry_res

def add_dict(res, name, **kwargs):
    res['backbone'].append(name)
    for key, value in kwargs.items():
        res[key].append(value)
    return res

if __name__ == '__main__':
    en_all_res = {'backbone': [], 'station_acc': [], 'time_acc': [], 'time_station_acc':[], 'mae': [], 'mape': [], 'rmse': []}
    gantry_all_res = {'backbone': [], 'station_acc': [], 'time_acc': [], 'time_station_acc':[], 'mae': [], 'mape': [], 'rmse': []}
    
    en_res, gantry_res = main('/mnt/yaocctao/trajPred/data/predictions/AttNHP/pred.pkl','EasyTemporalPointProcess')
    en_all_res = add_dict(en_all_res, 'AttNHP', **en_res)
    gantry_all_res = add_dict(gantry_all_res, 'AttNHP', **gantry_res)
    
    en_res, gantry_res = main('/mnt/yaocctao/trajPred/data/predictions/IntensityFree/pred.pkl','EasyTemporalPointProcess')
    en_all_res = add_dict(en_all_res, 'IntensityFree', **en_res)
    gantry_all_res = add_dict(gantry_all_res, 'IntensityFree', **gantry_res)
    
    en_res, gantry_res = main('/mnt/yaocctao/trajPred/data/predictions/iTransformer','based-transformer')
    en_all_res = add_dict(en_all_res, 'iTransformer', **en_res)
    gantry_all_res = add_dict(gantry_all_res, 'iTransformer', **gantry_res)
    
    en_res, gantry_res = main('/mnt/yaocctao/trajPred/data/predictions/Pyraformer','based-transformer')
    en_all_res = add_dict(en_all_res, 'Pyraformer', **en_res)
    gantry_all_res = add_dict(gantry_all_res, 'Pyraformer', **gantry_res)
    
    en_res, gantry_res = main('/mnt/yaocctao/trajPred/data/predictions/trajLSTM','based-transformer')
    en_all_res = add_dict(en_all_res, 'trajLSTM', **en_res)
    gantry_all_res = add_dict(gantry_all_res, 'trajLSTM', **gantry_res)
    
    en_res, gantry_res = main('/mnt/yaocctao/trajPred/data/predictions/Transformer','based-transformer')
    en_all_res = add_dict(en_all_res, 'Transformer', **en_res)
    gantry_all_res = add_dict(gantry_all_res, 'Transformer', **gantry_res)
    
    en_res, gantry_res = main('data/predictions/TrajPredTransformerV1','TrajPredTransformer')
    en_all_res = add_dict(en_all_res, 'TrajPredTransformerV1(ours)', **en_res)
    gantry_all_res = add_dict(gantry_all_res, 'TrajPredTransformerV1(ours)', **gantry_res)
    
    print("en metrics:")
    en_res = pd.DataFrame(en_all_res)
    print(en_res)
    print('-'*50)
    print("gantry metrics:")
    gantry_res = pd.DataFrame(gantry_all_res)
    print(gantry_res)