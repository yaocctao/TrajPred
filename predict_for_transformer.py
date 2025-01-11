import os
import random
import time
import numpy as np
import torch
from models.transformer import TrajPredTransformerV3,TrajPredTransformerV1
from utils.config import read_config
from utils.metrics import Accuracy, TimeAccuracy, mae_loss, time_station_Accuracy, unnormalize, save_predictions
from utils.standardization import exp_unnormalize, exp_normalize
from models.tokenizer import Trajtokenizer
from evaluate import extract_single_data
from data.load_data import *

def TrajPredTransformerV1_evaluate(model, loader, mean, std):
    model.eval()
    travel_predictsions = []
    travel_predprobs = []
    travel_labels = []
    travel_intervals_predictions = []
    travel_intervals_labels = []
    with torch.no_grad():
        predictions = torch.tensor([]).to(loader.dataset.device)
        labels = torch.tensor([]).to(loader.dataset.device)
        predictions_intervals = torch.tensor([]).to(loader.dataset.device)
        labels_intervals = torch.tensor([]).to(loader.dataset.device)
        travel_masks = torch.tensor([], dtype=torch.bool).to(loader.dataset.device)
        for _, batch in tqdm(enumerate(loader)):
            if batch is None:
                continue
            src = batch['src']
            enc_src = batch['enc_src']
            tgt = batch['tgt']
            travel_mask = batch['travel_mask'].bool()
            travel_index = batch['travel_index']
            enc_DoW = batch['enc_DoW']
            enc_HoD = batch['enc_HoD']
            intervals = batch['intervals']
            intervals_tgt = intervals.clone()
            
            pred_index = batch['pre_index']
            travel_masks = torch.cat((travel_masks, travel_mask[pred_index]), 0)

            enc_intervals = batch['enc_intervals']
            output, intervals_output =model(enc_src, enc_intervals, src, enc_HoD, enc_DoW, intervals, travel_index)
            # output, intervals_output = torch.tensor(20.), torch.tensor(20.)
            # intervals_output = unnormalize(intervals_output, mean, std)
            intervals_output = exp_unnormalize(intervals_output, 0.0055730214824861605, mean, std)
            # intervals_tgt = unnormalize(intervals_tgt, mean, std)
            intervals_tgt = exp_unnormalize(intervals_tgt, 0.0055730214824861605, mean, std)

            #超出长度导致和easyTemporalPointProcess的结果不一致
            # if 0 in tgt[pred_index]:
            #     print("intervals_output:", tgt[pred_index])
            # count = 0
            # for i in range(len(pred_index)):
            #     count += int((pred_index[i] == True).sum())
            #     if count > 2177:
            #         print(i)
            #         break
            predictions = torch.cat((predictions, torch.argmax(output, -1)[pred_index]), 0)
            labels = torch.cat((labels, tgt[pred_index]), 0)
            predictions_intervals = torch.cat((predictions_intervals, intervals_output[pred_index]), 0)
            labels_intervals = torch.cat((labels_intervals, intervals_tgt[pred_index]), 0)
            
            travel_predictsion, travel_label = extract_single_data(torch.argmax(output, -1), tgt, pred_index)
            predprobs = extract_single_data(output, None, pred_index)
            travel_intervals_prediction, travel_intervals_label = extract_single_data(intervals_output, intervals_tgt, pred_index)
            
            travel_predictsions.extend(travel_predictsion)
            travel_predprobs.extend(predprobs)
            travel_labels.extend(travel_label)
            travel_intervals_predictions.extend(travel_intervals_prediction)
            travel_intervals_labels.extend(travel_intervals_label)
            
        save_predictions(travel_predictsions, travel_labels, travel_intervals_predictions, travel_intervals_labels, "data/predictions/TrajPredTransformerV1", travel_predprobs)
            
        time_station_acc = time_station_Accuracy(predictions_intervals.flatten(), labels_intervals.flatten(), predictions, labels, 1.5)
        acc = Accuracy(predictions, labels)
        time_acc = TimeAccuracy(predictions_intervals.flatten(), labels_intervals.flatten(), 1.5)
        mae = mae_loss(predictions_intervals/60, labels_intervals/60)
        
        
        en_time_station_acc = time_station_Accuracy(predictions_intervals.flatten()[travel_masks], labels_intervals.flatten()[travel_masks], predictions[travel_masks], labels[travel_masks], 60)
        en_acc = Accuracy(predictions[travel_masks], labels[travel_masks])
        en_time_acc = TimeAccuracy(predictions_intervals.flatten()[travel_masks], labels_intervals.flatten()[travel_masks], 60)
    model.train()
    return acc, time_acc, time_station_acc, mae, en_time_station_acc, en_acc, en_time_acc

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    



if __name__ == "__main__":
    random_seed(42)
    path = "./config/Transformer.ini"
    conf = read_config(path)
    trajTokenizer = Trajtokenizer(conf)
    model_conf = read_config(conf["TRAIN"]["model_config_path"])
    model = TrajPredTransformerV1(model_conf["MODEL"])
    model.to(conf["DATASET"]["device"])
    model_path = os.path.join(conf['TRAIN']['save_path'], "best_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    mean = float(conf["DATASET"]["mean"])
    std = float(conf["DATASET"]["std"])
    load_data = eval(model_conf["MODEL"]["load_data_method"])
    train_loader, dev_loader, test_loader, history_loader, train_num = load_data(conf)
    acc, time_acc, time_station_acc, mae, en_time_station_acc, en_acc, en_time_acc = TrajPredTransformerV1_evaluate(model, test_loader, mean, std)