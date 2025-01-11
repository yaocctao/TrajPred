import os
import torch

from models.tokenizer import Trajtokenizer
from train import random_seed
from utils.config import read_config
from models.LSTM import TrajLSTM
from utils.config import read_config, print_config, load_to_dict_config
from data.load_data import *
from utils.metrics import Accuracy, time_station_Accuracy, TimeAccuracy, mae_loss, unnormalize, save_predictions
from evaluate import TrajLSTM_evaluate

def extract_single_data(predictions, labels, predict_index):
    new_predicttions = []
    new_labels = []
    for i, index in enumerate(predict_index):
        new_predicttions.append(list(predictions[i][index].detach().cpu().numpy()))
        new_labels.append(list(labels[i][index].detach().cpu().numpy()))
    
    return new_predicttions, new_labels

def TrajLSTM_evaluate(model, loader, mean, std):
    model.eval()
    travel_predictsions = []
    travel_labels = []
    travel_intervals_predictions = []
    travel_intervals_labels = []
    with torch.no_grad():
        predictions = torch.tensor([]).to(loader.dataset.device)
        labels = torch.tensor([]).to(loader.dataset.device)
        predictions_intervals = torch.tensor([]).to(loader.dataset.device)
        labels_intervals = torch.tensor([]).to(loader.dataset.device)
        for i, batch in tqdm(enumerate(loader)):
            src = batch['src']
            tgt = batch['tgt']
            
            intervals = batch['intervals']
            intervals_tgt = batch['tgt_intervals']
            
            # pred_index = batch['pre_index']
            
            output, intervals_output = model(src, intervals)
            intervals_output = unnormalize(intervals_output, mean, std)
            intervals_tgt = unnormalize(intervals_tgt, mean, std)
            
            predictions = torch.cat((predictions, torch.argmax(output, -1)), 0)
            labels = torch.cat((labels, tgt), 0)
            predictions_intervals = torch.cat((predictions_intervals, intervals_output), 0)
            labels_intervals = torch.cat((labels_intervals, intervals_tgt), 0)
            
        travel_predictsions.extend(list(predictions.squeeze(-1).detach().cpu().numpy()))
        travel_labels.extend(list(labels.squeeze(-1).detach().cpu().numpy()))
        travel_intervals_predictions.extend(list(predictions_intervals.squeeze(-1).detach().cpu().numpy()))
        travel_intervals_labels.extend(list(labels_intervals.squeeze(-1).detach().cpu().numpy()))
        
         # result save
        folder_path = './data/predictions/' + 'trajLSTM' + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        save_predictions(travel_predictsions, travel_labels, travel_intervals_predictions, travel_intervals_labels, folder_path)
        
        time_station_acc = time_station_Accuracy(predictions_intervals.flatten(), labels_intervals.flatten(), predictions, labels, 1.5)
        acc = Accuracy(predictions, labels)
        time_acc = TimeAccuracy(predictions_intervals.flatten(), labels_intervals.flatten(), 1.5)
        mae = mae_loss(predictions_intervals/60, labels_intervals/60)
    model.train()
    return acc, time_acc, time_station_acc, mae

random_seed(42)
path = "./config/LSTM.ini"
conf = read_config(path)
# tokenizer = Trajtokenizer(conf)
model_path = os.path.join(conf['TRAIN']['save_path'], "best_model.pth")
model_conf = read_config(conf["TRAIN"]["model_config_path"])
mean = float(conf["DATASET"]["mean"])
std = float(conf["DATASET"]["std"])
load_data = eval(model_conf["MODEL"]["load_data_method"])
test_loader = load_data(conf, True)
model = eval(conf["TRAIN"]["model_name"])(model_conf["MODEL"])
model.to(conf["DATASET"]["device"])
model.load_state_dict(torch.load(model_path))
model.eval()
acc, time_acc, time_station_acc, mae = TrajLSTM_evaluate(model, test_loader, mean, std)
print(acc)
print(time_acc)
print(time_station_acc)
print(mae)