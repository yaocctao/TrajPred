import numpy as np
import torch
from utils.metrics import Accuracy, TimeAccuracy, mae_loss, time_station_Accuracy, unnormalize, MinMaxUnnormalize, save_predictions
from utils.standardization import exp_unnormalize
from tqdm import tqdm

# def TrajPredTransformerV1_evaluate(model, loader, mean, std):
#     model.eval()
#     with torch.no_grad():
#         predictions = torch.tensor([]).to(loader.dataset.device)
#         labels = torch.tensor([]).to(loader.dataset.device)
#         predictions_intervals = torch.tensor([]).to(loader.dataset.device)
#         labels_intervals = torch.tensor([]).to(loader.dataset.device)
#         travel_masks = torch.tensor([], dtype=torch.bool).to(loader.dataset.device)
#         for _, batch in tqdm(enumerate(loader)):
#             if batch is None:
#                 continue
#             src = batch['src']
#             enc_src = batch['enc_src']
#             tgt = batch['tgt']
#             travel_mask = batch['travel_mask'].bool()
#             travel_index = batch['travel_index']
#             enc_DoW = batch['enc_DoW']
#             enc_HoD = batch['enc_HoD']
#             intervals = batch['intervals']
#             intervals_tgt = intervals.clone()
            
#             pred_index = batch['pre_index']
#             travel_masks = torch.cat((travel_masks, travel_mask[pred_index]), 0)

#             enc_intervals = batch['enc_intervals']
#             output, intervals_output =model(enc_src, enc_intervals, src, enc_HoD, enc_DoW, intervals, travel_index)
#             # intervals_output = unnormalize(intervals_output, mean, std)
#             intervals_output = exp_unnormalize(intervals_output, 0.0055730214824861605, mean, std)
#             # intervals_tgt = unnormalize(intervals_tgt, mean, std)
#             intervals_tgt = exp_unnormalize(intervals_tgt, 0.0055730214824861605, mean, std)

#             predictions = torch.cat((predictions, torch.argmax(output, -1)[pred_index]), 0)
#             labels = torch.cat((labels, tgt[pred_index]), 0)
#             predictions_intervals = torch.cat((predictions_intervals, intervals_output[pred_index]), 0)
#             labels_intervals = torch.cat((labels_intervals, intervals_tgt[pred_index]), 0)
#         time_station_acc = time_station_Accuracy(predictions_intervals.flatten(), labels_intervals.flatten(), predictions, labels, 1.5)
#         acc = Accuracy(predictions, labels)
#         time_acc = TimeAccuracy(predictions_intervals.flatten(), labels_intervals.flatten(), 1.5)
#         mae = mae_loss(predictions_intervals/60, labels_intervals/60)
        
        
#         en_time_station_acc = time_station_Accuracy(predictions_intervals.flatten()[travel_masks], labels_intervals.flatten()[travel_masks], predictions[travel_masks], labels[travel_masks], 60)
#         en_acc = Accuracy(predictions[travel_masks], labels[travel_masks])
#         en_time_acc = TimeAccuracy(predictions_intervals.flatten()[travel_masks], labels_intervals.flatten()[travel_masks], 60)
#     model.train()
#     return acc, time_acc, time_station_acc, mae, en_time_station_acc, en_acc, en_time_acc

def extract_single_data(predictions, labels = None, predict_index = None):
    new_predicttions = []
    new_labels = []
    if labels is None:
        for i, index in enumerate(predict_index):
            new_predicttions.append(predictions[i][index].detach().cpu().numpy().tolist())
        
        return new_predicttions
    else:
        for i, index in enumerate(predict_index):
            new_predicttions.append(list(predictions[i][index].detach().cpu().numpy()))
            new_labels.append(list(labels[i][index].detach().cpu().numpy()))
        
        return new_predicttions, new_labels

def TrajPredTransformerV1_evaluate(model, loader, mean, std):
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
            # intervals_output = unnormalize(intervals_output, mean, std)
            intervals_output = exp_unnormalize(intervals_output, 0.0055730214824861605, mean, std)
            # intervals_tgt = unnormalize(intervals_tgt, mean, std)
            intervals_tgt = exp_unnormalize(intervals_tgt, 0.0055730214824861605, mean, std)

            predictions = torch.cat((predictions, torch.argmax(output, -1)[pred_index]), 0)
            labels = torch.cat((labels, tgt[pred_index]), 0)
            predictions_intervals = torch.cat((predictions_intervals, intervals_output[pred_index]), 0)
            labels_intervals = torch.cat((labels_intervals, intervals_tgt[pred_index]), 0)
            
            travel_predictsion, travel_label = extract_single_data(torch.argmax(output, -1), tgt, pred_index)
            travel_intervals_prediction, travel_intervals_label = extract_single_data(intervals_output, intervals_tgt, pred_index)
            
            travel_predictsions.extend(travel_predictsion)
            travel_labels.extend(travel_label)
            travel_intervals_predictions.extend(travel_intervals_prediction)
            travel_intervals_labels.extend(travel_intervals_label)
            
        # save_predictions(travel_predictsions, travel_labels, travel_intervals_predictions, travel_intervals_labels, "data/predictions")
            
        time_station_acc = time_station_Accuracy(predictions_intervals.flatten(), labels_intervals.flatten(), predictions, labels, 1.5)
        acc = Accuracy(predictions, labels)
        time_acc = TimeAccuracy(predictions_intervals.flatten(), labels_intervals.flatten(), 1.5)
        mae = mae_loss(predictions_intervals/60, labels_intervals/60)
        
        
        en_time_station_acc = time_station_Accuracy(predictions_intervals.flatten()[travel_masks], labels_intervals.flatten()[travel_masks], predictions[travel_masks], labels[travel_masks], 60)
        en_acc = Accuracy(predictions[travel_masks], labels[travel_masks])
        en_time_acc = TimeAccuracy(predictions_intervals.flatten()[travel_masks], labels_intervals.flatten()[travel_masks], 60)
    model.train()
    return acc, time_acc, time_station_acc, mae, en_time_station_acc, en_acc, en_time_acc

def TrajPredTransformerV2_evaluate(model, loader, mean, std):
    model.eval()
    with torch.no_grad():
        predictions = torch.tensor([]).to(loader.dataset.device)
        labels = torch.tensor([]).to(loader.dataset.device)
        predictions_intervals = torch.tensor([]).to(loader.dataset.device)
        labels_intervals = torch.tensor([]).to(loader.dataset.device)
        for _, batch in enumerate(loader):
            if batch is None:
                continue
            src = batch['src']
            enc_src = batch['enc_src']
            tgt = batch['tgt']
            enc_DoW = batch['enc_DoW']
            enc_HoD = batch['enc_HoD']
            travel_index = batch['travel_index']
            travel_mask = batch['travel_mask']
            intervals = batch['intervals']
            intervals_tgt = batch['intervals'].clone()
            
            pred_index = batch['pre_index']

            enc_intervals = batch['enc_intervals']
            output, intervals_output =model(enc_src, enc_intervals, src, enc_HoD, enc_DoW, travel_index, travel_mask, intervals, tgt)
            intervals_output = unnormalize(intervals_output, mean, std)
            # intervals_output = MinMaxUnnormalize(intervals_output, mean, std)
            intervals_tgt = unnormalize(intervals_tgt, mean, std)
            # intervals_tgt = MinMaxUnnormalize(intervals_tgt, mean, std)
            
            predictions = torch.cat((predictions, torch.argmax(output, -1)[pred_index]), 0)
            labels = torch.cat((labels, tgt[pred_index]), 0)
            predictions_intervals = torch.cat((predictions_intervals, intervals_output[pred_index]), 0)
            labels_intervals = torch.cat((labels_intervals, intervals_tgt[pred_index]), 0)
        time_station_acc = time_station_Accuracy(predictions_intervals.flatten(), labels_intervals.flatten(), predictions, labels, 5)
        acc = Accuracy(predictions, labels)
        time_acc = TimeAccuracy(predictions_intervals.flatten(), labels_intervals.flatten(), 5)
        mae = mae_loss(predictions_intervals/60, labels_intervals/60)
        
    model.train()
    return acc, time_acc, time_station_acc, mae

def TrajPredTransformerV3_evaluate(model, loader, mean, std):
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
            # intervals_output = unnormalize(intervals_output, mean, std)
            intervals_output = exp_unnormalize(intervals_output, 0.0055730214824861605, mean, std)
            # intervals_tgt = unnormalize(intervals_tgt, mean, std)
            intervals_tgt = exp_unnormalize(intervals_tgt, 0.0055730214824861605, mean, std)

            predictions = torch.cat((predictions, torch.argmax(output, -1)[pred_index]), 0)
            labels = torch.cat((labels, tgt[pred_index]), 0)
            predictions_intervals = torch.cat((predictions_intervals, intervals_output[pred_index]), 0)
            labels_intervals = torch.cat((labels_intervals, intervals_tgt[pred_index]), 0)
            
            travel_predictsion, travel_label = extract_single_data(torch.argmax(output, -1), tgt, pred_index)
            travel_intervals_prediction, travel_intervals_label = extract_single_data(intervals_output, intervals_tgt, pred_index)
            
            travel_predictsions.extend(travel_predictsion)
            travel_labels.extend(travel_label)
            travel_intervals_predictions.extend(travel_intervals_prediction)
            travel_intervals_labels.extend(travel_intervals_label)
            
        # save_predictions(travel_predictsions, travel_labels, travel_intervals_predictions, travel_intervals_labels, "data/predictions")
            
        time_station_acc = time_station_Accuracy(predictions_intervals.flatten(), labels_intervals.flatten(), predictions, labels, 1.5)
        acc = Accuracy(predictions, labels)
        time_acc = TimeAccuracy(predictions_intervals.flatten(), labels_intervals.flatten(), 1.5)
        mae = mae_loss(predictions_intervals/60, labels_intervals/60)
        
        
        en_time_station_acc = time_station_Accuracy(predictions_intervals.flatten()[travel_masks], labels_intervals.flatten()[travel_masks], predictions[travel_masks], labels[travel_masks], 60)
        en_acc = Accuracy(predictions[travel_masks], labels[travel_masks])
        en_time_acc = TimeAccuracy(predictions_intervals.flatten()[travel_masks], labels_intervals.flatten()[travel_masks], 60)
    model.train()
    return acc, time_acc, time_station_acc, mae, en_time_station_acc, en_acc, en_time_acc


def TrajLSTM_evaluate(model, loader, mean, std):
    model.eval()
    with torch.no_grad():
        predictions = torch.tensor([]).to(loader.dataset.device)
        labels = torch.tensor([]).to(loader.dataset.device)
        predictions_intervals = torch.tensor([]).to(loader.dataset.device)
        labels_intervals = torch.tensor([]).to(loader.dataset.device)
        for i, batch in enumerate(loader):
            src = batch['src']
            tgt = batch['tgt']
            
            intervals = batch['intervals']
            intervals_tgt = batch['tgt_intervals']
            
            # pred_index = batch['pre_index']
            
            output, intervals_output = model(src, intervals)
            intervals_output = unnormalize(intervals_output, mean, std)
            # intervals_output = exp_unnormalize(intervals_output)
            intervals_tgt = unnormalize(intervals_tgt, mean, std)
            # intervals_tgt = exp_unnormalize(intervals_tgt)
            
            predictions = torch.cat((predictions, torch.argmax(output, -1)), 0)
            labels = torch.cat((labels, tgt), 0)
            predictions_intervals = torch.cat((predictions_intervals, intervals_output), 0)
            labels_intervals = torch.cat((labels_intervals, intervals_tgt), 0)
        time_station_acc = time_station_Accuracy(predictions_intervals.flatten(), labels_intervals.flatten(), predictions, labels, 1.5)
        acc = Accuracy(predictions, labels)
        time_acc = TimeAccuracy(predictions_intervals.flatten(), labels_intervals.flatten(), 1.5)
        mae = mae_loss(predictions_intervals/60, labels_intervals/60)
    model.train()
    return acc, time_acc, time_station_acc, mae
