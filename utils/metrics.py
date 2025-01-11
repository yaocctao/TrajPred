import torch, os
import numpy as np

def Accuracy(pred, label):
    corrects = (pred == label).sum().float()
    total = len(label)
    
    return (corrects/total).cpu().detach().data.numpy()

def TimeAccuracy(pred, label, threshold = 60):
    time_corrects = (torch.abs(pred - label) < threshold).sum().float()
    total = len(label)
    
    return (time_corrects/total).cpu().detach().data.numpy()

def time_station_Accuracy(time_pred, time_label, station_pred, station_label, threshold = 60):
    station_corrects = (station_pred == station_label)
    total = len(station_label)
     
    time_corrects = (torch.abs(time_pred - time_label) < threshold)
    
    #station_corrects和time_corrects都为True的有多少个
    corrects = (station_corrects & time_corrects).sum().float()
    
    return (corrects/total).cpu().detach().data.numpy()

def mae_loss(predicted, observed, null_val=0.0):
    mask = (observed != null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(predicted - observed)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def mape(predicted, observed, null_val=0.0):
    """
    计算 MAPE (Mean Absolute Percentage Error)
    """
    mask = (observed != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    # 计算绝对百分比误差
    loss = torch.abs((predicted - observed) / observed)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss) * 100  # 返回百分比

def rmse(predicted, observed, null_val=0.0):
    """
    计算 RMSE (Root Mean Square Error)
    """
    mask = (observed != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    # 计算平方误差
    loss = (predicted - observed) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.sqrt(torch.mean(loss))  # 返回平方根的均值

def normalize(data, mean = 465.5223178538118, std = 959.7372163727321):
    return (data - mean) / std

def unnormalize(data, mean = 465.5223178538118, std = 959.7372163727321):
    return data * std + mean

def MinMaxNormalize(data, min_val = 0, max_val = 1):
    return (data - min_val) / (max_val - min_val)

def MinMaxUnnormalize(data, min_val = 0, max_val = 1):
    return data * (max_val - min_val) + min_val

def save_predictions(travel_predictsions, travel_labels, travel_intervals_predictions, travel_intervals_labels, path, travel_predprobs = None):
    predictions = {
        "travel_predictions":travel_predictsions,
        "travel_intervals_predictions":travel_intervals_predictions,
        }
    predictions_save_path = os.path.join(path, "predictions.npy")
    np.save(predictions_save_path, predictions)
    
    labels = {
        "travel_labels":travel_labels,
        "travel_intervals_labels":travel_intervals_labels,
    }
    labels_save_path = os.path.join(path, "labels.npy")
    np.save(labels_save_path, labels)
    
    if travel_predprobs is not None:
        predprobs = {
            "travel_predprobs":travel_predprobs,
        }
        predprobs_save_path = os.path.join(path, "predprobs.npy")
        np.save(predprobs_save_path, predprobs)