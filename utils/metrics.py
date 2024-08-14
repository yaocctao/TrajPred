import torch

def Accuracy(pred, label):
    corrects = (pred == label).sum().float()
    total = len(label)
    
    return (corrects/total).cpu().detach().data.numpy()

def TimeAccuracy(pred, label, threshold = 60):
    time_corrects = (torch.abs(pred - label) < threshold).sum().float()
    total = len(label)
    
    return (time_corrects/total).cpu().detach().data.numpy()

def time_station_Accuracy(time_pred, time_label, station_pred, station_label):
    station_corrects = (station_pred == station_label)
    total = len(station_label)
     
    time_corrects = (torch.abs(time_pred - time_label) < 60)
    
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


def normalize(data, mean = 465.5223178538118, std = 959.7372163727321):
    return (data - mean) / std

def unnormalize(data, mean = 465.5223178538118, std = 959.7372163727321):
    return data * std + mean