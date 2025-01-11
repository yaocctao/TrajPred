import os
import random
import time
import numpy as np
import torch
from models.transformer import TrajPredTransformerV3,TrajPredTransformerV1
from utils.config import read_config
from utils.metrics import Accuracy, TimeAccuracy, mae_loss, time_station_Accuracy, unnormalize
from utils.standardization import exp_unnormalize, exp_normalize
from models.tokenizer import Trajtokenizer
from fastapi import FastAPI
import asyncio, uvicorn


app = FastAPI()

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    
random_seed(42)
path = "./config/Transformer.ini"
conf = read_config(path)
trajTokenizer = Trajtokenizer(conf)
model_conf = read_config(conf["TRAIN"]["model_config_path"])
model = TrajPredTransformerV1(model_conf["MODEL"])
model.to(conf["DATASET"]["device"])
model_path = os.path.join(conf['TRAIN']['save_path'], "best_model.pth")
model.load_state_dict(torch.load(model_path))
mean = float(conf["DATASET"]["mean"])
std = float(conf["DATASET"]["std"])
model.eval()


# '341309', '34130B', '34130D', '34130F', '35060D' 'xxxx'
# 2.8167e+01, 1.2217e+01, 5.6333e+00, 1.0667e+00, 5.7000e+00, 2.8333e-01
# data = {'src':['3804EN', '34060D', '35130F', '35130D', '351309', '351301', '340525', '7902EX', '7902EN', '340527', '35032B', '350329', '350327', '350325', '350323', '342701', '8004EN', '35052B', 
#                '350529', '350527', '350525', '350523', '341301', '341303', '341305', '34130B', '34130D', '34130F', '35060D', '3804EX', '6103EN', '35031D', '340415', '6104EX', '8004EN', '35052B', 
#                '350529', '350527', '350525', '350523', '341303', '341305', '341307', '341309', '34130B', '34130D', '34130F', '35060D', '3804EX', '3804EN', '34060D', '35130F', '35130D', '35130B', 
#                '351309', '351307', '351305', '351303', '351301','340523', '340525', '340527', '35032B', '350327', '350325', '350323', '342701', '8004EN', '35052B', '350529', '350527', '350525', 
#                '350523', '341301', '341303', '341305', '341307'], 'enc_src':['3804EN'], 'enc_DoW':[0], 'enc_HoD':[0],
#         'intervals':[0.0, 1.6667e+00, 5.9833e+00, 1.0167e+00, 2.3167e+01, 7.3017e+01,
#         1.5333e+01, 1.1500e+00, 8.1667e+00, 1.0000e+00, 6.7833e+00, 1.5000e+00,
#         7.4500e+00, 9.0500e+00, 3.9667e+00, 2.7000e+00, 3.0150e+02, 1.4633e+01,
#         1.0000e+00, 6.5833e+00, 9.3333e-01, 7.7333e+00, 9.1333e+00, 1.3300e+01,
#         2.0367e+01, 4.1767e+01, 5.6667e+00, 1.0333e+00, 5.8167e+00, 3.8333e-01,
#         2.4898e+03, 1.1050e+01, 2.4200e+01, 1.2333e+00, 2.0642e+02, 1.5467e+01,
#         1.0000e+00, 7.1167e+00, 1.0167e+00, 8.1500e+00, 2.3100e+01, 2.1650e+01,
#         1.7500e+00, 2.8850e+01, 1.2917e+01, 5.7833e+00, 1.0667e+00, 6.1000e+00,
#         3.5000e-01, 2.2861e+03, 1.8667e+00, 5.7667e+00, 9.8333e-01, 5.7167e+00,
#         1.2367e+01, 2.8267e+01, 1.7000e+00, 2.0850e+01, 1.3517e+01, 7.5833e+00,
#         7.5667e+00, 9.6667e-01, 6.3833e+00, 8.7833e+00, 8.7000e+00, 3.6167e+00,
#         3.0833e+00, 2.1883e+02, 1.4867e+01, 1.0000e+00, 6.4500e+00, 1.0000e+00,
#         9.1833e+00, 9.0167e+00, 1.3200e+01, 2.0650e+01, 1.8167e+00,
#         ], 'enc_intervals':[0], 'travel_index':[0]}

# '3107EN', '341A09', '341A0B','4504EX'
#  58.2500, 4.0833, 6.4667, 1.1167
data = {'src':['3107EN', '341A09', '341A0B', '4504EX', '3107EN', '341A09', '341A0B', '341A0D', '4505EX', '3107EN',
               '341A09', '341A0B', '341A0D', '4505EX', '4503EN', '351A09', '3107EX',  ], 'enc_src':['3804EN'], 'enc_DoW':[0], 'enc_HoD':[0],
        'intervals':[ 0.0000,   4.6000,   6.8833,   1.0000, 552.6669,   4.5333,   6.9833,
          3.0667,   1.9833,   0.0000,   4.4667,   6.8333,   3.0333,   1.6833,
         41.0833,   1.9833,   4.8167,     ], 'enc_intervals':[0],'travel_index':[0]}

def predict_tokenize(key, value):
    if key == 'intervals':
        value = torch.tensor(value).to(conf["DATASET"]["device"])
        value = exp_normalize(value, 0.0055730214824861605, mean, std)
        value[0] = 0.0
        return trajTokenizer.time_tokenize(value, trajTokenizer.mask_index, True).unsqueeze(0)
    elif key == 'src':
        src = trajTokenizer.stations_tokenize(value)
        return trajTokenizer.tokenize(src).unsqueeze(0)
    elif key == 'enc_src':
        src = trajTokenizer.stations_tokenize(value)
        return trajTokenizer.tokenize(src).unsqueeze(0)
    elif key == 'enc_DoW':
        value = torch.tensor(value).to(conf["DATASET"]["device"])
        return trajTokenizer.time_tokenize(value, trajTokenizer.mask_index, True).unsqueeze(0)
    elif key == 'enc_HoD':
        value = torch.tensor(value).to(conf["DATASET"]["device"])
        return trajTokenizer.time_tokenize(value, trajTokenizer.mask_index, True).unsqueeze(0)
    elif key == 'enc_intervals':
        value = torch.tensor(value).to(conf["DATASET"]["device"])
        return trajTokenizer.time_tokenize(value, trajTokenizer.mask_index, True).unsqueeze(0)
    elif key == 'travel_index':
        value = torch.tensor(value).to(conf["DATASET"]["device"])
        return trajTokenizer.time_tokenize(value, trajTokenizer.mask_index, True).unsqueeze(0)
        

def predict(data, predict_len = 10):
    with torch.no_grad():
        start = time.time()
        travel_len = len(data['src'])
        temp_len = travel_len

        tensor_data = {key: predict_tokenize(key, value) for key, value in data.items()}
        # output, intervals_output =model(enc_src, enc_intervals, src, enc_HoD, enc_DoW, travel_index, travel_mask, intervals)
        for _ in range(predict_len):
            output, intervals_output = model(**tensor_data)
            tensor_data['src'][0][travel_len + 1] = torch.argmax(output, -1)[0][travel_len] 
            # intervals_output = exp_unnormalize(intervals_output, 0.0055730214824861605, mean, std)
            tensor_data['intervals'][0][travel_len] = intervals_output[0][travel_len]
            travel_len += 1
        print("用时：",time.time()-start)
        #将Nan替换为0
        intervals_res = tensor_data['intervals'][0][temp_len : travel_len + 1]
        intervals_res = exp_unnormalize(intervals_res, 0.0055730214824861605, mean, std)
        intervals_res = torch.where(torch.isnan(intervals_res), torch.zeros_like(intervals_res), intervals_res)
        intervals_res = intervals_res.cpu().numpy().tolist()
        
        return trajTokenizer.enuntokenize(tensor_data['src'][0][temp_len + 1: travel_len + 1]), intervals_res
    
@app.get('/')
async def read_root():
    res = predict(data, 1)
    return res

# res = predict(data, 10)
# print(res)
#uvicorn predict:app --host=0.0.0.0 --port=80

# ab 一般常用参数就是 -n， -t ，和 -c。

# -c （concurrency）表示用多少并发来进行测试；

# -t 表示测试持续多长时间；

# -n 表示要发送多少次测试请求。

# 一般 -t 或者 -n 选一个用。

# 例如模拟GET请求进行测试：

# ab -n 20000 -c 1000 http://0.0.0.0:8989/firmware/latest?model=xxx
if __name__ == "__main__":
    uvicorn.run(
        app="predict:app",
        host="0.0.0.0",
        port=8000,
        log_level="debug",
        workers=10,
    )
