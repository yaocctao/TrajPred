import json
from models.transformer import TrajPredTransformer, TrajPredTransformerV1
from utils.config import read_config, print_config, load_to_dict_config
from data.load_data import load_ETCEn_data, load_ETCEn_dataV1
from utils.metrics import Accuracy, time_station_Accuracy, TimeAccuracy, mae_loss, unnormalize
import torch, os, tqdm, numpy as np, random
from torch.utils.tensorboard import SummaryWriter


def train(conf):
    #判断save_model文件路径是否存在
    if not os.path.exists(conf["TRAIN"]["save_path"]):
        os.makedirs(conf["TRAIN"]["save_path"])
    if not os.path.exists(f'{conf["TRAIN"]["save_path"]}/metrics.json'):
        with open(f'{conf["TRAIN"]["save_path"]}/metrics.json', 'w') as result_file:
            result = {}
            json.dump(result, result_file)  
    result = {}
    print("-------------config--------------:\n")
    print_config(conf)
    load_to_dict_config(conf, result)
    print("---------------------------------:\n")
    train_loader, dev_loader, test_loader, history_loader, train_num = load_ETCEn_data(conf)
    model_conf = read_config(conf["TRAIN"]["model_config_path"])
    print("-------------model_config--------------:\n")
    print_config(model_conf)
    load_to_dict_config(model_conf, result)
    print("---------------------------------:\n")
    model = TrajPredTransformer(model_conf["MODEL"])
    model.to(conf["DATASET"]["device"])
    epochs = int(conf["TRAIN"]["max_epoch"])
    eval_epoch = int(conf["EVALUATE"]["epoch"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(conf["TRAIN"]["lr"]))

    model.train()
    acc = -1

    with tqdm.tqdm(total=train_num*epochs, unit='ex') as bar:
        bar.set_description(f'train loss')
        for epoch in range(1, epochs + 1):
            total_loss = 0
            step = 0
            total_intevals_loss = 0
            for i, batch in enumerate(train_loader):
                src = batch['src']
                tgt = batch['tgt']
                DoW = batch['DoW']
                HoD = batch['HoD']
                intervals = batch['intervals']
                intervals_tgt = batch['intervals_tgt']
                
                optimizer.zero_grad()
                output, intervals_output = model(src, HoD, DoW, intervals)
                # loss = model.loss(output[:,2:], tgt[:,2:], intervals_output[:,2:].flatten(), intervals_tgt[:,2:].flatten())
                station_loss = model.des_loss(output[:,2:], tgt[:,2:])
                intervals_loss = model.mae_loss(intervals_output[:,2:].flatten(), intervals_tgt[:,2:].flatten())
                loss =  0.5 * intervals_loss + 0.5 * station_loss
                total_loss += loss.item()
                loss.backward()
                total_intevals_loss += intervals_loss.item()
                optimizer.step()
                bar.update(src.shape[0])
                bar.set_postfix(
                    intervals_loss=f'{total_intevals_loss/(step+1):.2f}',
                    station_loss=f'{station_loss.item():.2f}',
                    avg_loss = f'{total_loss/(step+1):.2f}'
                )
                step += 1
            # print(f"Epoch {epoch} Loss:{total_loss/step}")
            if epoch % eval_epoch == 0:
                eval_acc, eval_time_acc, eval_time_station_acc, eval_mae = evaluate(model, dev_loader)
                test_acc, test_time_acc, test_time_station_acc, test_mae = evaluate(model, test_loader)
                history_acc, history_time_acc, history_time_station_acc, history_mae = evaluate(model, history_loader)
                print(f"\nEval time_station_acc:{eval_time_station_acc}, time_acc:{eval_time_acc}, station_acc:{eval_acc}, mae:{str(float(eval_mae))} \
                      \nTest time_station_acc:{test_time_station_acc}, time_acc:{test_time_acc}, station_acc:{test_acc}, mae:{str(float(test_mae))} \
                      \nhistory time_station_acc:{history_time_station_acc}, time_acc:{history_time_acc}, station_acc:{history_acc}, mae:{str(float(history_mae))}")
                
                result[f"epoch_{epoch}"] = {
                    "EVAL":{"time_station_acc":float(eval_time_station_acc),"time_acc":float(eval_time_acc),"station_acc":float(eval_acc), "mae":float(eval_mae)},
                    "TEST":{"time_station_acc":float(test_time_station_acc),"time_acc":float(test_time_acc),"station_acc":float(test_acc), "mae":float(test_mae)},
                    "HISTORY":{"time_station_acc":float(history_time_station_acc),"time_acc":float(history_time_acc),"station_acc":float(history_acc), "mae":float(history_mae)}
                    }
                with open(os.path.join(conf["TRAIN"]["save_path"],"metrics.json"), "w") as result_file:
                    json.dump(result, result_file)
                if eval_acc > acc:
                    acc = eval_acc
                    torch.save(model.state_dict(), os.path.join(conf["TRAIN"]["save_path"],"best_model.pth"))
                    

def trainV1(conf):
    #判断save_model文件路径是否存在
    if not os.path.exists(conf["TRAIN"]["save_path"]):
        os.makedirs(conf["TRAIN"]["save_path"])
    if not os.path.exists(f'{conf["TRAIN"]["save_path"]}/metrics.json'):
        with open(f'{conf["TRAIN"]["save_path"]}/metrics.json', 'w') as result_file:
            result = {}
            json.dump(result, result_file)  
    result = {}
    print("-------------config--------------:\n")
    print_config(conf)
    load_to_dict_config(conf, result)
    print("---------------------------------:\n")
    train_loader, dev_loader, test_loader, history_loader, train_num = load_ETCEn_dataV1(conf)
    model_conf = read_config(conf["TRAIN"]["model_config_path"])
    print("-------------model_config--------------:\n")
    print_config(model_conf)
    load_to_dict_config(model_conf, result)
    print("---------------------------------:\n")
    model = TrajPredTransformerV1(model_conf["MODEL"])
    model.to(conf["DATASET"]["device"])
    epochs = int(conf["TRAIN"]["max_epoch"])
    eval_epoch = int(conf["EVALUATE"]["epoch"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(conf["TRAIN"]["lr"]))
    if not os.path.exists(conf["LOG"]["save_path"]):
        os.makedirs(conf["LOG"]["save_path"])
    writer =  SummaryWriter(conf["LOG"]["save_path"])

    model.train()
    acc = -1

    global_step = 0
    with tqdm.tqdm(total=train_num*epochs, unit='ex') as bar:
        bar.set_description(f'train loss')
        for epoch in range(1, epochs + 1):
            total_loss = 0
            total_intevals_loss = 0
            for i, batch in enumerate(train_loader):
                src = batch['src']
                enc_src = batch['enc_src']
                tgt = batch['tgt']
                enc_DoW = batch['enc_DoW']
                enc_HoD = batch['enc_HoD']
                intervals = batch['intervals']
                enc_intervals = batch['enc_intervals']
                intervals_tgt = intervals.clone()
                
                optimizer.zero_grad()
                output, intervals_output = model(enc_src, enc_intervals, src, enc_HoD, enc_DoW, intervals, tgt)
                loss, station_loss, intervals_loss = model.loss(output, tgt, intervals_output[:,1:].flatten(), intervals_tgt[:,1:].flatten())
                # station_loss = model.des_loss(output[:,2:], tgt[:,2:])
                # intervals_loss = model.mae_loss(intervals_output[:,2:].flatten(), intervals_tgt[:,2:].flatten())
                # loss =  0.5 * intervals_loss + 0.5 * station_loss
                total_loss += loss.item()
                loss.backward()
                total_intevals_loss += intervals_loss.item()
                optimizer.step()
                bar.update(src.shape[0])
                bar.set_postfix(
                    intervals_loss=f'{total_intevals_loss/(i+1):.2f}',
                    station_loss=f'{station_loss.item():.2f}',
                    avg_loss = f'{total_loss/(i+1):.2f}'
                )
                global_step += 1
            # print(f"Epoch {epoch} Loss:{total_loss/step}")
            if epoch % eval_epoch == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param, global_step)
                eval_acc, eval_time_acc, eval_time_station_acc, eval_mae = evaluate(model, dev_loader)
                test_acc, test_time_acc, test_time_station_acc, test_mae = evaluate(model, test_loader)
                history_acc, history_time_acc, history_time_station_acc, history_mae = evaluate(model, history_loader)
                print(f"\nEval time_station_acc:{eval_time_station_acc}, time_acc:{eval_time_acc}, station_acc:{eval_acc}, mae:{str(float(eval_mae))} \
                      \nTest time_station_acc:{test_time_station_acc}, time_acc:{test_time_acc}, station_acc:{test_acc}, mae:{str(float(test_mae))} \
                      \nhistory time_station_acc:{history_time_station_acc}, time_acc:{history_time_acc}, station_acc:{history_acc}, mae:{str(float(history_mae))}")
                
                result[f"epoch_{epoch}"] = {
                    "EVAL":{"time_station_acc":float(eval_time_station_acc),"time_acc":float(eval_time_acc),"station_acc":float(eval_acc), "mae":float(eval_mae)},
                    "TEST":{"time_station_acc":float(test_time_station_acc),"time_acc":float(test_time_acc),"station_acc":float(test_acc), "mae":float(test_mae)},
                    "HISTORY":{"time_station_acc":float(history_time_station_acc),"time_acc":float(history_time_acc),"station_acc":float(history_acc), "mae":float(history_mae)}
                    }
                with open(os.path.join(conf["TRAIN"]["save_path"],"metrics.json"), "w") as result_file:
                    json.dump(result, result_file)
                if eval_acc > acc:
                    acc = eval_acc
                    torch.save(model.state_dict(), os.path.join(conf["TRAIN"]["save_path"],"best_model.pth"))

def evaluate(model, loader):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            src = batch['src']
            enc_src = batch['enc_src']
            tgt = batch['tgt']
            enc_DoW = batch['enc_DoW']
            enc_HoD = batch['enc_HoD']
            intervals = batch['intervals']
            intervals_tgt = batch['intervals'].clone()
            pred_index = torch.sum(src != 0, dim = -1) - 1
            intervals[torch.arange(tgt.shape[0]), pred_index] = 0
            enc_intervals = batch['enc_intervals']
            output, intervals_output =model(enc_src, enc_intervals, src, enc_HoD, enc_DoW, intervals)
            intervals_output = unnormalize(intervals_output)
            intervals_tgt = unnormalize(intervals_tgt)
            if i == 0:
                predictions = torch.argmax(output, -1)[torch.arange(tgt.shape[0]), pred_index]
                labels = tgt[torch.arange(tgt.shape[0]), pred_index]
                predictions_intervals = intervals_output[torch.arange(tgt.shape[0]), pred_index]
                labels_intervals = intervals_tgt[torch.arange(tgt.shape[0]), pred_index]
            else:
                predictions = torch.cat((predictions, torch.argmax(output, -1)[torch.arange(tgt.shape[0]), pred_index]), 0)
                labels = torch.cat((labels, tgt[torch.arange(tgt.shape[0]), pred_index]), 0)
                predictions_intervals = torch.cat((predictions_intervals, intervals_output[torch.arange(tgt.shape[0]), pred_index]), 0)
                labels_intervals = torch.cat((labels_intervals, intervals_tgt[torch.arange(tgt.shape[0]), pred_index]), 0)
        time_station_acc = time_station_Accuracy(predictions_intervals.flatten(), labels_intervals.flatten(), predictions, labels)
        acc = Accuracy(predictions, labels)
        time_acc = TimeAccuracy(predictions_intervals.flatten(), labels_intervals.flatten(), 60)
        mae = mae_loss(predictions_intervals/60, labels_intervals/60)
    model.train()
    return acc, time_acc, time_station_acc, mae

def predict(model_path, conf, loader):
    model_conf = read_config(conf["TRAIN"]["model_config_path"])
    model = TrajPredTransformerV1(model_conf["MODEL"])
    model.to(conf["DATASET"]["device"])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            src = batch['src']
            enc_src = batch['enc_src']
            tgt = batch['tgt']
            enc_DoW = batch['enc_DoW']
            enc_HoD = batch['enc_HoD']
            intervals = batch['intervals']
            enc_intervals = batch['enc_intervals']
            intervals_tgt = batch['intervals_tgt']
            output, intervals_output =model(enc_src, enc_intervals, src, enc_HoD, enc_DoW, intervals)
            intervals_output = unnormalize(intervals_output)
            intervals_tgt = unnormalize(intervals_tgt)
            pred_index = torch.sum(src != 0, dim = -1) - 1
            if i == 0:
                predictions = torch.argmax(output, -1)[torch.arange(tgt.shape[0]), pred_index]
                labels = tgt[torch.arange(tgt.shape[0]), pred_index]
                predictions_intervals = intervals_output[torch.arange(tgt.shape[0]), pred_index]
                labels_intervals = intervals_tgt[torch.arange(tgt.shape[0]), pred_index]
            else:
                predictions = torch.cat((predictions, torch.argmax(output, -1)[torch.arange(tgt.shape[0]), pred_index]), 0)
                labels = torch.cat((labels, tgt[torch.arange(tgt.shape[0]), pred_index]), 0)
                predictions_intervals = torch.cat((predictions_intervals, intervals_output[torch.arange(tgt.shape[0]), pred_index]), 0)
                labels_intervals = torch.cat((labels_intervals, intervals_tgt[torch.arange(tgt.shape[0]), pred_index]), 0)
        time_station_acc = time_station_Accuracy(predictions_intervals.flatten(), labels_intervals.flatten(), predictions, labels)
        acc = Accuracy(predictions, labels)
        time_acc = TimeAccuracy(predictions_intervals.flatten(), labels_intervals.flatten(), 60)
        mae = mae_loss(predictions_intervals/60, labels_intervals/60)
    # model.train()
    return acc, time_acc, time_station_acc, mae

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    #nohup python train.py > new1.log 2>&1 &
    random_seed(42)
    path = "./config/config.ini"
    conf = read_config(path)
    trainV1(conf)
    # train_loader, dev_loader, test_loader, history_loader, train_num = load_ETCEn_dataV1(conf)
    # predict("weights/with_encoder/best_model.pth", conf, history_loader)