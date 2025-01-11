import importlib
import json
from models.transformer import *
from models.LSTM import TrajLSTM
from utils.config import read_config, print_config, load_to_dict_config
from data.load_data import *
from utils.metrics import Accuracy, time_station_Accuracy, TimeAccuracy, mae_loss, unnormalize
import torch, os, numpy as np, random
from torch.utils.tensorboard import SummaryWriter
from evaluate import *


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

    with tqdm(total=train_num*epochs, unit='ex') as bar:
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

def TrajPredTransformerV1_train(conf):
    #判断save_model文件路径是否存在
    if not os.path.exists(conf["TRAIN"]["save_path"]):
        os.makedirs(conf["TRAIN"]["save_path"])
    if not os.path.exists(f'{conf["TRAIN"]["save_path"]}/metrics.json'):
        with open(f'{conf["TRAIN"]["save_path"]}/metrics.json', 'w') as result_file:
            result = {}
            json.dump(result, result_file)  
    result = {}
    result[f'best_epoch'] = {}
    print("-------------config--------------:\n")
    print_config(conf)
    load_to_dict_config(conf, result)
    print("---------------------------------:\n")
    # module_name = __import__("test_call_function_by_string1")
    model_conf = read_config(conf["TRAIN"]["model_config_path"])
    load_data = eval(model_conf["MODEL"]["load_data_method"])
    train_loader, dev_loader, test_loader, history_loader, train_num = load_data(conf)
    print("-------------model_config--------------:\n")
    print_config(model_conf)
    load_to_dict_config(model_conf, result)
    print("---------------------------------:\n")
    model = eval(conf["TRAIN"]["model_name"])(model_conf["MODEL"])
    model.to(conf["DATASET"]["device"])
    epochs = int(conf["TRAIN"]["max_epoch"])
    eval_epoch = int(conf["EVALUATE"]["epoch"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(conf["TRAIN"]["lr"]))
    if not os.path.exists(conf["LOG"]["save_path"]):
        os.makedirs(conf["LOG"]["save_path"])
    writer =  SummaryWriter(conf["LOG"]["save_path"])
    mean = float(conf["DATASET"]["mean"])
    std = float(conf["DATASET"]["std"])

    model.train()
    acc = -1

    global_step = 0
    with tqdm(total=train_num*epochs, unit='ex') as bar:
        bar.set_description(f'train loss')
        for epoch in range(1, epochs + 1):
            total_loss = 0
            total_intevals_loss = 0
            for i, batch in enumerate(train_loader):
                src = batch['src']
                enc_src = batch['enc_src']
                travel_index = batch['travel_index']
                tgt = batch['tgt']
                enc_DoW = batch['enc_DoW']
                enc_HoD = batch['enc_HoD']
                intervals = batch['intervals']
                enc_intervals = batch['enc_intervals']
                intervals_tgt = intervals.clone()
                
                optimizer.zero_grad()
                output, intervals_output = model(enc_src, enc_intervals, src, enc_HoD, enc_DoW, intervals, travel_index, tgt)
                loss, station_loss, intervals_loss = model.loss(output, tgt, intervals_output[:,1:].flatten(), intervals_tgt[:,1:].flatten(), mean, std)
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
                eval_acc, eval_time_acc, eval_time_station_acc, eval_mae, eval_en_time_station_acc, eval_en_acc, eval_en_time_acc = eval(conf["TRAIN"]["model_name"] + "_" + "evaluate")(model, dev_loader, mean, std)
                test_acc, test_time_acc, test_time_station_acc, test_mae, test_en_time_station_acc, test_en_acc, test_en_time_acc = eval(conf["TRAIN"]["model_name"] + "_" + "evaluate")(model, test_loader, mean, std)
                history_acc, history_time_acc, history_time_station_acc, history_mae, history_en_time_station_acc, history_en_acc, history_en_time_acc = eval(conf["TRAIN"]["model_name"] + "_" + "evaluate")(model, history_loader, mean, std)
                print(f"\nEval time_station_acc:{round(eval_time_station_acc * 100, 5)}, time_acc:{round(eval_time_acc * 100, 5)}, station_acc:{round(eval_acc * 100, 5)}, mae:{str(float(eval_mae))} \
                        en_time_station_acc:{round(eval_en_time_station_acc * 100, 5)}, en_time_acc:{round(eval_en_time_acc * 100, 5)}, en_station_acc:{round(eval_en_acc * 100, 5)} \
                        \nTest time_station_acc:{round(test_time_station_acc * 100, 5)}, time_acc:{round(test_time_acc * 100, 5)}, station_acc:{round(test_acc * 100, 5)}, mae:{str(float(test_mae))} \
                        en_time_station_acc:{round(test_en_time_station_acc * 100, 5)}, en_time_acc:{round(test_en_time_acc * 100, 5)}, en_station_acc:{round(test_en_acc * 100, 5)} \
                        \nhistory time_station_acc:{round(history_time_station_acc * 100, 5)}, time_acc:{round(history_time_acc * 100, 5)}, station_acc:{round(history_acc * 100, 5)}, mae:{str(float(history_mae))}\
                        en_time_station_acc:{round(history_en_time_station_acc * 100, 5)}, en_time_acc:{round(history_en_time_acc * 100, 5)}, en_station_acc:{round(history_en_acc * 100, 5)}"
                        )                
                result[f"epoch_{epoch}"] = {
                    "EVAL":{"time_station_acc":float(eval_time_station_acc),"time_acc":float(eval_time_acc),"station_acc":float(eval_acc), "mae":float(eval_mae), "en_time_station_acc":float(eval_en_time_station_acc), "en_time_acc":float(eval_en_time_acc), "en_station_acc":float(eval_en_acc)},
                    "TEST":{"time_station_acc":float(test_time_station_acc),"time_acc":float(test_time_acc),"station_acc":float(test_acc), "mae":float(test_mae), "en_time_station_acc":float(test_en_time_station_acc), "en_time_acc":float(test_en_time_acc), "en_station_acc":float(test_en_acc)},
                    "HISTORY":{"time_station_acc":float(history_time_station_acc),"time_acc":float(history_time_acc),"station_acc":float(history_acc), "mae":float(history_mae), "en_time_station_acc":float(history_en_time_station_acc), "en_time_acc":float(history_en_time_acc), "en_station_acc":float(history_en_acc)}
                    }
                
                torch.save(model.state_dict(), os.path.join(conf["TRAIN"]["save_path"],f"epoch_{epoch}.pth"))
                if eval_time_station_acc > acc:
                    result[f"best_epoch"] = {
                        "EVAL":{"time_station_acc":float(eval_time_station_acc),"time_acc":float(eval_time_acc),"station_acc":float(eval_acc), "mae":float(eval_mae), "en_time_station_acc":float(eval_en_time_station_acc), "en_time_acc":float(eval_en_time_acc), "en_station_acc":float(eval_en_acc)},
                        "TEST":{"time_station_acc":float(test_time_station_acc),"time_acc":float(test_time_acc),"station_acc":float(test_acc), "mae":float(test_mae), "en_time_station_acc":float(test_en_time_station_acc), "en_time_acc":float(test_en_time_acc), "en_station_acc":float(test_en_acc)},
                        "HISTORY":{"time_station_acc":float(history_time_station_acc),"time_acc":float(history_time_acc),"station_acc":float(history_acc), "mae":float(history_mae), "en_time_station_acc":float(history_en_time_station_acc), "en_time_acc":float(history_en_time_acc), "en_station_acc":float(history_en_acc)}
                        }
                    acc = eval_time_station_acc
                    torch.save(model.state_dict(), os.path.join(conf["TRAIN"]["save_path"],"best_model.pth"))

                with open(os.path.join(conf["TRAIN"]["save_path"],"metrics.json"), "w") as result_file:
                    json.dump(result, result_file)

def test(conf):
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
    # module_name = __import__("test_call_function_by_string1")
    model_conf = read_config(conf["TRAIN"]["model_config_path"])
    load_data = eval(model_conf["MODEL"]["load_data_method"])
    train_loader, dev_loader, test_loader, history_loader, train_num = load_data(conf)
    print("-------------model_config--------------:\n")
    print_config(model_conf)
    load_to_dict_config(model_conf, result)
    print("---------------------------------:\n")
    model = eval(conf["TRAIN"]["model_name"])(model_conf["MODEL"])
    model_path = os.path.join(conf['TRAIN']['save_path'], "best_model.pth")
    model.to(conf["DATASET"]["device"])
    model.load_state_dict(torch.load(model_path))
    mean = float(conf["DATASET"]["mean"])
    std = float(conf["DATASET"]["std"])

    eval_acc, eval_time_acc, eval_time_station_acc, eval_mae, eval_en_time_station_acc, eval_en_acc, eval_en_time_acc = eval(conf["TRAIN"]["model_name"] + "_" + "evaluate")(model, dev_loader, mean, std)
    test_acc, test_time_acc, test_time_station_acc, test_mae, test_en_time_station_acc, test_en_acc, test_en_time_acc = eval(conf["TRAIN"]["model_name"] + "_" + "evaluate")(model, test_loader, mean, std)
    history_acc, history_time_acc, history_time_station_acc, history_mae, history_en_time_station_acc, history_en_acc, history_en_time_acc = eval(conf["TRAIN"]["model_name"] + "_" + "evaluate")(model, history_loader, mean, std)
    print(f"\nEval time_station_acc:{round(eval_time_station_acc * 100, 5)}, time_acc:{round(eval_time_acc * 100, 5)}, station_acc:{round(eval_acc * 100, 5)}, mae:{str(float(eval_mae))} \
            en_time_station_acc:{round(eval_en_time_station_acc * 100, 5)}, en_time_acc:{round(eval_en_time_acc * 100, 5)}, en_station_acc:{round(eval_en_acc * 100, 5)} \
            \nTest time_station_acc:{round(test_time_station_acc * 100, 5)}, time_acc:{round(test_time_acc * 100, 5)}, station_acc:{round(test_acc * 100, 5)}, mae:{str(float(test_mae))} \
            en_time_station_acc:{round(test_en_time_station_acc * 100, 5)}, en_time_acc:{round(test_en_time_acc * 100, 5)}, en_station_acc:{round(test_en_acc * 100, 5)} \
            \nhistory time_station_acc:{round(history_time_station_acc * 100, 5)}, time_acc:{round(history_time_acc * 100, 5)}, station_acc:{round(history_acc * 100, 5)}, mae:{str(float(history_mae))}\
            en_time_station_acc:{round(history_en_time_station_acc * 100, 5)}, en_time_acc:{round(history_en_time_acc * 100, 5)}, en_station_acc:{round(history_en_acc * 100, 5)}"
            )

def TrajPredTransformerV2_train(conf):
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
    # module_name = __import__("test_call_function_by_string1")
    model_conf = read_config(conf["TRAIN"]["model_config_path"])
    load_data = eval(model_conf["MODEL"]["load_data_method"])
    train_loader, dev_loader, test_loader, history_loader, train_num = load_data(conf)
    print("-------------model_config--------------:\n")
    print_config(model_conf)
    load_to_dict_config(model_conf, result)
    print("---------------------------------:\n")
    model = eval(conf["TRAIN"]["model_name"])(model_conf["MODEL"])
    model.to(conf["DATASET"]["device"])
    epochs = int(conf["TRAIN"]["max_epoch"])
    eval_epoch = int(conf["EVALUATE"]["epoch"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(conf["TRAIN"]["lr"]))
    if not os.path.exists(conf["LOG"]["save_path"]):
        os.makedirs(conf["LOG"]["save_path"])
    writer =  SummaryWriter(conf["LOG"]["save_path"])
    mean = float(conf["DATASET"]["mean"])
    std = float(conf["DATASET"]["std"])

    model.train()
    acc = -1

    global_step = 0
    with tqdm(total=train_num*epochs, unit='ex') as bar:
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
                travel_index = batch['travel_index']
                travel_mask = batch['travel_mask']
                intervals = batch['intervals']
                enc_intervals = batch['enc_intervals']
                intervals_tgt = intervals.clone()
                
                optimizer.zero_grad()
                output, intervals_output = model(enc_src, enc_intervals, src, enc_HoD, enc_DoW, travel_index, travel_mask, intervals, tgt)
                loss, station_loss, intervals_loss = model.loss(output, tgt, intervals_output[:,1:].flatten(), intervals_tgt[:,1:].flatten(), mean, std)
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
                # for name, param in model.named_parameters():
                #     writer.add_histogram(name, param, global_step)
                eval_acc, eval_time_acc, eval_time_station_acc, eval_mae = eval(conf["TRAIN"]["model_name"] + "_" + "evaluate")(model, dev_loader, mean, std)
                test_acc, test_time_acc, test_time_station_acc, test_mae = eval(conf["TRAIN"]["model_name"] + "_" + "evaluate")(model, test_loader, mean, std)
                history_acc, history_time_acc, history_time_station_acc, history_mae = eval(conf["TRAIN"]["model_name"] + "_" + "evaluate")(model, history_loader, mean, std)
                print(f"\nEval time_station_acc:{round(eval_time_station_acc * 100, 5)}, time_acc:{round(eval_time_acc * 100, 5)}, station_acc:{round(eval_acc * 100, 5)}, mae:{str(float(eval_mae))} \
                      \nTest time_station_acc:{round(test_time_station_acc * 100, 5)}, time_acc:{round(test_time_acc * 100, 5)}, station_acc:{round(test_acc * 100, 5)}, mae:{str(float(test_mae))} \
                      \nhistory time_station_acc:{round(history_time_station_acc * 100, 5)}, time_acc:{round(history_time_acc * 100, 5)}, station_acc:{round(history_acc * 100, 5)}, mae:{str(float(history_mae))}")
                
                result[f"epoch_{epoch}"] = {
                    "EVAL":{"time_station_acc":float(eval_time_station_acc),"time_acc":float(eval_time_acc),"station_acc":float(eval_acc), "mae":float(eval_mae)},
                    "TEST":{"time_station_acc":float(test_time_station_acc),"time_acc":float(test_time_acc),"station_acc":float(test_acc), "mae":float(test_mae)},
                    "HISTORY":{"time_station_acc":float(history_time_station_acc),"time_acc":float(history_time_acc),"station_acc":float(history_acc), "mae":float(history_mae)}
                    }
                with open(os.path.join(conf["TRAIN"]["save_path"],"metrics.json"), "w") as result_file:
                    json.dump(result, result_file)
                torch.save(model.state_dict(), os.path.join(conf["TRAIN"]["save_path"],f"epoch_{epoch}.pth"))
                if eval_acc > acc:
                    acc = eval_acc
                    torch.save(model.state_dict(), os.path.join(conf["TRAIN"]["save_path"],"best_model.pth"))

def TrajPredTransformerV3_train(conf):
    #判断save_model文件路径是否存在
    if not os.path.exists(conf["TRAIN"]["save_path"]):
        os.makedirs(conf["TRAIN"]["save_path"])
    if not os.path.exists(f'{conf["TRAIN"]["save_path"]}/metrics.json'):
        with open(f'{conf["TRAIN"]["save_path"]}/metrics.json', 'w') as result_file:
            result = {}
            json.dump(result, result_file)  
    result = {}
    result[f'best_epoch'] = {}
    print("-------------config--------------:\n")
    print_config(conf)
    load_to_dict_config(conf, result)
    print("---------------------------------:\n")
    # module_name = __import__("test_call_function_by_string1")
    model_conf = read_config(conf["TRAIN"]["model_config_path"])
    load_data = eval(model_conf["MODEL"]["load_data_method"])
    train_loader, dev_loader, test_loader, history_loader, train_num = load_data(conf)
    print("-------------model_config--------------:\n")
    print_config(model_conf)
    load_to_dict_config(model_conf, result)
    print("---------------------------------:\n")
    model = eval(conf["TRAIN"]["model_name"])(model_conf["MODEL"])
    model.to(conf["DATASET"]["device"])
    epochs = int(conf["TRAIN"]["max_epoch"])
    eval_epoch = int(conf["EVALUATE"]["epoch"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(conf["TRAIN"]["lr"]))
    if not os.path.exists(conf["LOG"]["save_path"]):
        os.makedirs(conf["LOG"]["save_path"])
    writer =  SummaryWriter(conf["LOG"]["save_path"])
    mean = float(conf["DATASET"]["mean"])
    std = float(conf["DATASET"]["std"])

    model.train()
    acc = -1

    global_step = 0
    with tqdm(total=train_num*epochs, unit='ex') as bar:
        bar.set_description(f'train loss')
        for epoch in range(1, epochs + 1):
            total_loss = 0
            total_intevals_loss = 0
            for i, batch in enumerate(train_loader):
                src = batch['src']
                enc_src = batch['enc_src']
                travel_index = batch['travel_index']
                tgt = batch['tgt']
                enc_DoW = batch['enc_DoW']
                enc_HoD = batch['enc_HoD']
                intervals = batch['intervals']
                enc_intervals = batch['enc_intervals']
                intervals_tgt = intervals.clone()
                
                optimizer.zero_grad()
                output, intervals_output = model(enc_src, enc_intervals, src, enc_HoD, enc_DoW, intervals, travel_index, tgt)
                loss, station_loss, intervals_loss = model.loss(output, tgt, intervals_output[:,1:].flatten(), intervals_tgt[:,1:].flatten(), mean, std)
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
                eval_acc, eval_time_acc, eval_time_station_acc, eval_mae, eval_en_time_station_acc, eval_en_acc, eval_en_time_acc = eval(conf["TRAIN"]["model_name"] + "_" + "evaluate")(model, dev_loader, mean, std)
                test_acc, test_time_acc, test_time_station_acc, test_mae, test_en_time_station_acc, test_en_acc, test_en_time_acc = eval(conf["TRAIN"]["model_name"] + "_" + "evaluate")(model, test_loader, mean, std)
                history_acc, history_time_acc, history_time_station_acc, history_mae, history_en_time_station_acc, history_en_acc, history_en_time_acc = eval(conf["TRAIN"]["model_name"] + "_" + "evaluate")(model, history_loader, mean, std)
                print(f"\nEval time_station_acc:{round(eval_time_station_acc * 100, 5)}, time_acc:{round(eval_time_acc * 100, 5)}, station_acc:{round(eval_acc * 100, 5)}, mae:{str(float(eval_mae))} \
                        en_time_station_acc:{round(eval_en_time_station_acc * 100, 5)}, en_time_acc:{round(eval_en_time_acc * 100, 5)}, en_station_acc:{round(eval_en_acc * 100, 5)} \
                        \nTest time_station_acc:{round(test_time_station_acc * 100, 5)}, time_acc:{round(test_time_acc * 100, 5)}, station_acc:{round(test_acc * 100, 5)}, mae:{str(float(test_mae))} \
                        en_time_station_acc:{round(test_en_time_station_acc * 100, 5)}, en_time_acc:{round(test_en_time_acc * 100, 5)}, en_station_acc:{round(test_en_acc * 100, 5)} \
                        \nhistory time_station_acc:{round(history_time_station_acc * 100, 5)}, time_acc:{round(history_time_acc * 100, 5)}, station_acc:{round(history_acc * 100, 5)}, mae:{str(float(history_mae))}\
                        en_time_station_acc:{round(history_en_time_station_acc * 100, 5)}, en_time_acc:{round(history_en_time_acc * 100, 5)}, en_station_acc:{round(history_en_acc * 100, 5)}"
                        )                
                result[f"epoch_{epoch}"] = {
                    "EVAL":{"time_station_acc":float(eval_time_station_acc),"time_acc":float(eval_time_acc),"station_acc":float(eval_acc), "mae":float(eval_mae), "en_time_station_acc":float(eval_en_time_station_acc), "en_time_acc":float(eval_en_time_acc), "en_station_acc":float(eval_en_acc)},
                    "TEST":{"time_station_acc":float(test_time_station_acc),"time_acc":float(test_time_acc),"station_acc":float(test_acc), "mae":float(test_mae), "en_time_station_acc":float(test_en_time_station_acc), "en_time_acc":float(test_en_time_acc), "en_station_acc":float(test_en_acc)},
                    "HISTORY":{"time_station_acc":float(history_time_station_acc),"time_acc":float(history_time_acc),"station_acc":float(history_acc), "mae":float(history_mae), "en_time_station_acc":float(history_en_time_station_acc), "en_time_acc":float(history_en_time_acc), "en_station_acc":float(history_en_acc)}
                    }
                
                torch.save(model.state_dict(), os.path.join(conf["TRAIN"]["save_path"],f"epoch_{epoch}.pth"))
                if eval_time_station_acc > acc:
                    result[f"best_epoch"] = {
                        "EVAL":{"time_station_acc":float(eval_time_station_acc),"time_acc":float(eval_time_acc),"station_acc":float(eval_acc), "mae":float(eval_mae), "en_time_station_acc":float(eval_en_time_station_acc), "en_time_acc":float(eval_en_time_acc), "en_station_acc":float(eval_en_acc)},
                        "TEST":{"time_station_acc":float(test_time_station_acc),"time_acc":float(test_time_acc),"station_acc":float(test_acc), "mae":float(test_mae), "en_time_station_acc":float(test_en_time_station_acc), "en_time_acc":float(test_en_time_acc), "en_station_acc":float(test_en_acc)},
                        "HISTORY":{"time_station_acc":float(history_time_station_acc),"time_acc":float(history_time_acc),"station_acc":float(history_acc), "mae":float(history_mae), "en_time_station_acc":float(history_en_time_station_acc), "en_time_acc":float(history_en_time_acc), "en_station_acc":float(history_en_acc)}
                        }
                    acc = eval_time_station_acc
                    torch.save(model.state_dict(), os.path.join(conf["TRAIN"]["save_path"],"best_model.pth"))

                with open(os.path.join(conf["TRAIN"]["save_path"],"metrics.json"), "w") as result_file:
                    json.dump(result, result_file)


def TrajLSTM_train(conf):
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
    model_conf = read_config(conf["TRAIN"]["model_config_path"])
    load_data = eval(model_conf["MODEL"]["load_data_method"])
    train_loader, dev_loader, test_loader, history_loader, train_num = load_data(conf)
    print("-------------model_config--------------:\n")
    print_config(model_conf)
    load_to_dict_config(model_conf, result)
    print("---------------------------------:\n")
    model = eval(conf["TRAIN"]["model_name"])(model_conf["MODEL"])
    model.to(conf["DATASET"]["device"])
    epochs = int(conf["TRAIN"]["max_epoch"])
    eval_epoch = int(conf["EVALUATE"]["epoch"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(conf["TRAIN"]["lr"]))
    if not os.path.exists(conf["LOG"]["save_path"]):
        os.makedirs(conf["LOG"]["save_path"])
    writer =  SummaryWriter(conf["LOG"]["save_path"])
    
    mean = float(conf["DATASET"]["mean"])
    std = float(conf["DATASET"]["std"])

    model.train()
    acc = -1

    global_step = 0
    with tqdm(total=train_num*epochs, unit='ex') as bar:
        bar.set_description(f'train loss')
        for epoch in range(1, epochs + 1):
            total_loss = 0
            total_intevals_loss = 0
            for i, batch in enumerate(train_loader):
                src = batch['src']
                tgt = batch['tgt']
                intervals = batch['intervals']
                intervals_tgt = batch['tgt_intervals']
                
                optimizer.zero_grad()
                output, intervals_output = model(src, intervals)
                loss, station_loss, intervals_loss = model.loss(output, tgt, intervals_output.flatten(), intervals_tgt.flatten())
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
                eval_acc, eval_time_acc, eval_time_station_acc, eval_mae = eval(conf["TRAIN"]["model_name"] + "_" + "evaluate")(model, dev_loader, mean, std)
                test_acc, test_time_acc, test_time_station_acc, test_mae = eval(conf["TRAIN"]["model_name"] + "_" + "evaluate")(model, test_loader, mean, std)
                history_acc, history_time_acc, history_time_station_acc, history_mae = eval(conf["TRAIN"]["model_name"] + "_" + "evaluate")(model, history_loader, mean, std)
                print(f"\nEval time_station_acc:{round(eval_time_station_acc * 100, 5)}, time_acc:{round(eval_time_acc * 100, 5)}, station_acc:{round(eval_acc * 100, 5)}, mae:{str(float(eval_mae))} \
                      \nTest time_station_acc:{round(test_time_station_acc * 100, 5)}, time_acc:{round(test_time_acc * 100, 5)}, station_acc:{round(test_acc * 100, 5)}, mae:{str(float(test_mae))} \
                      \nhistory time_station_acc:{round(history_time_station_acc * 100, 5)}, time_acc:{round(history_time_acc * 100, 5)}, station_acc:{round(history_acc * 100, 5)}, mae:{str(float(history_mae))}")
                
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


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    #nohup python train.py > Transformer.log 2>&1 &
    random_seed(42)
    path = "./config/Transformer.ini"
    conf = read_config(path)
    eval(conf["TRAIN"]["model_name"]+"_"+"train")(conf)
    # test(conf)
    # train_loader, dev_loader, test_loader, history_loader, train_num = load_ETCEn_dataV1(conf)
    # predict("weights/with_encoder/best_model.pth", conf, history_loader)