import configparser
import torch.nn.functional as F


def read_config(path):
    cf = configparser.ConfigParser()
    cf.read(path, encoding="utf-8")  # 读取config.ini
    return cf

def print_config(cf):
    # 遍历所有部分（sections）
    for section in cf.sections():
        print(f"[{section}]")
        # 遍历该部分的所有键值对
        for key, value in cf.items(section):
            print(f"{key} = {value}")
        print()

def load_to_dict_config(cf, items):
    # 遍历所有部分（sections）
    for section in cf.sections():
        items[section] = {}
        # 遍历该部分的所有键值对
        for key, value in cf.items(section):
            items[section][key] = value