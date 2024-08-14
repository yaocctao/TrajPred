import os, json
import torch.nn.functional as F
from utils.config import read_config

class Entokenizer():
    def __init__(self, conf):
        stations_vocab_path = os.path.join(conf["DATASET"]["path"], "stations2id.json")
        model_conf = read_config(conf["TRAIN"]["model_config_path"])
        self.max_len = int(model_conf["MODEL"]["max_len"])
        self.stations2id = json.load(open(stations_vocab_path, 'r'))
        self.id2stations = {v: k for k, v in self.stations2id.items()}
        self.mask_index = self.stations2id["<MASK>"]
        self.bos_index = self.stations2id["<bos>"]
        self.eos_index = self.stations2id["<eos>"]

    def tokenize(self, seq):
        seq = F.pad(seq, (1, 0), value=self.bos_index)
        # seq = F.pad(seq, (0, 1), value=self.eos_index)
        seq_len = len(seq)
        if seq_len > self.max_len:
            seq = seq[:self.max_len]
        else:
            seq = F.pad(seq, (0, self.max_len - seq_len), value = self.mask_index)
        return seq
    
    def time_tokenize(self, seq, index_value, is_unsqueeze = False):
        seq = F.pad(seq, (1, 0), value=0)
        seq_len = len(seq)
        if seq_len > self.max_len:
            seq = seq[:self.max_len]
        else:
            seq = F.pad(seq, (0, self.max_len - seq_len), value = index_value)
            
        if is_unsqueeze:
            return seq.unsqueeze(-1)
        else:
            return seq
    
    def enuntokenize(self, seq):
        return [self.id2stations[int(token)] for token in seq]
    
    def enuntokenize_batch(self, batch_tokens):
        return [self.enuntokenize(tokens) for tokens in batch_tokens]

    def tokenize_batch(self, batch_tokens):
        return [self.tokenize(tokens) for tokens in batch_tokens]