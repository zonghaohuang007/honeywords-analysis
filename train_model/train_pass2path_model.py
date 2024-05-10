
import numpy as np

import transformers

import os
import sys
import random
import argparse
from utils import *
from utils_pwd import *
from train_model.mymodels import *
from time import time
from math import floor, log2
from scipy import stats

from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


torch.set_num_threads(1)
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class DataGen(Dataset):
    def __init__(self, data_path, char_map, char2digits, find_med_backtrace, tran_dict):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.char_map = char_map
        self.char2digits = char2digits
        self.find_med_backtrace = find_med_backtrace
        self.tran_dict = tran_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        
        x = random.sample(self.data[idx], 2)
        _, path = self.find_med_backtrace(x[0], x[1])
        idx_path = path2idx(path, self.tran_dict)
        x = self.char2digits(x, self.char_map, 30, add_sos=True)
        x1 = x[0][1:]
        x2 = [len(self.tran_dict) + 1] + idx_path + [0 for _ in range(30 - len(idx_path))]
        assert len(x2) == 31
        x3 = idx_path + [0 for _ in range(31 - len(idx_path))]
        assert len(x3) == 31
        
        return x1, np.array(x2), np.array(x3)


def train_model():

    # define model
    model = PASS2PATH(args.char_size, args.vocab_size, args.embed_dim, args.lstm_dim, args.num_layer, args.dropout_rate).to(device)

    # model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[args.local_rank])
    else:
        model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('==> total number of model parameters: {} | total number of trainable parameters: {}'.format(pytorch_total_params, pytorch_trainable_params))

    # data loader
    train_dataset = DataGen(args.data_path, char_map, char2digits, find_med_backtrace, tran_dict)
    world_size = torch.cuda.device_count()
    datasampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=args.local_rank)
    train_dataset = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        sampler=datasampler
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataset)*args.n_epoch*0.1, num_training_steps=len(train_dataset)*args.n_epoch)

    tic0 = time()
    step = 0
    for epoch in range(args.n_epoch):
        print('====== start {}-th epoch ======='.format(epoch + 1))

        for x1_out, x2_in, x2_out in train_dataset:

            # train the model
            optimizer.zero_grad()
            model.train()

            src_in = x1_out.to(torch.int64).to(device)
            tgt_in = x2_in.to(torch.int64).to(device)
            tgt_out = x2_out.to(torch.int64).to(device)
            
            encoder_state = (torch.zeros(args.num_layer, tgt_in.shape[0], args.lstm_dim).to(device), torch.zeros(args.num_layer, tgt_in.shape[0], args.lstm_dim).to(device))

            logits = model(tgt_in, x=src_in, encoder_state=encoder_state)
            
            loss = criterion(logits.reshape(-1, args.vocab_size), tgt_out.reshape(-1, ))

            loss.backward()
            optimizer.step()
            # scheduler.step()

            if step % 2000 == 0:
                print(' | '.join([f'==> train info: epoch {epoch + 1} | step {step} | '
                                  f'sec/step {round((time() - tic0) / 2000, 2)}']
                                  + [f'loss: {trunc_decimal(loss.item())}']
                                 )
                      )

                tic0 = time()
            step = step + 1

        print('====== end {}-th epoch ======='.format(epoch + 1))
        dic = {'model': model.module.state_dict(), 'embed_dim': args.embed_dim, 'lstm_dim': args.lstm_dim, 'num_layer': args.num_layer, 'dropout_rate': args.dropout_rate, 'epoch': epoch+1}
        torch.save(dic, model_path + '/model')

    print('==> finished rnn-based pass2path model.')


if __name__ == '__main__':
    # get parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--embed_dim', type=int, default=200)
    parser.add_argument('--lstm_dim', type=int, default=128)
    parser.add_argument('--dropout_rate', type=int, default=0.4)
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=3e-04)
    parser.add_argument('--data_path', type=str, default='/BreachCompilationAnalysis/preprocessed_data/train_data.pickle')
    parser.add_argument("--local_rank", type=int, default=-1, help="number of cpu threads to use during batch generation")
    args = parser.parse_args()

    exp_name = 'pass2path_model'

    # set up paths
    model_path = './saved_model/' + exp_name
    if not os.path.exists('./saved_model'):
        os.makedirs('./saved_model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # load CM map: a dictionary to map char or special symbol or number into digit
    char_map, char_map_inv = load_charmap('./char_map.pickle')
    char_size = len(char_map) + 1
    id_sos = len(char_map)
    args.char_size = char_size

    with open('trans_dict_2idx.json', 'r') as f:
        tran_dict = json.load(f)
    args.vocab_size = len(tran_dict) + 2

    # Initialize Process Group
    dist_backend = 'nccl'
    print('args.local_rank: ', args.local_rank)
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=dist_backend)

    device = torch.device(args.local_rank)

    train_model()
