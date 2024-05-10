
import numpy as np

import os
import sys
import random
import argparse
from utils import *
from transformer_model import *
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
    def __init__(self, data_path, char_map, char2digits):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.char_map = char_map
        self.char2digits = char2digits

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        
        x = random.sample(self.data[idx], 2)
        x = self.char2digits(x, self.char_map, 30, add_sos=True)
        x1 = x[0][1:]
        x2 = x[1][:-1]
        x3 = x[1][1:]
        
        return x1, x2, x3


def train_model():

    model = Transformer_model(args.char_size, args.out_dim, args.num_layer, args.d_model, args.nhead)
    # model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[args.local_rank])
    else:
        model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('==> total number of model parameters: {} | total number of trainable parameters: {}'.format(pytorch_total_params, pytorch_trainable_params))

    # data loader
    train_dataset = DataGen(args.data_path, char_map, char2digits)
    world_size = torch.cuda.device_count()
    datasampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=args.local_rank)
    train_dataset = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        sampler=datasampler
    )

    criterion = nn.CrossEntropyLoss()

    # lr = lr_schedule(args.lr, epoch, max_epoch=args.n_epoch)
    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print('==> learning rate is {}'.format(lr))

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
            
            mask = (torch.ones(tgt_in.shape[1], tgt_in.shape[1]) - torch.triu(torch.ones(tgt_in.shape[1], tgt_in.shape[1]))).to(torch.bool).transpose(0, 1).to(device)
        
            logits = model([src_in, tgt_in], mask)
            
            loss = criterion(logits.reshape(-1, args.char_size), tgt_out.reshape(-1, ))

            loss.backward()
            optimizer.step()

            if step % 2000 == 0:
                print(' | '.join([f'==> train info: epoch {epoch + 1} | step {step} | '
                                  f'sec/step {round((time() - tic0) / 2000, 2)}']
                                  + [f'loss: {trunc_decimal(loss.item())}']
                                 )
                      )
                tic0 = time()
            step = step + 1

        print('====== end {}-th epoch ======='.format(epoch + 1))
        dic = {'model': model.module.state_dict(), 'out_dim': args.out_dim, 'num_layer': args.num_layer, 'd_model': args.d_model, 'nhead': args.nhead, 'epoch': epoch+1}
        torch.save(dic, model_path + '/model')

    print('==> finished transformer-based similarity model.')


if __name__ == '__main__':
    # get parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch', type=int, default=30)
    parser.add_argument('--out_dim', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-03)
    parser.add_argument('--data_path', type=str, default='/BreachCompilationAnalysis/preprocessed_data/train_data.pickle')
    parser.add_argument("--local_rank", type=int, default=-1, help="number of cpu threads to use during batch generation")
    args = parser.parse_args()

    exp_name = 'similarity_model'

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

    # Initialize Process Group
    dist_backend = 'nccl'
    print('args.local_rank: ', args.local_rank)
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=dist_backend)

    device = torch.device(args.local_rank)

    train_model()
