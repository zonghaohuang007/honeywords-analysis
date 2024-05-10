
import numpy as np
import random
from time import time
import pickle
from tqdm import tqdm
import os
import sys
import random
import argparse
from utils import *
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


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class RNN_Model(nn.Module):
    def __init__(self, vocab_size, dim, num_layer, num_classes):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layer = num_layer
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim)
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            num_layers=num_layer,
            batch_first = True
        )
        self.fc1 = nn.Linear(self.dim, 32)
        self.drop_out = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, args.num_classes)

    def forward(self, x):
        prev_state = (torch.zeros(self.num_layer, x.shape[0], self.dim).to(device), torch.zeros(self.num_layer, x.shape[0], self.dim).to(device))       
        embed = self.embedding(x)
        output, _ = self.lstm(embed, prev_state)
        logits = self.fc1(output)
        logits = nn.ReLU()(logits)
        logits = self.drop_out(logits)
        logits = self.fc2(logits)
        logits = logits[:, -1, :]
        return nn.Sigmoid()(logits) 


class DataGen(Dataset):
    def __init__(self, data_path, char_map, char2digits):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.char_map = char_map
        self.char2digits = char2digits
        self.data = []
        self.label = []
        self.class_name = []
        label = 0
        for k, v in data.items():
            self.data += v
            self.label += [label] * len(v)
            self.class_name.append(k)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        
        x = self.data[idx]
        y = self.label[idx]
        return x, y


def train_model():

    # define model
    model = RNN_Model(args.char_size, args.dim, args.num_layer, args.num_classes).to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[args.local_rank])

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

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()

    tic0 = time()
    for epoch in range(args.n_epoch):
        print('====== start {}-th epoch ======='.format(epoch + 1))
        for inputs, labels in train_dataset:

            # train the model
            optimizer.zero_grad()
            model.train()

            inputs = inputs.to(torch.int64).to(device)
            labels = labels.to(torch.int64).to(device)

            logits = model(inputs)
            
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

        print(' | '.join([f'==> train info: epoch {epoch + 1} | '
                                  f'sec/epoch {round((time() - tic0), 2)}']
                                  + [f'loss: {loss.item()}']
                                 )
                      )
        tic0 = time()

        print('====== end {}-th epoch ======='.format(epoch + 1))
        dic = {'model': model.module.state_dict(), 'dim': args.dim, 'num_layer': args.num_layer, 'epoch': epoch+1}
        torch.save(dic, model_path + '/model')

    print('==> finished training.')


if __name__ == '__main__':
    # get parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch', type=int, default=30)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-03)
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--num_classes', type=int, default=39)
    parser.add_argument("--local_rank", type=int, default=-1, help="number of cpu threads to use during batch generation")
    args = parser.parse_args()

    exp_name = 'classifier_model'

    # set up paths
    model_path = './saved_model/' + exp_name
    if not os.path.exists('./saved_model'):
        os.makedirs('./saved_model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # make char map
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
