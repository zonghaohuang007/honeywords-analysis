
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
        print('==> loaded data.')
        self.char_map = char_map
        self.char2digits = char2digits

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):

        x = random.sample(self.data[idx], 2)
        x = self.char2digits(x, self.char_map, 30, add_sos=False)
        x1 = x[0]
        x2 = x[1]
        
        return x1, x2


class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(SimCLR_Loss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):

        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)
        # print(z.shape)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # print(sim.shape)

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        
        #SIMCLR
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long() #.float()
        
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss
    
    
def train_model():

    model = Transformer_embedding(args.char_size, args.out_dim, args.num_layer, args.d_model, args.nhead, args.embedding_dim)
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

    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print('==> learning rate is {}'.format(lr))

    tic0 = time()
    step = 0
    for epoch in range(args.n_epoch):
        print('====== start {}-th epoch ======='.format(epoch + 1))

        for x1, x2 in tqdm(train_dataset):

            # train the model
            optimizer.zero_grad()
            model.train()

            x1 = x1.to(torch.int64).to(device)
            x2 = x2.to(torch.int64).to(device)
            
            y1 = model(x1)
            y2 = model(x2)
            
            criterion = SimCLR_Loss(batch_size = x1.shape[0], temperature = 0.5)
            loss = criterion(y1, y2)

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
        dic = {'model': model.module.state_dict(), 'out_dim': args.out_dim, 'num_layer': args.num_layer, 'd_model': args.d_model, 'nhead': args.nhead, 'embedding_dim': args.embedding_dim}
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
    parser.add_argument('--embedding_dim', type=int, default=64)
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
