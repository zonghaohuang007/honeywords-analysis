import torch
import torch.nn as nn

import math
import numpy as np
import random
from time import time
import pickle
from tqdm import tqdm


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Transformer_embedding(nn.Module):
    def __init__(self, vocab_size, out_dim, num_layer, d_model, nhead, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.out_dim = out_dim
        self.num_layer = num_layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=self.out_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=self.num_layer)
        self.dense = nn.Linear(30*d_model, embedding_dim)

    def forward(self, inputs):
        x = self.embedding(inputs)
        output = self.encoder(x)
        output = torch.flatten(output, start_dim=1)
        output = self.dense(output)
        return output


class Transformer_model(nn.Module):
    def __init__(self, vocab_size, out_dim, num_layer, d_model, nhead):
        super().__init__()
        self.vocab_size = vocab_size
        self.out_dim = out_dim
        self.num_layer = num_layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, dim_feedforward=self.out_dim, num_encoder_layers=self.num_layer, num_decoder_layers=self.num_layer, batch_first=True)
        self.dense = nn.Linear(d_model, vocab_size)

    def forward(self, inputs, mask):
        x, y = inputs
        x = self.embedding(x)
        y = self.embedding(y)
        output = self.transformer(x, y, tgt_mask=mask)
        output = self.dense(output)
        return output


class Transformer_pass2path_model(nn.Module):
    def __init__(self, char_size, vocab_size, out_dim, num_layer, d_model, nhead):
        super().__init__()
        self.char_size = char_size
        self.vocab_size = vocab_size
        self.out_dim = out_dim
        self.num_layer = num_layer
        self.embedding = nn.Embedding(num_embeddings=char_size, embedding_dim=d_model)
        self.embedding2 = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, dim_feedforward=self.out_dim, num_encoder_layers=self.num_layer, num_decoder_layers=self.num_layer, batch_first=True)
        self.dense = nn.Linear(d_model, vocab_size)

    def forward(self, inputs, mask):
        x, y = inputs
        x = self.embedding(x)
        y = self.embedding2(y)
        output = self.transformer(x, y, tgt_mask=mask)
        output = self.dense(output)
        return output


class Generator:
    def __init__(self, model, metric_model, char_map, char_map_inv, char2digits, digits2char, pw_len, eps):
        super().__init__()
        self.model = model
        self.metric_model = metric_model
        self.eps = eps
        self.char_map = char_map
        self.char_map_inv = char_map_inv
        self.digits2char = digits2char
        self.char2digits = char2digits
        self.pw_len = pw_len
        self.id_sos = len(char_map_inv)

    def sample(self, password):
        if self.eps > -1:
            A = self.char2digits([password], self.char_map, self.pw_len, add_sos=False)
            with torch.no_grad():
                A = torch.Tensor(A).cuda().to(torch.int64)
                A = self.metric_model(A)

        while True:
            hw = []
            ti0 = time()
            input_data = self.char2digits([password], self.char_map, self.pw_len)
            src_in = torch.Tensor(input_data).to(device).to(torch.int64)
            pro = 1
            tgt_input = [[self.id_sos]]  # initial input shape is [1, 1, ]
            tgt_in = torch.Tensor(tgt_input).to(device).to(torch.int64)  # start of string
            mask = (torch.ones(tgt_in.shape[1], tgt_in.shape[1]) - torch.triu(torch.ones(tgt_in.shape[1], tgt_in.shape[1]))).to(torch.bool).transpose(0, 1).to(device)
            for i in range(self.pw_len):
                # predictions.shape is [1, 1, vocab_size]
                with torch.no_grad():
                    predictions = self.model([src_in, tgt_in], mask)
                predictions = nn.Softmax(dim=2)(predictions)
                predictions = torch.squeeze(predictions[0, -1, :]).cpu().numpy()  # shape of [vocab_size, ]
                predictions[self.id_sos] = 0
                predictions = predictions / np.sum(predictions)
                if np.all((predictions == 0)):
                    break
                else:
                    predicted_ids = np.random.choice(range(len(self.char_map_inv)+1), p=predictions)
                    pro = pro * predictions[predicted_ids]
                    tgt_in = torch.cat((tgt_in, torch.Tensor([[predicted_ids]]).to(device).to(torch.int64)), dim=1)
                    mask = (torch.ones(tgt_in.shape[1], tgt_in.shape[1]) - torch.triu(torch.ones(tgt_in.shape[1], tgt_in.shape[1]))).to(torch.bool).transpose(0, 1).to(device)
                    hw.append(predicted_ids)
                    if predicted_ids == 0:
                        break
            hw = self.digits2char(hw, self.char_map_inv)
            if self.eps > -1:
                B = self.char2digits([hw], self.char_map, self.pw_len, add_sos=False)
                with torch.no_grad():
                    B = torch.Tensor(B).cuda().to(torch.int64)
                    B = self.metric_model(B)
                    score = nn.CosineSimilarity()(A, B).cpu().numpy()[0]
                if score > self.eps and hw != password:
                    break
            else:
                if hw != password:
                    break
        return hw

    def sample_list(self, password, num_hw):

        if self.eps > -1:
            A = self.char2digits([password], self.char_map, self.pw_len, add_sos=False)
            with torch.no_grad():
                A = torch.Tensor(A).cuda().to(torch.int64)
                A = self.metric_model(A)

        num = 0
        hws = []
        pros = []
        ti0 = time()
        input_data = self.char2digits([password], self.char_map, self.pw_len)
        src_in = torch.Tensor(input_data).to(device).to(torch.int64)
        while num < num_hw:
            hw = []
            pro = 1
            tgt_input = [[self.id_sos]]  # initial input shape is [1, 1, ]
            tgt_in = torch.Tensor(tgt_input).to(device).to(torch.int64)  # start of string
            mask = (torch.ones(tgt_in.shape[1], tgt_in.shape[1]) - torch.triu(torch.ones(tgt_in.shape[1], tgt_in.shape[1]))).to(torch.bool).transpose(0, 1).to(device)
            for i in range(self.pw_len):
                # predictions.shape is [1, 1, vocab_size]
                with torch.no_grad():
                    predictions = self.model([src_in, tgt_in], mask)
                predictions = nn.Softmax(dim=2)(predictions)
                predictions = torch.squeeze(predictions[0, -1, :]).cpu().numpy()  # shape of [vocab_size, ]
                predictions[self.id_sos] = 0
                predictions = predictions / np.sum(predictions)
                if np.all((predictions == 0)):
                    break
                else:
                    predicted_ids = np.random.choice(range(len(self.char_map_inv)+1), p=predictions)
                    pro = pro * predictions[predicted_ids]
                    tgt_in = torch.cat((tgt_in, torch.Tensor([[predicted_ids]]).to(device).to(torch.int64)), dim=1)
                    mask = (torch.ones(tgt_in.shape[1], tgt_in.shape[1]) - torch.triu(torch.ones(tgt_in.shape[1], tgt_in.shape[1]))).to(torch.bool).transpose(0, 1).to(device)
                    hw.append(predicted_ids)
                    if predicted_ids == 0:
                        break
            hw = self.digits2char(hw, self.char_map_inv)

            if self.eps > -1:
                B = self.char2digits([hw], self.char_map, self.pw_len, add_sos=False)
                with torch.no_grad():
                    B = torch.Tensor(B).cuda().to(torch.int64)
                    B = self.metric_model(B)
                    score = nn.CosineSimilarity()(A, B).cpu().numpy()[0]
                if score > self.eps and hw not in hws and hw != password:
                    hws.append(hw)
                    pros.append(pro)
                    num = num + 1
            else:
                if hw not in hws and hw != password:
                    hws.append(hw)
                    pros.append(pro)
                    num = num + 1

        sorted_idx = np.argsort(pros)[::-1]
        hws = [hws[idx] for idx in sorted_idx]
        return [password] + hws


class Generator_pass2path:
    def __init__(self, model, char_map, char_map_inv, char2digits, path2word, trans_dict_2path, idx2path, idx_look_up, pw_len):
        super().__init__()
        self.model = model
        self.char_map = char_map
        self.char_map_inv = char_map_inv
        self.char2digits = char2digits
        self.trans_dict_2path = trans_dict_2path
        self.look_up = idx_look_up
        self.idx2path = idx2path
        self.path2word = path2word
        self.pw_len = pw_len
        self.id_sos = len(char_map_inv)
        self.vocab_size = len(trans_dict_2path) + 2

        self.look_up[str(self.vocab_size-1)] = []

    def sample(self, password):

        pwd_len= len(password)

        while True:
            hw = []
            ti0 = time()
            input_data = self.char2digits([password], self.char_map, self.pw_len)
            src_in = torch.Tensor(input_data).to(device).to(torch.int64)
            pro = 1
            predicted_ids = self.vocab_size-1 
            tgt_input = [[predicted_ids]]  # initial input shape is [1, 1, ]
            tgt_in = torch.Tensor(tgt_input).to(device).to(torch.int64)  # start of string
            mask = (torch.ones(tgt_in.shape[1], tgt_in.shape[1]) - torch.triu(torch.ones(tgt_in.shape[1], tgt_in.shape[1]))).to(torch.bool).transpose(0, 1).to(device)
            for i in range(self.pw_len):
                # predictions.shape is [1, 1, vocab_size]
                with torch.no_grad():
                    predictions = self.model([src_in, tgt_in], mask)
                predictions = nn.Softmax(dim=2)(predictions)
                predictions = torch.squeeze(predictions[0, -1, :]).cpu().numpy()  # shape of [vocab_size, ]
                predictions[self.vocab_size-1] = 0
                for j in range(1, self.vocab_size):
                    if (j + 30) % 31 >= pwd_len:
                        predictions[j] = 0
                table = self.look_up[str(predicted_ids)]
                for j in range(len(table)):
                    predictions[table[j]] = 0
                if i == 0:
                    predictions[0] = 0
                predictions = predictions / np.sum(predictions)
                if np.all((predictions == 0)):
                    break
                else:
                    predicted_ids = np.random.choice(range(self.vocab_size), p=predictions)
                    pro = pro * predictions[predicted_ids]
                    tgt_in = torch.cat((tgt_in, torch.Tensor([[predicted_ids]]).to(device).to(torch.int64)), dim=1)
                    mask = (torch.ones(tgt_in.shape[1], tgt_in.shape[1]) - torch.triu(torch.ones(tgt_in.shape[1], tgt_in.shape[1]))).to(torch.bool).transpose(0, 1).to(device)
                    hw.append(predicted_ids)
                    if predicted_ids == 0:
                        break
            path = []
            for i in hw:
                if i != 0:
                    path.append(i)
                else:
                    break
            hw = self.idx2path(path, self.trans_dict_2path)
            hw = self.path2word(password, hw)
            if hw != password and len(hw) <= 30:
                break
        return hw

    def sample_list(self, password, num_hw):

        pwd_len= len(password)

        num = 0
        hws = []
        pros = []
        ti0 = time()
        input_data = self.char2digits([password], self.char_map, self.pw_len)
        src_in = torch.Tensor(input_data).to(device).to(torch.int64)
        while num < num_hw:
            hw = []
            pro = 1
            predicted_ids = self.vocab_size-1 
            tgt_input = [[predicted_ids]]
            tgt_in = torch.Tensor(tgt_input).to(device).to(torch.int64)  # start of string
            mask = (torch.ones(tgt_in.shape[1], tgt_in.shape[1]) - torch.triu(torch.ones(tgt_in.shape[1], tgt_in.shape[1]))).to(torch.bool).transpose(0, 1).to(device)
            for i in range(self.pw_len):
                # predictions.shape is [1, 1, vocab_size]
                with torch.no_grad():
                    predictions = self.model([src_in, tgt_in], mask)
                predictions = nn.Softmax(dim=2)(predictions)
                predictions = torch.squeeze(predictions[0, -1, :]).cpu().numpy()  # shape of [vocab_size, ]
                predictions[self.vocab_size-1] = 0
                for j in range(1, self.vocab_size):
                    if (j + 30) % 31 >= pwd_len:
                        predictions[j] = 0
                table = self.look_up[str(predicted_ids)]
                for j in range(len(table)):
                    predictions[table[j]] = 0
                predictions = predictions / np.sum(predictions)
                if np.all((predictions == 0)):
                    break
                else:
                    predicted_ids = np.random.choice(range(self.vocab_size), p=predictions)
                    pro = pro * predictions[predicted_ids]
                    tgt_in = torch.cat((tgt_in, torch.Tensor([[predicted_ids]]).to(device).to(torch.int64)), dim=1)
                    mask = (torch.ones(tgt_in.shape[1], tgt_in.shape[1]) - torch.triu(torch.ones(tgt_in.shape[1], tgt_in.shape[1]))).to(torch.bool).transpose(0, 1).to(device)
                    hw.append(predicted_ids)
                    if predicted_ids == 0:
                        break
            path = []
            for i in hw:
                if i != 0:
                    path.append(i)
                else:
                    break
            hw = self.idx2path(path, self.trans_dict_2path)
            hw = self.path2word(password, hw)

            if hw not in hws and hw != password:
                hws.append(hw)
                pros.append(pro)
                num = num + 1

        sorted_idx = np.argsort(pros)[::-1]
        hws = [hws[idx] for idx in sorted_idx]
        return [password] + hws