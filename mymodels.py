import torch
import torch.nn as nn

import math
import numpy as np
import random
from time import time
import pickle
from tqdm import tqdm


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


'''
architecture of RNN model
'''
class RNN_Model(nn.Module):
    def __init__(self, vocab_size, dim, num_layer):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layer = num_layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim)
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            num_layers=num_layer,
            batch_first = True
        )
        self.fc = nn.Linear(self.dim, vocab_size)

    def forward(self, x, prev_state=None, return_state=False):
        if prev_state is None:
            prev_state = (torch.zeros(self.num_layer, x.shape[0], self.dim).to(device), torch.zeros(self.num_layer, x.shape[0], self.dim).to(device))       
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        if return_state:
            return logits, state
        else:
            return logits


'''
A generator for RNN
'''
class Generator:
    def __init__(self, model, char_map, char_map_inv, char2digits, digits2char, pw_len, num_hw):
        super().__init__()
        self.model = model
        self.char_map = char_map
        self.char_map_inv = char_map_inv
        self.digits2char = digits2char
        self.char2digits = char2digits
        self.pw_len = pw_len
        self.id_sos = len(char_map_inv)
        self.num_hw = num_hw

    def sample(self, pw=None, pw_dependent=False):
        if pw_dependent:
            ti0 = time()
            success = False
            while not success:
                hw = []
                inputs_rnn = [[self.id_sos]]  # initial input shape is [1, 1, ]
                tensor_input = torch.Tensor(inputs_rnn).to(device).to(torch.int64)  # start of string
                states = None
                pro = 1
                for i in range(len(pw)):
                    # predictions.shape is [1, 1, vocab_size]
                    with torch.no_grad():
                        predictions, states = self.model(tensor_input, prev_state=states, return_state=True)
                    predictions = nn.Softmax(dim=2)(predictions)
                    predictions = torch.squeeze(predictions).cpu().numpy()  # shape of [vocab_size, ]
                    predictions[self.id_sos] = 0
                    predictions[0] = 0
                    predictions = predictions / np.sum(predictions)
                    if np.all((predictions == 0)):
                        break
                    else:
                        predicted_ids = np.random.choice(range(len(self.char_map_inv)+1), p=predictions)
                        tensor_input = torch.Tensor([[predicted_ids]]).to(device).to(torch.int64)
                        hw.append(predicted_ids)
                        pro = pro * predictions[predicted_ids]
                    if i == len(pw) - 1:
                        success = True
            hw = self.digits2char(hw, self.char_map_inv)
            assert len(hw) == len(pw), 'error!'
        else:
            ti0 = time()
            hw = []
            inputs_rnn = [[self.id_sos]]  # initial input shape is [1, 1, ]
            tensor_input = torch.Tensor(inputs_rnn).to(device).to(torch.int64)  # start of string
            states = None
            pro = 1
            for i in range(self.pw_len):
                # predictions.shape is [1, 1, vocab_size]
                with torch.no_grad():
                    predictions, states = self.model(tensor_input, prev_state=states, return_state=True)
                predictions = nn.Softmax(dim=2)(predictions)
                predictions = torch.squeeze(predictions).cpu().numpy()  # shape of [vocab_size, ]
                predictions[self.id_sos] = 0
                predictions = predictions / np.sum(predictions)
                if np.all((predictions == 0)):
                    break
                else:
                    predicted_ids = np.random.choice(range(len(self.char_map_inv)+1), p=predictions)
                    tensor_input = torch.Tensor([[predicted_ids]]).to(device).to(torch.int64)
                    hw.append(predicted_ids)
                    pro = pro * predictions[predicted_ids]
                    if predicted_ids == 0:
                        break
            hw = self.digits2char(hw, self.char_map_inv)

        return hw

    def sample_false_alarm(self, pw, sample=True):

        hws = []
        pros = []
        num = 0

        while num < self.num_hw:
            ti0 = time()
            hw = []
            inputs_rnn = [[self.id_sos]]  # initial input shape is [1, 1, ]
            tensor_input = torch.Tensor(inputs_rnn).to(device).to(torch.int64)  # start of string
            states = None
            pro = 1
            for i in range(self.pw_len):
                # predictions.shape is [1, 1, vocab_size]
                with torch.no_grad():
                    predictions, states = self.model(tensor_input, prev_state=states, return_state=True)
                predictions = nn.Softmax(dim=2)(predictions)
                predictions = torch.squeeze(predictions).cpu().numpy()  # shape of [vocab_size, ]
                predictions[self.id_sos] = 0
                predictions = predictions / np.sum(predictions)
                if np.all((predictions == 0)):
                    break
                else:
                    predicted_ids = np.random.choice(range(len(self.char_map_inv)+1), p=predictions)
                    tensor_input = torch.Tensor([[predicted_ids]]).to(device).to(torch.int64)
                    hw.append(predicted_ids)
                    pro = pro * predictions[predicted_ids]
                    if predicted_ids == 0:
                        break
            hw = self.digits2char(hw, self.char_map_inv)
            if hw != pw and hw not in hws:
                if not sample and len(hw) == len(pw): 
                    hws.append(hw)
                    pros.append(pro)
                    num = num + 1
                else:
                    hws.append(hw)
                    pros.append(pro)
                    num = num + 1

        sorted_idx = np.argsort(pros)[::-1]
        hws = [hws[idx] for idx in sorted_idx]

        return [pw] + hws


class Password_Meter:
    def __init__(self, model, char_map_inv, char_map, char2digits, pw_len):
        super().__init__()
        self.model = model
        self.char_map_inv = char_map_inv
        self.char_map = char_map
        self.char2digits = char2digits
        self.pw_len = pw_len
        self.vocab_size = len(char_map) + 1
        self.id_sos = len(char_map)

    def measure(self, pw):
        input_data = self.char2digits([pw], self.char_map, self.pw_len)
        input_seq = [self.id_sos] + list(input_data[0])
        next_seq = list(input_data[0]) + [0]
        probability = 1
        tensor_input = torch.Tensor([input_seq]).to(device).to(torch.int64)
        with torch.no_grad():
            output = self.model(tensor_input)
            output = nn.Softmax(dim=-1)(output).cpu().numpy()
        for i in range(self.pw_len + 1):
            if next_seq[i] != 0: 
                probability = probability * output[0, i, int(next_seq[i])]
            else:
                break
        return probability


'''
Architecture of pass2path
'''
class PASS2PATH(nn.Module):
    def __init__(self, vocab_size, vocab_size2, embed_dim, lstm_dim, num_layer, dropout_rate):
        super().__init__()
        self.vocab_size = vocab_size
        self.vocab_size2 = vocab_size2
        self.embed_dim = embed_dim
        self.lstm_dim = lstm_dim
        self.num_layer = num_layer
        self.embedding1 = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.embedding2 = nn.Embedding(num_embeddings=vocab_size2, embedding_dim=embed_dim)
        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_dim,
            num_layers=num_layer,
            dropout=dropout_rate,
            batch_first = True
        )
        self.decoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_dim,
            num_layers=num_layer,
            dropout=dropout_rate,
            batch_first = True
        )
        self.fc = nn.Linear(self.lstm_dim, vocab_size2)

    def forward(self, y, x=None, prev_state=None, encoder_state=None, return_state=False):
        assert (x is not None and prev_state is None) or (x is None and prev_state is not None)
        if x is not None:
            embedx = self.embedding1(x)
            output, prev_state = self.encoder(embedx, encoder_state)
        embedy = self.embedding2(y)
        output, state = self.decoder(embedy, prev_state)
        logits = self.fc(output)
        if return_state:
            return logits, state
        else:
            return logits


'''
Generator of pass2path
'''
class Generator_pass2path:
    def __init__(self, model, char_map, char_map_inv, char2digits, path2word, trans_dict_2path, idx2path, idx_look_up, pw_len, args):
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
        self.args = args

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
            encoder_state = (torch.zeros(self.args.num_layer, tgt_in.shape[0], self.args.lstm_dim).to(device), torch.zeros(self.args.num_layer, tgt_in.shape[0], self.args.lstm_dim).to(device))

            for i in range(self.pw_len):
                # predictions.shape is [1, 1, vocab_size]
                with torch.no_grad():
                    if i == 0:
                        predictions, states = self.model(tgt_in, x=src_in, encoder_state=encoder_state, return_state=True)
                    else:
                        predictions, states = self.model(tgt_in, prev_state=states, return_state=True)
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
            encoder_state = (torch.zeros(self.args.num_layer, tgt_in.shape[0], self.args.lstm_dim).to(device), torch.zeros(self.args.num_layer, tgt_in.shape[0], self.args.lstm_dim).to(device))

            for i in range(self.pw_len):
                # predictions.shape is [1, 1, vocab_size]
                with torch.no_grad():
                    if i == 0:
                        predictions, states = self.model(tgt_in, x=src_in, encoder_state=encoder_state, return_state=True)
                    else:
                        predictions, states = self.model(tgt_in, prev_state=states, return_state=True)
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

            if hw not in hws and hw != password and len(hw) <= 30:
                hws.append(hw)
                pros.append(pro)
                num = num + 1

        sorted_idx = np.argsort(pros)[::-1]
        hws = [hws[idx] for idx in sorted_idx]
        return [password] + hws
    

'''
architecture of similarity model
'''
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


'''
architecture of tweak
'''
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


class Generator_tweak:
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


'''
list model
'''
class list_model:
    def __init__(self, char_map_inv):
        self.size = None
        self.model_dict = None
        self.frequency = None
        self.data_list = None
        self.char_map_inv = char_map_inv

    def load_data(self, data):
        print('==> start loading data...')
        model_dict = {}
        frequency = {}
        frequency_by_length = {}
        self.size = len(data)
        for i in tqdm(range(len(data))):
            if data[i] not in model_dict.keys():
                model_dict[data[i]] = 1 / self.size
                frequency[data[i]] = 1
            else:
                model_dict[data[i]] = model_dict[data[i]] + 1 / self.size
                frequency[data[i]] = frequency[data[i]] + 1
            if len(data[i]) not in frequency_by_length.keys():
                frequency_by_length[len(data[i])] = {}            
            if data[i] not in frequency_by_length[len(data[i])].keys():
                frequency_by_length[len(data[i])][data[i]] = 1
            else:
                frequency_by_length[len(data[i])][data[i]] += 1

        sorted_model_dict = {k: v for k, v in sorted(model_dict.items(), key=lambda item: item[1], reverse=True)}
        self.model_dict = sorted_model_dict
        sorted_frequency = {k: v for k, v in sorted(frequency.items(), key=lambda item: item[1], reverse=True)}
        self.frequency = sorted_frequency
        self.frequency_by_length = {}
        for i in frequency_by_length.keys():
            self.frequency_by_length[i] = {k: v for k, v in sorted(frequency_by_length[i].items(), key=lambda item: item[1], reverse=True)}
        self.data_list = data
        data_bin = {}
        freq = {}
        for k, v in self.frequency.items():
            if str(v) in data_bin:
                data_bin[str(v)].append(k)
            else:
                data_bin[str(v)] = [k]
            if str(v) in freq:
                freq[str(v)] = freq[str(v)] + 1
            else:
                freq[str(v)] = 1
        self.data_bin = data_bin
        print(freq)
        print('==> finished loaded data')

    def predict_strength(self, pw):
        model_dict = self.model_dict
        if pw in model_dict.keys():
            return model_dict[pw]
        else:
            return 1 / (self.size + 1)

    def count_frequency(self, pw):
        return self.frequency[pw]

    def generate_honeywords(self, pw, num, sample=True):
        if sample:
            hws = []
            sample_index = random.sample(range(self.size), num)
            for i in sample_index:
                hws.append(self.data_list[i])
            hws.append(pw)
            assert len(hws) == num + 1, 'the number of hws is not enough'
            order = list(range(len(hws)))
            random.shuffle(order)
            shuffled_hws = [hws[o] for o in order]
            pw_index = shuffled_hws.index(pw)
            return shuffled_hws, pw_index
        else:
            # obtain password strength according to list model
            hws = []
            k = 0
            while k < num:
                sample_index = random.sample(range(self.size), 1)
                sampled_hw = self.data_list[sample_index[0]]
                if sampled_hw != pw and len(sampled_hw) == len(pw):
                    hws.append(sampled_hw)
                    k = k + 1
            hws.append(pw)
            assert len(hws) == num + 1, 'the number of hws is not enough'
            order = list(range(len(hws)))
            random.shuffle(order)
            shuffled_hws = [hws[o] for o in order]
            pw_index = shuffled_hws.index(pw)
            return shuffled_hws, pw_index

    def false_alarm(self, num, pw=None, sample=True):
        hws = []
        if not sample:
            n = 0
            while n < num:
                sample_index = random.sample(range(self.size), 1)
                if self.data_list[sample_index[0]] != pw and len(self.data_list[sample_index[0]]) == len(pw):
                    hws.append(self.data_list[0])
                    n = n + 1
        else:
            sample_index = random.sample(range(self.size), num)
            for i in sample_index:
                hws.append(self.data_list[i])
        assert len(hws) == num, 'the number of hws is not enough'
    
        return hws