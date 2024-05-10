
import comet_ml
import numpy as np
from comet_ml import Experiment

import torch
import torch.nn as nn
from utils import *
import argparse
from math import log2, log10, exp
import os
from time import time


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
        self.fc1 = nn.Linear(self.dim, 64)
        self.drop_out = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        prev_state = (torch.zeros(self.num_layer, x.shape[0], self.dim).to(device), torch.zeros(self.num_layer, x.shape[0], self.dim).to(device))       
        embed = self.embedding(x)
        output, _ = self.lstm(embed, prev_state)
        output = self.drop_out(output)
        output = self.fc1(output)
        output = self.drop_out(output)
        output = nn.ReLU()(output)
        output = self.fc2(output)
        output = output[:, -1, :]
        return output


if __name__ == '__main__':

    # get parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_mode', type=str, default='fg')  
    parser.add_argument('--hw_model', type=str, default='lastpass-1pass_new')
    parser.add_argument('--data_path', type=str, default='./data/new/target_data.pickle')
    parser.add_argument('--pw_dependent', type=bool, default=False)
    args = parser.parse_args()

    exp_name = 'attack_{}_classifier(mode:{})'.format(args.hw_model, args.eval_mode)

    args.name_file = args.data_path.split('/')[-1].split('.')[0]
    if args.pw_dependent:
        args.data_path = './hw/' + args.hw_model + '/' + args.name_file + '/pw_dependent'
    else:
        args.data_path = './hw/' + args.hw_model + '/' + args.name_file

    # args.data_path = './hw/lastpass-1pass_new(hw:100)/fixed'

    distribution = {2: 64.30002293626886, 3: 17.670457134678124, 4: 6.637033680438871, 5: 4.627505442546224, 6: 2.3938560999288074, 7: 1.527684351160744, 8: 0.8832319100280378, 9: 0.5155777777116425, 10: 0.3748350206600237, 11: 0.25824193772231524, 12: 0.17962991936347855, 13: 0.12592512387108964, 14: 0.09609842841151271, 15: 0.07191914659703152, 16: 0.0581485029659999, 17: 0.045538174910788806, 18: 0.03722862060229485, 19: 0.03019270789594443, 20: 0.026133390052158714, 21: 0.022608542251436832, 22: 0.020185305102989798, 23: 0.018755967834051163, 24: 0.01697440102383837, 25: 0.014701244288629891, 26: 0.012607775667230843, 27: 0.01077158775424075, 28: 0.009225351115806779, 29: 0.007946094260106703, 30: 0.006963424887711392}
    x_num = []
    x_pro = []
    for k, v in distribution.items():
        x_num.append(k - 1)
        x_pro.append(v / 100)

    # load CM map: a dictionary to map char or special symbol or number into digit
    char_map, char_map_inv = load_charmap('./char_map.pickle')
    char_size = len(char_map) + 1
    id_sos = len(char_map)
    args.char_size = char_size

    # set up comet ml experiment
    print('==> setting up comet experiment...')
    experiment = Experiment(project_name='Honeywords-evaluation', auto_param_logging=False,
                            api_key= "U1kuka6SA58EBpCa8Ct6CY1fp",
                            auto_metric_logging=False,
                            parse_args=False)
    comet_ml.config.experiment = None
    experiment.set_name(exp_name)
    experiment.add_tag('')
    experiment.log_parameters(vars(args))

    with open('./data/new/classes_name.pickle', 'rb') as f:
        classes_name = pickle.load(f)

    # load model
    dic = torch.load('./saved_model/classifier_lastpass-1pass_new38/model', map_location='cuda:0')
    model = RNN_Model(args.char_size, dic['dim'], dic['num_layer'], len(classes_name)-1)
    model.load_state_dict(dic['model'])
    model.eval()
    model.to(device)

    if True:
        
        data_path = args.data_path
        # load data
        with open(data_path + '/sweetwords.pickle', 'rb') as f:
            sw = pickle.load(f)
        with open(data_path + '/password_index.pickle', 'rb') as f:
            pw_index = pickle.load(f)
        with open(data_path + '/aux_passwords.pickle', 'rb') as f:
            aux_pw = pickle.load(f)
        print('==> loaded data successfully.')

        if True:

            fg_success_track = np.zeros(shape=[len(sw[0]), ], dtype=np.float32)
            fg_guess_track = list(range(1, len(sw[0]) + 1))
            fg_guess_track = np.array(fg_guess_track)
            num = 0
            tic0 = time()

            acc = 0

            for i in range(len(sw)):
                
                num_aux = np.random.choice(x_num, p=x_pro)

                hw = sw[i]
                pw = hw[pw_index[i]]
                y = aux_pw[i][:num_aux]

                for j in range(len(hw)):
                    special_chars = []
                    for z in hw[j]:
                        if z not in char_map.keys() and z not in special_chars:
                            special_chars.append(z)
                    if len(special_chars) != 0:
                        for z in special_chars:
                            hw[j] = hw[j].replace(z,'')

                preds = np.zeros(shape=[len(classes_name)-1,])
                pros = np.zeros(shape=[len(classes_name)-1,])
                for j in y:
                    pw_ = char2digits([j], char_map, 14, add_sos=False)
                    with torch.no_grad():
                        pw_tensor = torch.Tensor(pw_).to(torch.int64).to(device)
                        output = nn.Softmax(dim=1)(model(pw_tensor))
                        pred = int(torch.argmax(output, dim=1)[0].cpu().numpy())
                    pros[pred] += output[0][pred].cpu().numpy()
                    preds[pred] += 1

                sorted_index = sorted(list(range(len(preds))), key=lambda e: (preds[e], pros[e]))[::-1]
                prediction = sorted_index[0]

                success_track = np.zeros(shape=[len(sw[0]), ], dtype=np.float32)
                scores = []
                for j in hw:
                    if len(j) == 14:
                        pw_ = char2digits([j], char_map, 14, add_sos=False)
                        with torch.no_grad():
                            pw_tensor = torch.Tensor(pw_).to(torch.int64).to(device)
                            output = model(pw_tensor)
                            output = nn.Softmax(dim=1)(output)
                            scores.append(output[0][prediction].item())
                    else:
                        scores.append(0)
      
                sort_index = np.argsort(scores)[::-1]
                guess_n = 0
                account_break = False
                while not account_break:
                    predicted_id = sort_index[guess_n]
                    predicted_pw = hw[predicted_id]
                    if predicted_id == pw_index[i]:
                        for j in range(guess_n, len(hw)):
                            success_track[j] = 1
                        account_break = True
                    guess_n = guess_n + 1
                fg_success_track = (fg_success_track * num + success_track) / (num + 1)
                num = num + 1
            plt = imagesc([fg_guess_track, fg_success_track], mode='fg', title='flatness graph', experiment=experiment, step=0, label='distribution')
            plt.savefig(data_path + '/' + args.eval_mode + '_distribution.png')
            with open(data_path + '/' + args.eval_mode + '_distribution.pickle', 'wb') as f:
                pickle.dump([fg_guess_track, fg_success_track], f)
            print('finished.')

    experiment.send_notification('finished')
    print('==> finished.')
