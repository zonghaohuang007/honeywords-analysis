
import numpy as np

import torch
import torch.nn as nn
from mymodels import *
from utils import *
from utils_pwd import *
from chaffing import *
import argparse
from math import log2
import json


torch.set_num_threads(1)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def generate_hw(pw, beta):
    if args.hw_gen_method == 'rnn':
        sw = []
        num = 0
        while num < beta:
            hw = hw_generator.sample(pw, False)
            if hw != pw and hw not in sw:
                sw.append(hw)
                num = num + 1
    elif args.hw_gen_method == 'rnn*':
        sw = []
        num = 0
        while num < beta:
            hw = hw_generator.sample(pw, True)
            if hw != pw and hw not in sw:
                sw.append(hw)
                num = num + 1
    elif args.hw_gen_method == 'CBT3':
        sw = []
        num = 0
        while num < beta:
            hw = tweak_t_position(pw, t=3)
            if hw != pw and hw not in sw:
                sw.append(hw)
                num = num + 1
    elif args.hw_gen_method == 'CBT4':
        sw = []
        num = 0
        while num < beta:
            hw = tweak_t_position(pw, t=4)
            if hw != pw and hw not in sw:
                sw.append(hw)
                num = num + 1
    elif args.hw_gen_method == 'CBT*':
        sw = []
        num = 0
        num_ = 0
        while num < args.n_hw - 1:
            if num_ > 1000000:
                hw = tweak_t_position(pw, 3)  # to resolve the issues that some passwords cannot be used to generate k honeywords
            else:
                hw = chafffing_by_tweak(pw)
                num_ += 1
            if hw != pw and hw not in sw:
                sw.append(hw)
                num = num + 1
    elif args.hw_gen_method == 'CHM':
        sw = chaffing_with_a_hybrid_model(hw_generator, pw, beta+1, 10)[1:]
    elif args.hw_gen_method == 'list':
        sw, _ = hw_generator.generate_honeywords(pw, beta, sample=True)
    elif args.hw_gen_method == 'list*':
        sw, _ = hw_generator.generate_honeywords(pw, beta, sample=False)
    elif args.hw_gen_method == 'pcfg':
        # pcfg_model()
        print('please refer to https://github.com/lakiw/pcfg_cracker')
    elif args.hw_gen_method == 'pcfg*':
        args.pw_dependent = True
        # pcfg_model()
        print('please refer to https://github.com/lakiw/pcfg_cracker')
    elif args.hw_gen_method == 'markov':
        num = 0
        sw = []
        while num < beta:
            hw = generate_markov(hw_generator[0], hw_generator[1])
            if hw != pw and hw not in sw:
                if args.pw_dependent:
                    if len(hw) == len(pw):
                        sw.append(hw)
                        num = num + 1
                else:
                    sw.append(hw)
                    num = num + 1
    elif args.hw_gen_method == 'markov*':
        args.pw_dependent = True
        num = 0
        sw = []
        while num < beta:
            hw = generate_markov(hw_generator[0], hw_generator[1])
            if hw != pw and hw not in sw:
                if args.pw_dependent:
                    if len(hw) == len(pw):
                        sw.append(hw)
                        num = num + 1
                else:
                    sw.append(hw)
                    num = num + 1
    elif args.hw_gen_method == 'tweak':
        sw = []
        num = 0
        while num < beta:
            hw = hw_generator.sample(pw)
            if hw not in sw and hw != pw:
                sw.append(hw)
                num = num + 1
    elif args.hw_gen_method == 'pass2path':
        sw = []
        num = 0
        while num < beta:
            hw = hw_generator.sample(pw)
            if hw not in sw and hw != pw:
                sw.append(hw)
                num = num + 1

    return sw 


if __name__ == '__main__':

    # get parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=int, default=[1000])
    parser.add_argument('--hw_gen_method', type=str, default='rnn')
    parser.add_argument('--n_hw', type=int, default=20)
    args = parser.parse_args()
   
    args.data_path = './hw/{}(hw:{})'.format(args.hw_gen_method, args.n_hw)

    # load CM map: a dictionary to map char or special symbol or number into digit
    char_map, char_map_inv = load_charmap('./char_map.pickle')
    char_size = len(char_map) + 1
    id_sos = len(char_map)
    args.char_size = char_size

    # load model
    if args.hw_gen_method == 'rnn':
        # load model
        dic = torch.load('./saved_model/rnn_model/model', map_location='cuda:0')
        model = RNN_Model(args.char_size, dic['dim'], dic['num_layer'])
        model.load_state_dict(dic['model'])
        model.eval()
        model.to(device)
        # generator
        hw_generator = Generator(model, char_map, char_map_inv, char2digits, digits2char, 30, 1)
          
    elif args.hw_gen_method == 'rnn*':
        # load model
        dic = torch.load('./saved_model/rnn_model/model', map_location='cuda:0')
        model = RNN_Model(args.char_size, dic['dim'], dic['num_layer'])
        model.load_state_dict(dic['model'])
        model.eval()
        model.to(device)
        # generator
        hw_generator = Generator(model, char_map, char_map_inv, char2digits, digits2char, 30, 1)

    elif args.hw_gen_method == 'CHM':
        hw_generator = fasttext.load_model("/Saved_model/model_trained_on_4iq_500_epochs.bin")

    elif args.hw_gen_method == 'list':
        # load data
        with open('/BreachCompilationAnalysis/preprocessed_data/train_data.pickle', 'rb') as f:
            data = pickle.load(f)

        data_pool = [j for i in data for j in i]

        # build list model
        hw_generator = list_model(char_map_inv)
        hw_generator.load_data(data_pool)

    elif args.hw_gen_method == 'list*':
        # load data
        with open('/BreachCompilationAnalysis/preprocessed_data/train_data.pickle', 'rb') as f:
            data = pickle.load(f)

        data_pool = [j for i in data for j in i]

        # build list model
        hw_generator = list_model(char_map_inv)
        hw_generator.load_data(data_pool)

    elif args.hw_gen_method == 'pcfg':
        # pcfg_model()
        print('please refer to https://github.com/lakiw/pcfg_cracker')

    elif args.hw_gen_method == 'pcfg*':
        args.pw_dependent = True
        # pcfg_model()
        print('please refer to https://github.com/lakiw/pcfg_cracker')

    elif args.hw_gen_method == 'markov':
        # markov model
        print('LOADING markov model...')
        with open('./saved_model/markov_model.pickle', 'rb') as f:
            model = pickle.load(f)
        chain = model['chain']
        starts = model['starts']
        hw_generator = [chain, starts]

    elif args.hw_gen_method == 'markov*':

        # markov model
        print('LOADING markov model...')
        with open('./saved_model/markov_model.pickle', 'rb') as f:
            model = pickle.load(f)
        chain = model['chain']
        starts = model['starts']
        hw_generator = [chain, starts]

    elif args.hw_gen_method == 'tweak':
        # load model
        dic = torch.load('./saved_model/tweak_model/model', map_location='cuda:0')
        model = Transformer_model(args.char_size, dic['out_dim'], dic['num_layer'], dic['d_model'], dic['nhead'])
        model.load_state_dict(dic['model'])
        model.to(device)

        # generator
        hw_generator = Generator_tweak(model, None, char_map, char_map_inv, char2digits, digits2char, 30, -1)

    elif args.hw_gen_method == 'pass2path':
        with open('trans_dict_2path.json', 'r') as f:
            trans_dict_2path = json.load(f)
        args.vocab_size = len(trans_dict_2path) + 2

        with open('idx_look_up.json', 'r') as f:
            look_up = json.load(f)

        # load model
        dic = torch.load('./saved_model/pass2path_model/model', map_location='cuda:0')
        model = PASS2PATH(args.char_size, args.vocab_size, dic['embed_dim'], dic['lstm_dim'], dic['num_layer'], dic['dropout_rate'])
        model.load_state_dict(dic['model'])
        model.to(device)
        hw_generator = Generator_pass2path(model, char_map, char_map_inv, char2digits, path2word, trans_dict_2path, idx2path, look_up, 30, args)

    # load data
    with open(args.data_path + '/sweetwords.pickle', 'rb') as f:
        sw = pickle.load(f)
    with open(args.data_path + '/password_index.pickle', 'rb') as f:
        pw_index = pickle.load(f)
    with open(args.data_path + '/aux_passwords.pickle', 'rb') as f:
        aux_pw = pickle.load(f)
    print('==> loaded data successfully.')

    max_beta = max(args.beta)

    fg_success_track = {}
    fg_guess_track = {}
    for i in args.beta:
        alpha = min(args.n_hw-1, i)
        fg_success_track[i] = np.zeros(shape=[alpha, ], dtype=np.float32)
        fg_guess_track[i] = np.array(list(range(1, alpha + 1)))

    num = 0
    tic0 = time()
    for i in range(len(sw)): 
        hw = sw[i]
        input_pw = hw[pw_index[i]]
 
        generated_hw = generate_hw(input_pw, max_beta)
        assert len(generated_hw) == max_beta, 'error'
        for k in args.beta:
            alpha = min(args.n_hw-1, k)
            success_track = np.zeros(shape=[alpha, ], dtype=np.float32)
            success_n = 0
            for j in generated_hw[:k]:
                if j in hw and j != hw[pw_index[i]]:
                    success_n += 1
            for j in range(success_n):
                success_track[j] = 1
            fg_success_track[k] = (fg_success_track[k] * num + success_track) / (num + 1)
        num = num + 1
        if num % 100 == 0:
            print('==> finished {} hw evaluations | cost time: {} sec/sw'.format(num, (time() - tic0) / 100))
            for k in args.beta:
                plt = imagesc([fg_guess_track[k], fg_success_track[k]], label=args.hw_gen_method)
            tic0 = time()
    for k in args.beta:
        plt = imagesc([fg_guess_track[k], fg_success_track[k]], label=args.hw_gen_method)
        plt.savefig(args.data_path + '/' + 'FP.png')

    with open(args.data_path + '/FP.pickle', 'wb') as f:
        pickle.dump([fg_guess_track, fg_success_track], f)
    
    print('==> finished.')
    