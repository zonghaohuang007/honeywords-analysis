
import numpy as np
import argparse
import json
from utils import *
from utils_pwd import *
from mymodels import *
import os
from math import log2
from chaffing import *


torch.set_num_threads(1)


def rnn_model():

    # set up paths
    cm_path = './char_map.pickle'

    # load CM map: a dictionary to map char or special symbol or number into digit
    char_map, char_map_inv = load_charmap(cm_path)
    char_size = len(char_map) + 1
    args.char_size = char_size

    # load data
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    # load model
    dic = torch.load('./saved_model/rnn_model/model', map_location='cuda:0')
    model = RNN_Model(args.char_size, dic['dim'], dic['num_layer'])
    model.load_state_dict(dic['model'])
    model.eval()
    model.to(device)

    # generator
    hw_generator = Generator(model, char_map, char_map_inv, char2digits, digits2char, 30, 1)
    print('==> loaded model successfully!')

    sweetwords = []
    password_index = []
    aux_passwords = []

    for i in range(len(data)):

        pw_idx = random.sample(list(range(len(data[i]))), 1)[0]
        pw = data[i][pw_idx]

        aux_pw = [data[i][j] for j in range(len(data[i])) if j != pw_idx]

        sw = []
        num = 0
        while num < args.n_hw - 1:
            hw = hw_generator.sample(pw, args.pw_dependent)
            if hw != pw and hw not in sw:
                sw.append(hw)
                num = num + 1
        
        sw.append(pw)
        assert len(sw) == args.n_hw, 'error!'
        random.shuffle(sw)
        pw_index = sw.index(pw)

        sweetwords.append(sw)
        password_index.append(pw_index)
        aux_passwords.append(aux_pw)

        if (i+1) % 100 == 0:

            print('==> {} set honeywords completed'.format(i+1))

    with open(args.save_path + '/sweetwords.pickle', 'wb') as f:
        pickle.dump(sweetwords, f)
    with open(args.save_path + '/password_index.pickle', 'wb') as f:
        pickle.dump(password_index, f)
    with open(args.save_path + '/aux_passwords.pickle', 'wb') as f:
        pickle.dump(aux_passwords, f)

    print('==> finished.')


def chaffing_by_tweaking(t):

    # load data
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    sweetwords = []
    password_index = []
    aux_passwords = []

    for i in range(len(data)):

        pw_idx = random.sample(list(range(len(data[i]))), 1)[0]
        pw = data[i][pw_idx]

        aux_pw = [data[i][j] for j in range(len(data[i])) if j != pw_idx]

        sw = []
        num = 0
        while num < args.n_hw - 1:
            hw = tweak_t_position(pw, t=t)
            if hw != pw and hw not in sw:
                sw.append(hw)
                num = num + 1
        sw.append(pw)
        assert len(sw) == args.n_hw, 'error'
        random.shuffle(sw)
        pw_index = sw.index(pw)

        sweetwords.append(sw)
        password_index.append(pw_index)
        aux_passwords.append(aux_pw)

        if (i+1) % 100 == 0:
            print('{} set honeywords generation!'.format((i+1)))

    with open(args.save_path + '/sweetwords.pickle', 'wb') as f:
        pickle.dump(sweetwords, f)
    with open(args.save_path + '/password_index.pickle', 'wb') as f:
        pickle.dump(password_index, f)
    with open(args.save_path + '/aux_passwords.pickle', 'wb') as f:
        pickle.dump(aux_passwords, f)
    print('==> finished.')


def chaffing_by_tweaking_random():

    # load data
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    sweetwords = []
    password_index = []
    aux_passwords = []

    for i in range(len(data)):

        pw_idx = random.sample(list(range(len(data[i]))), 1)[0]
        pw = data[i][pw_idx]

        aux_pw = [data[i][j] for j in range(len(data[i])) if j != pw_idx]

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

        sw.append(pw)
        assert len(sw) == args.n_hw, 'error'
        random.shuffle(sw)
        pw_index = sw.index(pw)

        sweetwords.append(sw)
        password_index.append(pw_index)
        aux_passwords.append(aux_pw)

        if (i+1) % 100 == 0:
            print('{} set honeywords generation!'.format((i+1)))

    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    with open(args.save_path + '/sweetwords.pickle', 'wb') as f:
        pickle.dump(sweetwords, f)
    with open(args.save_path + '/password_index.pickle', 'wb') as f:
        pickle.dump(password_index, f)
    with open(args.save_path + '/aux_passwords.pickle', 'wb') as f:
        pickle.dump(aux_passwords, f)
    print('==> finished.')


def chaffing_by_hybrid():

    model = fasttext.load_model("/Saved_model/model_trained_on_4iq_500_epochs.bin")

    # load data
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    sweetwords = []
    password_index = []
    aux_passwords = []

    for i in range(len(data)):

        pw_idx = random.sample(list(range(len(data[i]))), 1)[0]
        pw = data[i][pw_idx]

        aux_pw = [data[i][j] for j in range(len(data[i])) if j != pw_idx]

        sw = chaffing_with_a_hybrid_model(model, pw, args.n_hw, 10)[1:]

        sw.append(pw)
        assert len(sw) == args.n_hw, 'error'
        random.shuffle(sw)
        pw_index = sw.index(pw)

        sweetwords.append(sw)
        password_index.append(pw_index)
        aux_passwords.append(aux_pw)

        if (i+1) % 100 == 0:
            print('{} set honeywords generation!'.format((i+1)))

    with open(args.save_path + '/sweetwords.pickle', 'wb') as f:
        pickle.dump(sweetwords, f)
    with open(args.save_path + '/password_index.pickle', 'wb') as f:
        pickle.dump(password_index, f)
    with open(args.save_path + '/aux_passwords.pickle', 'wb') as f:
        pickle.dump(aux_passwords, f)
    print('==> finished.')


def markov_model():

    # set up paths
    cm_path = './char_map.pickle'

    # load CM map: a dictionary to map char or special symbol or number into digit
    char_map, char_map_inv = load_charmap(cm_path)
    char_size = len(char_map) + 1
    args.char_size = char_size

    # load data
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    # markov model
    print('LOADING markov model...')
    with open('./saved_model/markov_model.pickle', 'rb') as f:
        model = pickle.load(f)
    chain = model['chain']
    starts = model['starts']
    
    print('start generating hws...')

    sweetwords = []
    password_index = []
    aux_passwords = []

    for i in range(len(data)):

        pw_idx = random.sample(list(range(len(data[i]))), 1)[0]
        pw = data[i][pw_idx]

        aux_pw = [data[i][j] for j in range(len(data[i])) if j != pw_idx]

        num = 0
        sw = []
        while num < args.n_hw - 1:
            hw = generate_markov(chain, starts)
            if hw != pw and hw not in sw:
                if args.pw_dependent:
                    if len(hw) == len(pw):
                        sw.append(hw)
                        num = num + 1
                else:
                    sw.append(hw)
                    num = num + 1
        sw.append(pw)
        assert len(sw) == args.n_hw, 'error'
        random.shuffle(sw)
        pw_index = sw.index(pw)

        sweetwords.append(sw)
        password_index.append(pw_index)
        aux_passwords.append(aux_pw)

        if (i+1) % 100 == 0:
            print('{} set honeywords generation!'.format((i+1)))

    with open(args.save_path + '/sweetwords.pickle', 'wb') as f:
        pickle.dump(sweetwords, f)
    with open(args.save_path + '/password_index.pickle', 'wb') as f:
        pickle.dump(password_index, f)
    with open(args.save_path + '/aux_passwords.pickle', 'wb') as f:
        pickle.dump(aux_passwords, f)
    print('==> finished.')


def list_model():

    # set up paths
    cm_path = './char_map.pickle'

    # load CM map: a dictionary to map char or special symbol or number into digit
    char_map, char_map_inv = load_charmap(cm_path)
    char_size = len(char_map) + 1
    args.char_size = char_size

    # load data
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    data_pool = [j for i in data for j in i]

    # build list model
    model = list_model(char_map_inv)
    model.load_data(data_pool)

    sweetwords = []
    password_index = []
    aux_passwords = []

    if args.pw_dependent:
        sample = False
    else:
        sample = True

    for i in range(len(data)):

        pw_idx = random.sample(list(range(len(data[i]))), 1)[0]
        pw = data[i][pw_idx]

        aux_pw = [data[i][j] for j in range(len(data[i])) if j != pw_idx]

        sw, pw_index = model.generate_honeywords(pw, args.n_hw - 1, sample=sample)
        if sw:
            sweetwords.append(sw)
            password_index.append(pw_index)
            aux_passwords.append(aux_pw)
        if (i+1) % 100 == 0:
            print('{} set honeywords generation!'.format((i+1)))

    with open(args.save_path + '/sweetwords.pickle', 'wb') as f:
        pickle.dump(sweetwords, f)
    with open(args.save_path + '/password_index.pickle', 'wb') as f:
        pickle.dump(password_index, f)
    with open(args.save_path + '/aux_passwords.pickle', 'wb') as f:
        pickle.dump(aux_passwords, f)
    print('==> finished.')


def tweak_model():
    # set up paths
    cm_path = './char_map.pickle'

    # load CM map: a dictionary to map char or special symbol or number into digit
    char_map, char_map_inv = load_charmap(cm_path)
    char_size = len(char_map) + 1
    args.char_size = char_size

    # load data
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    # load model
    dic = torch.load('./saved_model/tweak_model/model', map_location='cuda:0')
    model = Transformer_model(args.char_size, dic['out_dim'], dic['num_layer'], dic['d_model'], dic['nhead'])
    model.load_state_dict(dic['model'])
    model.to(device)

    # generator
    hw_generator = Generator_tweak(model, None, char_map, char_map_inv, char2digits, digits2char, 30, -1)
    print('==> loaded model successfully!')

    sweetwords = []
    password_index = []
    aux_passwords = []

    for i in range(len(data)):

        pw_idx = random.sample(list(range(len(data[i]))), 1)[0]
        pw = data[i][pw_idx]

        aux_pw = [data[i][j] for j in range(len(data[i])) if j != pw_idx]

        sw = []
        num = 0
        while num < args.n_hw - 1:
            hw = hw_generator.sample(pw)
            if hw not in sw and hw != pw:
                sw.append(hw)
                num = num + 1
        sw.append(pw)
        random.shuffle(sw)
        assert len(sw) == args.n_hw, 'error'
        pw_index = sw.index(pw)
        sweetwords.append(sw)
        password_index.append(pw_index)
        aux_passwords.append(aux_pw)
        if (i+1) % 100 == 0:
            print('{} set honeywords generation!'.format((i+1)))

    with open(args.save_path + '/sweetwords.pickle', 'wb') as f:
        pickle.dump(sweetwords, f)
    with open(args.save_path + '/password_index.pickle', 'wb') as f:
        pickle.dump(password_index, f)
    with open(args.save_path + '/aux_passwords.pickle', 'wb') as f:
        pickle.dump(aux_passwords, f)
    print('==> finished.')


def pass2path_model():

    # set up paths
    cm_path = './char_map.pickle'

    # load CM map: a dictionary to map char or special symbol or number into digit
    char_map, char_map_inv = load_charmap(cm_path)
    char_size = len(char_map) + 1
    args.char_size = char_size

    with open('trans_dict_2path.json', 'r') as f:
        trans_dict_2path = json.load(f)
    args.vocab_size = len(trans_dict_2path) + 2

    with open('idx_look_up.json', 'r') as f:
        look_up = json.load(f)

    # load data
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    # load model
    dic = torch.load('./saved_model/pass2path_model/model', map_location='cuda:0')
    model = PASS2PATH(args.char_size, args.vocab_size, dic['embed_dim'], dic['lstm_dim'], dic['num_layer'], dic['dropout_rate'])
    model.load_state_dict(dic['model'])
    model.to(device)

    args.embed_dim = dic['embed_dim']
    args.lstm_dim = dic['lstm_dim']
    args.num_layer = dic['num_layer'] 

    # generator
    hw_generator = Generator_pass2path(model, char_map, char_map_inv, char2digits, path2word, trans_dict_2path, idx2path, look_up, 30, args)
    print('==> loaded model successfully!')

    sweetwords = []
    password_index = []
    aux_passwords = []

    for i in range(len(data)):

        pw_idx = random.sample(list(range(len(data[i]))), 1)[0]
        pw = data[i][pw_idx]
        aux_pw = [data[i][j] for j in range(len(data[i])) if j != pw_idx]
        sw = []
        num = 0
        while num < args.n_hw - 1:
            hw = hw_generator.sample(pw)
            if hw not in sw and hw != pw:
                sw.append(hw)
                num = num + 1
        sw.append(pw)
        random.shuffle(sw)
        assert len(sw) == args.n_hw, 'error'
        pw_index = sw.index(pw)
        sweetwords.append(sw)
        password_index.append(pw_index)
        aux_passwords.append(aux_pw)
        if (i+1) % 100 == 0:
            print('{} set honeywords generation!'.format((i+1)))

    with open(args.save_path + '/sweetwords.pickle', 'wb') as f:
        pickle.dump(sweetwords, f)
    with open(args.save_path + '/password_index.pickle', 'wb') as f:
        pickle.dump(password_index, f)
    with open(args.save_path + '/aux_passwords.pickle', 'wb') as f:
        pickle.dump(aux_passwords, f)
    print('==> finished.')


if __name__ == '__main__':
    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_hw', type=int, default=1)
    # 'list*', 'pcfg*', 'rnn*', 'markov*' are data-dependent methods
    parser.add_argument('--hw_gen_method', default='rnn', type=str, choices=['CBT3', 'CBT4', 'CBT*', 'CHM', 'list', 'list*', 'pcfg', 'pcfg*', 'markov', 'markov*', 'rnn', 'rnn*', 'pass2path', 'tweak'])
    parser.add_argument('--data_path', type=str, default='/BreachCompilationAnalysis/preprocessed_data/test_data.pickle')
    args = parser.parse_args()

    if not os.path.exists('./hw'):
        os.makedirs('./hw')

    args.save_path = './hw/{}(hw:{})'.format(args.hw_gen_method, args.n_hw)

    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    args.pw_dependent = False

    if args.hw_gen_method == 'rnn':
        rnn_model()
    elif args.hw_gen_method == 'rnn*':
        args.pw_dependent = True
        rnn_model()
    elif args.hw_gen_method == 'CBT3':
        chaffing_by_tweaking(3)
    elif args.hw_gen_method == 'CBT4':
        chaffing_by_tweaking(4)
    elif args.hw_gen_method == 'CBT*':
        chaffing_by_tweaking_random()
    elif args.hw_gen_method == 'CHM':
        chaffing_by_hybrid()
    elif args.hw_gen_method == 'list':
        list_model()
    elif args.hw_gen_method == 'list*':
        args.pw_dependent = True
        list_model()
    elif args.hw_gen_method == 'pcfg':
        # pcfg_model()
        print('please refer to https://github.com/lakiw/pcfg_cracker')
    elif args.hw_gen_method == 'pcfg*':
        args.pw_dependent = True
        # pcfg_model()
        print('please refer to https://github.com/lakiw/pcfg_cracker')
    elif args.hw_gen_method == 'markov':
        markov_model()
    elif args.hw_gen_method == 'markov*':
        args.pw_dependent = True
        markov_model()
    elif args.hw_gen_method == 'tweak':
        tweak_model()
    elif args.hw_gen_method == 'pass2path':
        pass2path_model()