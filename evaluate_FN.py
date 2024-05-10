
import numpy as np

import torch
import torch.nn as nn
from mymodels import *
from utils import *
import argparse
from math import log2


torch.set_num_threads(1)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def measure_reuse(SW, X, args):
    return_true = False
    if args.hardness == 'easy':
        for sw in SW:
            if sw in X:
                return_true = True
                break
    elif args.hardness == 'medium':
        for sw in SW:
            if sw in X:
                return_true = False
                break
            if sw not in X and len(sw) >=4:
                for i in range(4, len(sw)+1):
                    for b in X:
                        c = sw[i-4:i]
                        assert len(c) == 4
                        if c in b:
                            return_true = True
    elif args.hardness == 'hard':
        return_true = True
        for sw in SW:
            if sw not in X and len(sw) >=4:
                for i in range(4, len(sw)+1):
                    for b in X:
                        c = sw[i-4:i]
                        assert len(c) == 4
                        if c in b:
                            return_true = False
            elif sw in X:
                return_true = False
    elif args.hardness == 'averaged':
        return_true = True

    return return_true


if __name__ == '__main__':

    # get parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--hw_gen_method', type=str, default='rnn')
    parser.add_argument('--n_hw', type=int, default=20)
    parser.add_argument('--hardness', type=str, default='easy')
    args = parser.parse_args()

    args.name_file = args.data_path.split('/')[-1].split('.')[0]
    if args.pw_dependent:
        args.data_path = './hw/' + args.hw_model + '/' + args.name_file + '/pw_dependent'
    elif args.target:
        args.data_path = './hw/' + args.hw_model + '/' + args.name_file + '/target'
    else:
        args.data_path = './hw/' + args.hw_model + '/' + args.name_file

    # load CM map: a dictionary to map char or special symbol or number into digit
    char_map, char_map_inv = load_charmap('./char_map.pickle')
    char_size = len(char_map) + 1
    id_sos = len(char_map)
    args.char_size = char_size

    # set up comet ml experiment
    print('==> setting up comet experiment...')
    experiment = Experiment(project_name='Honeywords-evaluation', auto_param_logging=False,
                            api_key='U1kuka6SA58EBpCa8Ct6CY1fp',
                            auto_metric_logging=False,
                            parse_args=False)
    comet_ml.config.experiment = None
    experiment.set_name(exp_name)
    experiment.add_tag('')
    experiment.log_parameters(vars(args))

    # load data
    with open(args.data_path + '/sweetwords.pickle', 'rb') as f:
        sw = pickle.load(f)
    with open(args.data_path + '/password_index.pickle', 'rb') as f:
        pw_index = pickle.load(f)
    # with open(args.data_path + '/aux_passwords.pickle', 'rb') as f:
    #     aux_pw = pickle.load(f)
    with open('/usr/project/xtmp/zh127/hw_data/data/data_20220904/auxlist_20220904_10000.pickle', 'rb') as f:
        aux_pw = pickle.load(f)
    print('==> loaded data successfully.')

    # load model
    dic = torch.load('./saved_model/transformer_metric_model/model', map_location='cuda:0')
    model = Transformer_embedding(char_size, dic['out_dim'], dic['num_layer'], dic['d_model'], dic['nhead'], dic['embedding_dim'])
    model.load_state_dict(dic['model'])
    model.to(device)
    model.eval()
    print('==> loaded model successfully!')

    if args.eval_mode == 'fg':
        fg_success_track = np.zeros(shape=[len(sw[0]), ], dtype=np.float32)
        fg_guess_track = list(range(1, len(sw[0]) + 1))
        fg_guess_track = np.array(fg_guess_track)
        num = 0
        tic0 = time()
        min_similarity = 0
        for i in range(len(sw)):
            hw = sw[i]
            hw = [j[:30] for j in hw]
            for j in range(len(hw)):
                special_chars = []
                for z in hw[j]:
                    if z not in char_map.keys() and z not in special_chars:
                        special_chars.append(z)
                if len(special_chars) != 0:
                    for z in special_chars:
                        hw[j] = hw[j].replace(z,'')
            pw = hw[pw_index[i]]
            if measure_reuse(hw, aux_pw[i], args):
                success_track = np.zeros(shape=[len(sw[0]), ], dtype=np.float32)
                with torch.no_grad():
                    if args.aux_list:
                        aux_pw_embedding_list = []
                        for aux_p in aux_pw[i]:
                            aux_pw_torch = char2digits([aux_p], char_map, 30, add_sos=False)
                            aux_pw_torch = torch.Tensor(aux_pw_torch).cuda().to(torch.int64)
                            aux_pw_embedding = model(aux_pw_torch)
                            aux_pw_embedding_list.append(aux_pw_embedding)
                    else:
                        aux_pw_torch = char2digits([aux_pw[i][0]], char_map, 30, add_sos=False)
                        aux_pw_torch = torch.Tensor(aux_pw_torch).cuda().to(torch.int64)
                        aux_pw_embedding = model(aux_pw_torch)
                    pw_torch = char2digits([pw], char_map, 30, add_sos=False)
                    pw_torch = torch.Tensor(pw_torch).cuda().to(torch.int64)
                    pw_embedding = model(pw_torch)
                similarity = []
                for j in range(len(hw)):
                    if args.aux_list:
                        similar_scores = []
                        for k in range(len(aux_pw_embedding_list)):
                            with torch.no_grad():
                                hw_torch = char2digits([hw[j]], char_map, 30, add_sos=False)
                                hw_torch = torch.Tensor(hw_torch).cuda().to(torch.int64)
                                hw_torch_embedding = model(hw_torch)
                                similar_score = nn.CosineSimilarity()(aux_pw_embedding_list[k], hw_torch_embedding).cpu().numpy()
                            similar_scores.append(similar_score[0])
                        similarity.append(max(similar_scores))
                    else:
                        with torch.no_grad():
                            hw_torch = char2digits([hw[j]], char_map, 30, add_sos=False)
                            hw_torch = torch.Tensor(hw_torch).cuda().to(torch.int64)
                            hw_torch_embedding = model(hw_torch)
                            similar_score = nn.CosineSimilarity()(aux_pw_embedding, hw_torch_embedding).cpu().numpy()
                        similarity.append(similar_score[0])
                sort_index = np.argsort(similarity)[::-1]
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
                if num % 100 == 0:
                    print('==> finished {} / {} hw evaluations | cost time: {} sec/sw'.format(num, i + 1, (time() - tic0) / 100))
                    plt = imagesc([fg_guess_track, fg_success_track], mode='fg', title='flatness graph', experiment=experiment, step=num, label=args.hw_model)
                    plt.savefig(args.data_path + '/' + args.eval_mode + '_{}.png'.format(args.hardness))
                    tic0 = time()
        plt = imagesc([fg_guess_track, fg_success_track], mode='fg', title='flatness graph', experiment=experiment, step=len(sw), label=args.hw_model)
        plt.savefig(args.data_path + '/' + args.eval_mode + '_{}.png'.format(args.hardness))
        if args.aux_list:
            with open(args.data_path + '/' + args.eval_mode + '_list_{}.pickle'.format(args.hardness), 'wb') as f:
                pickle.dump([fg_guess_track, fg_success_track], f)
        else:
            with open(args.data_path + '/' + args.eval_mode + '_{}.pickle'.format(args.hardness), 'wb') as f:
                pickle.dump([fg_guess_track, fg_success_track], f)
        print('==> attacked {} users'.format(num / len(sw)))
    else:
        print('no such evaluation mode...')

    experiment.send_notification('finished')
    print('==> finished.')

