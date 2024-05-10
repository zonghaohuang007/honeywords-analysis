import numpy as np
import pickle
from math import sin, cos, sqrt
import random
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


def digits2char(digits, cm_):
    char = []
    for i in range(len(digits)):
        char.append(cm_[int(digits[i])])
    output = ''.join(char)[:len(digits)].replace('\n', '')
    return output


def char2digits(char, cm, pw_len, add_sos=False):
    if add_sos:
        data_index = np.zeros((len(char), pw_len+2))
        # data_index = [[0 for _ in range(pw_len+2)] for _ in range(len(char))]
        for i, p in list(enumerate(char)):
            pp = p + ''.join(['\n'] * (pw_len + 1 - len(p)))
            data_index[i][0] = len(cm)
            data_index[i][1:] = [cm[c] for c in pp]
    else:
        data_index = np.zeros((len(char), pw_len))
        # data_index = [[0 for _ in range(pw_len)] for _ in range(len(char))]
        for i, p in list(enumerate(char)):
            pp = p + ''.join(['\n'] * (pw_len - len(p)))
            data_index[i] = [cm[c] for c in pp]
    return data_index


def make_charmap(cm_path):
    char_vocabulary = {'\n': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '0': 10,
                       'a': 11, 'b': 12,
                       'c': 13, 'd': 14, 'e': 15, 'f': 16, 'g': 17, 'h': 18, 'i': 19, 'j': 20, 'k': 21, 'l': 22,
                       'm': 23, 'n': 24,
                       'o': 25, 'p': 26, 'q': 27, 'r': 28, 's': 29, 't': 30, 'u': 31, 'v': 32, 'w': 33, 'x': 34,
                       'y': 35, 'z': 36,
                       '!': 37, '"': 38, '#': 39, '$': 40, '%': 41, '&': 42, "'": 43, '(': 44,
                       ')': 45, '*': 46,
                       '+': 47, ',': 48, '-': 49, '.': 50, '/': 51, ':': 52, ';': 53, '<': 54, '=': 55, '>': 56,
                       '?': 57, '@': 58,
                       '`': 59, '{': 60, '|': 61, '}': 62, '~': 63, '[': 64, '\\': 65, ']': 66, '^': 67, '_': 68,
                       'A': 69, 'B': 70, 'C': 71, 'D': 72, 'E': 73, 'F': 74, 'G': 75, 'H': 76, 'I': 77, 'J': 78,
                       'K': 79, 'L': 80, 'M': 81, 'N': 82, 'O': 83, 'P': 84, 'Q': 85, 'R': 86, 'S': 87, 'T': 88,
                       'U': 89, 'V': 90, 'W': 91, 'X': 92, 'Y': 93, 'Z': 94
                       }
    with open(cm_path, 'wb') as f:
        pickle.dump(char_vocabulary, f)
    print('==> made char map and saved at {}'.format(cm_path))


def load_charmap(path):
    with open(path, 'rb') as f:
        cm = pickle.load(f)  # dictionary of mapping
    cm_inv = [x[0] for x in sorted(cm.items(), key=lambda x: x[1])]  # map the index to the character
    return cm, cm_inv

    
def trunc_decimal(val):
    if val > 1e10:
        return 'inf'
    return int(val * 1000) / 1000


def imagesc(img, mode='fg', title=None, experiment=None, step=None, label=None):
    x, y = img
    plt.clf()
    plt.plot(x, y, label=label, linewidth=2)
    # plt.plot(x, y, color='blue', linestyle='dashed', linewidth=3, label=label)
    if mode == 'fg':
        y1 = np.array(list(range(1,len(y)+1)))/len(y)
        plt.plot(x, y1, label='baseline', linewidth=2)
        plt.ylabel('success rate')
    elif mode == 'sng':
        y1 = np.array(list(range(1,len(y)+1)))/100
        plt.plot(x, y1, label='baseline', linewidth=2)
        plt.ylabel('# success')
    plt.legend()
    plt.xlabel('# guess')
    # plt.show()
    if title:
        plt.title(title)
    if experiment:
        experiment.log_figure(figure_name=title, step=step)

    return plt


def lr_schedule(lrnrate, epoch, warmupperiod=5, schedule=None, max_epoch=250):
    if schedule is None:
        schedule = [max_epoch // 2.667, max_epoch // 1.6, max_epoch // 1.142]
    warmupfactor = min(1, (epoch + 1) / (1e-6 + warmupperiod))
    if epoch < schedule[0]:
        return 1e00 * lrnrate * warmupfactor
    elif epoch < schedule[1]:
        return 1e-1 * lrnrate * warmupfactor
    elif epoch < schedule[2]:
        return 1e-2 * lrnrate * warmupfactor
    else:
        return 1e-3 * lrnrate * warmupfactor
