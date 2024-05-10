import numpy as np
import string
import json
import csv
import itertools
import time
from ast import literal_eval


def find_med_backtrace(str1, str2):
    '''
    This function calculates the Minimum Edit Distance between 2 words using
    Dynamic Programming, and asserts the optimal transition path using backtracing.
    Input parameters: original word, target word
    Output: minimum edit distance, path
    Example: ('password', 'Passw0rd') -> 2.0, [('s', 'P', 0), ('s', '0', 5)]
    '''
    # Definitions:
    n = len(str1)
    m = len(str2)
    D = np.full((n + 1, m + 1), np.inf)
    op_arr_str = ["d", "i", "c", "s"]
    trace = np.full((n + 1, m + 1), None)
    for i in range(1 , n + 1):
        trace[i ,0] = (i - 1 ,0)
    for j in range(1 , m + 1):
        trace[0 ,j] = (0 ,j - 1)
    # Initialization:
    for i in range(n + 1):
        D[i,0] = i
    for j in range(m + 1):
        D[0,j] = j
    # Fill the matrices:
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            delete = D[i - 1, j] + 1
            insert = D[i, j-1] + 1
            if (str1[i - 1] == str2[j - 1]):
                sub = np.inf
                copy = D[i - 1, j - 1]
            else:
                sub = D[i - 1, j - 1] + 1
                copy = np.inf
            op_arr = [delete, insert, copy, sub]
            D[i ,j] = np.min(op_arr)
            op = np.argmin(op_arr)
            if (op == 0):
                # delete, go down
                trace[i,j] = (i-1, j)
            elif (op == 1):
                # insert, go left
                trace[i,j] = (i, j-1)
            else:
                # copy or subsitute, go diag
                trace[i,j] = (i-1, j-1)
#     print(trace)
    # Find the path of transitions:
    i = n
    j = m
    cursor = trace[i ,j]
    path = []
    while (cursor is not None):
        # 3 possible directions:
#         print(cursor)
        if (cursor[0] == i - 1 and cursor[1] == j - 1):
            # diagonal - sub or copy
            if (str1[cursor[0]] != str2[cursor[1]]):
                # substitute
                path.append(("s", str2[cursor[1]], cursor[0]))
            i = i - 1
            j = j - 1
        elif (cursor[0] == i and cursor[1] == j - 1):
            # go left - insert
            path.append(("i", str2[cursor[1]], cursor[0]))
            j = j - 1
        else:
            # (cursor[0] == i - 1 and cursor[1] == j )
            # go down - delete
            path.append(("d", None, cursor[0]))
            i = i - 1
        cursor = trace[cursor[0], cursor[1]]
    return D[n ,m], list(reversed(path))


def path2word(word, path):
    '''
    This function decodes the word in which the given path transitions the input word into.
    Input parameters: original word, transition path
    Output: decoded word
    '''
    if not path:
        return word
    final_word = []
    word_len = len(word)
    path_len = len(path)
    i = 0
    j = 0
    while i < word_len or j < path_len:
        if (j < path_len and path[j][2] == i):
            if (path[j][0] == "s"):
                # substitute
                final_word.append(path[j][1])
                i += 1
                j += 1
            elif (path[j][0] == "d"):
                # delete
                i += 1
                j += 1
            else:
                # "i", insert
                final_word.append(path[j][1])
                j += 1
        else:
            final_word.append(word[i])
            i += 1
    x = ''.join(final_word)
    return x


def generate_transition_dict():
    '''
    Generate a dictionary of all possible paths in a JSON format
    Assumptions: words' max length is 30 chars and words are comprised of 98 available characters
    'd' - ('d', None, 0-30) -> 31 options
    's' - ('s', 0-95, 0-30) -> 98x31 = 3038 options
    'i' - ('i', 0-95, 0-30) -> 98x31 = 3038 options
    Size of table: 31 + 3038 + 3038 = 6107
    '''
    max_len = 31
    d_list = [('d', None, i) for i in range(max_len)]
    # print(d_list)
    asci = list(string.ascii_letters)
    punc = list(string.punctuation)
    dig = list(string.digits)
    # chars = asci + punc + dig + [" ", "\t", "\x03", "\x04"]
    chars = asci + punc + dig
    print(len(chars))
    s_list = [('s', c, i) for c in chars for i in range(max_len)]
    # print(s_list)
    i_list = [('i', c, i) for c in chars for i in range(max_len)]
    # print(i_list)

    transition_table = d_list + s_list + i_list
    # print(len(transition_table))
    transition_dict_2idx = {}
    transition_dict_2path = {}
    for i in range(len(transition_table)):
        transition_dict_2idx[str(transition_table[i])] = i + 1
    #     transition_dict_2path[i + 1] = str(transition_table[i])
    # with open('trans_dict_2idx.json', 'w') as outfile:  
    #     json.dump(transition_dict_2idx, outfile)
    # with open('trans_dict_2path.json', 'w') as outfile:  
    #     json.dump(transition_dict_2path, outfile)
    print("Transitions dictionary created as trans_dict_2idx.json & trans_dict_2path.json")
    '''
    Read: 
    if filename:
        with open(filename, 'r') as f:
            transition_dict = json.load(f)
    '''


def path2idx(path, dictionary):
    '''
    This functions converts human-readable transition path to a
    dictionary-indices path (for future use in RNNs).
    Input parameters: human-readable path, dictionary
    Output: dictionary-indices path
    [('i', '\x04', 0), ('s', '\x03', 1), ('i', '2', 2), ('i', '\x04', 4)] ->
    [6076, 3008, 5737, 6080]
    '''
    idx_path = []
    for p in path:
        idx_path.append(dictionary[str(p)])
    return idx_path


def idx2path(path, dictionary):
    '''
    This functions converts dictionary-indices transition path to a
    human-readable path (for future use in RNNs).
    Input parameters: human-readable path, dictionary
    Output: dictionary-indices path
    [6076, 3008, 5737, 6080] ->
    [('i', '\x04', 0), ('s', '\x03', 1), ('i', '2', 2), ('i', '\x04', 4)]
    '''
    str_path = []
    for i in path:
        str_path.append(literal_eval(dictionary[str(i)]))
    return str_path


def path2idx(path, dictionary):
    '''
    This functions converts human-readable transition path to a
    dictionary-indices path (for future use in RNNs).
    Input parameters: human-readable path, dictionary
    Output: dictionary-indices path
    [('i', '\x04', 0), ('s', '\x03', 1), ('i', '2', 2), ('i', '\x04', 4)] ->
    [6076, 3008, 5737, 6080]
    '''
    idx_path = []
    for p in path:
        idx_path.append(dictionary[str(p)])
    return idx_path


def idx2path(path, dictionary):
    '''
    This functions converts dictionary-indices transition path to a
    human-readable path (for future use in RNNs).
    Input parameters: human-readable path, dictionary
    Output: dictionary-indices path
    [6076, 3008, 5737, 6080] ->
    [('i', '\x04', 0), ('s', '\x03', 1), ('i', '2', 2), ('i', '\x04', 4)]
    '''
    str_path = []
    for i in path:
        str_path.append(literal_eval(dictionary[str(i)]))
    return str_path


def idx2path_no_json(path, dictionary):
    '''
    This functions converts dictionary-indices transition path to a
    human-readable path (for future use in RNNs).
    Input parameters: human-readable path, dictionary
    Output: dictionary-indices path
    [6076, 3008, 5737, 6080] ->
    [('i', '\x04', 0), ('s', '\x03', 1), ('i', '2', 2), ('i', '\x04', 4)]
    '''
    str_path = []
    path = literal_eval(path)
    for i in path:
        str_path.append(dictionary[str(i)])
    return str_path


def generate_look_up_table(trans_dict_2idx, trans_dict_2path):
    keys = []
    idx = []
    for k, v in trans_dict_2idx.items():
        idx.append(v)
    for k, v in trans_dict_2path.items():
        keys.append(literal_eval(trans_dict_2path[k]))
    for i in range(len(keys)):
        assert (idx[i]+30) % 31 == keys[i][2]
        if i <= 30:
            assert keys[i][0] == 'd'
        elif i > 30 and i <= 2944:
            assert keys[i][0] == 's'
        else:
            assert keys[i][0] == 'i'
    look_up = {}
    for v in idx:
        look_up[str(v)] = []
        for i in idx:
            if (i + 30) % 31 < (v + 30) % 31:
                look_up[str(v)].append(i)
            elif (i + 30) % 31 == (v + 30) % 31:
                if v <= 2944:
                    look_up[str(v)].append(i)
    
    with open('idx_look_up.json', 'w') as outfile:  
        json.dump(look_up, outfile)

    # index 1-31 ==> delete
    # 32-2945 ==> substitute
    # 2946 - 5859


# x, y = find_med_backtrace('zonghao11', 'Zon1gh1ao')
# print(x)
# print(y)
# z = path2word('zonghao11', y)
# print(z)
# generate_transition_dict()
# with open('trans_dict_2idx.json', 'r') as f:
#     tran_dict = json.load(f)
# # print(len(tran_dict))
# # idx_path = path2idx(y, tran_dict)
# # print(idx_path)

# with open('trans_dict_2path.json', 'r') as f:
#     tran_dict_2path = json.load(f)
# # str_path = idx2path(idx_path, tran_dict)
# # print(tran_dict)
# # print(path2word('zonghao11', str_path))

# generate_look_up_table(tran_dict, tran_dict_2path)

