'''
This code is modified from https://github.com/philipperemy/tensorflow-1.4-billion-password-analysis
'''

import os
import pickle
import numpy as np


def find_common(list1, list2):
    k = False
    for i in list1:
        if i in list2:
            k = True
            break        
    return k

def merge_node(nodes, a, b):
    new_nodes = []
    b_x = None
    for x in nodes:
        if a in x:
            a_x = x
        elif b in x:
            b_x = x
        else:
            new_nodes.append(x)
    if b_x is not None:
        new_nodes.append(a_x + b_x)
    else:
        new_nodes.append(a_x)
    return new_nodes


if __name__ == '__main__':

    data = {}
    num = 0
    for i in range(1, 12):
        with open('/BreachCompilationAnalysis/PasswordPairs/data_' + str(i) + '.pickle', 'rb') as f:
            data_dir = pickle.load(f)
        print('loaded {}'.format(i))
        for j in data_dir.keys():
            pass_set = []
            email_set = []
            num_pass = 0
            for k, v in data_dir[j].items():
                pass_set.append(v)
                email_set.append(k)
                num_pass = num_pass + len(v)

            nodes = [[a] for a in range(len(pass_set))]
            for a in range(len(pass_set)):
                for b in range(len(pass_set)):
                    if find_common(pass_set[a], pass_set[b]):
                        nodes = merge_node(nodes, a, b)

            for a in nodes:
                data_ = []
                email_ = []
                for b in a:
                    data_ = data_ + pass_set[b]
                    email_.append(email_set[b])
                if len(data_) > 1:
                    if len(data_) > 30:
                        print('username: {}'.format(j))
                        print('emails: {}'.format(email_))
                    else:
                        if len(nodes) == 1:
                            data[j] = data_
                        else:
                            data[email_[0]] = data_

                num = num + 1
                if num % 1e7 == 0:
                    print('load {} users'.format(num))
    
    with open('/BreachCompilationAnalysis/PasswordPairs/data.pickle', 'wb') as f:
        pickle.dump(data, f)

    print('The total number of users is {}'.format(num))

    