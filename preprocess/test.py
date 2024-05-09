import glob
import os
import pickle


if __name__ == '__main__':

    with open('/BreachCompilationAnalysis/SimilarPassword/data.pickle', 'rb') as f:
        data_dir = pickle.load(f)
    num_dir = {}
    for i in data_dir:
        num = len(i)
        if num not in num_dir.keys():
            num_dir[num] = 1
        else:
            num_dir[num] = num_dir[num] + 1
    print(num_dir)

