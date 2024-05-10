import pickle
import random


with open('/BreachCompilationAnalysis/preprocessed_data/data.pickle', 'rb') as f:
    data = pickle.load(f)

num_train = int(len(data) * 0.8)

idx_list = list(range(len(data)))
train_idx = random.sample(idx_list, num_train)

train_data = []
test_data = []
for i in range(len(data)):
    if i in train_idx:
        train_data.append(data[i])
    else:
        test_data.append(data[i])

with open('/BreachCompilationAnalysis/preprocessed_data/train_data.pickle', 'wb') as f:
    pickle.dump(train_data, f)

with open('/BreachCompilationAnalysis/preprocessed_data/test_data.pickle', 'wb') as f:
    pickle.dump(test_data, f)
