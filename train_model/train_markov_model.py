
import pickle

with open('/BreachCompilationAnalysis/preprocessed_data/train_data.pickle', 'rb') as f:
    data = pickle.load(f)

model = {}
model['chain'] = {}
model['starts'] = []

for j in data:
    for i in j:
        # while len(i) < 4:
        #     i += '\n'
        # i += '\n'
        if len(i) < 4:
            model['starts'].append(list(i))
        else:
            i += '\n'
            model['starts'].append(list(i[:4]))

            for j in range(4, len(i)):
                if tuple(i[j-4:j]) not in model['chain'].keys():
                    model['chain'][tuple(i[j-4:j])] = [i[j]]
                else:
                    model['chain'][tuple(i[j-4:j])].append(i[j])

with open('./saved_model/markov_model.pickle', 'wb') as f:
    pickle.dump(model, f)