## Code for the paper "[The Impact of Exposed Passwords on Honeyword Efficacy](https://arxiv.org/pdf/2309.10323)'' accepted by the proceedings of the 33rd USENIX Security Symposium, August, 2024

### Environment

Please let me know if there is any missing package list here.
```
python==3.9.15
1password==0.6.2
ckl-psm==1.2
editdistance==0.8.1
fasttext==0.9.2
lastpass-python==0.3.2
numpy==1.23.5
pandas==1.5.0
scipy==1.8.0
torch==1.11.0
zxcvbn-python==4.4.24
tqdm==4.66.1
```

### Data download and preprocess

The first step is to download the 4iQ data. Please see "Get the data" in [the git repo](https://github.com/philipperemy/tensorflow-1.4-billion-password-analysis).

After downloading the data (e.g., by following the "Get the data" step), you will get a dir named "BreachCompilation" where there are data files included. Then you can reprocess the data by running (please make sure that your machine to process the data has enough memory):

```
python3 preprocess/read.py
python3 preprocess/process.py
```

The above running will filter out those passwords that do not meet some specific requirements, e.g., too long or too short, and group passwords based on the usernames of the emails such that each group represents a user.

### Split data into a training set and a test set

We split the users into a training group (80%) and a test group (20%) by running:

```
python3 preprocess/data_split.py
```

### Model training

In our work, there are some honeyword-generation methods that require a ML model, e.g., RNN, Tweak, and Pass2Path. Also, to implement the FN attacker in user-chosen case, we need a similarity model that measures the password similarity; to implement the FN attacker in algorithmically generated case, we need a classifier to classify a given password into one of the password generators. To train these ML models, you can run:
```
# Markov for honeyword-generation
python3 train_model/train_markov_model.py
# RNN for honeyword-generation
python3 -m torch.distributed.launch --nproc_per_node=<the number of gpus in your machine> train_model/train_rnn_model.py
# Tweak for honeyword-generation
python3 -m torch.distributed.launch --nproc_per_node=<the number of gpus in your machine> train_model/train_tweak_model.py
# Pass2path for honeyword-generation
python3 -m torch.distributed.launch --nproc_per_node=<the number of gpus in your machine> train_model/train_pass2path_model.py
# similarity model for FN attacker in user-chosen case
python3 -m torch.distributed.launch --nproc_per_node=<the number of gpus in your machine> train_model/train_similarity_model.py
# classifier for FN attacker in algorthmically generated case
python3 -m torch.distributed.launch --nproc_per_node=<the number of gpus in your machine> train_model/train_classifier_model.py --data_path <the path of machine-generated data> --num_classes <number of password managers>
```
For PCFG, please use [PCFG](https://github.com/lakiw/pcfg_cracker).


### Honeyword-generation (user-chosen case)

To generate honeywords, please run:
```
python3 generate_honeywords.py --n_hw <number of sweetwords> --hw_gen_method <honeyword-generation algorithm>
```

### FN attack evaluation (user-chosen case)

To evaluate the generated honeywords, please run:
```
python3 evaluate_FN.py --n_hw <number of sweetwords> --hw_gen_method <honeyword-generation algorithm> --hardness <hardness:{easy, medium, hard, average}>
```

### FP attack evaluation (user-chosen case)

To evaluate the generated honeywords, please run:
```
python3 evaluate_FP.py --n_hw <number of sweetwords> --hw_gen_method <honeyword-generation algorithm>
```


If you have any question on our work and this repo, please feel free to email the author.


If you find this git repo is helpful for your research, please consider to cite:
```
@inproceedings{huang2024:honeywords,
  title={The impact of exposed passwords on honeyword efficacy},
  author={Huang, Z. and Bauer, L. and Reiter M. K.},
  booktitle={33\textsuperscript{rd}} USENIX Security Symposium,
  year={2024}
}
```
