# Code for the paper "The Impact of Exposed Passwords on Honeyword Efficacy'' accepted by the proceedings of the 33rd USENIX Security Symposium, August, 2024

### Environment
```
python==3.9.15
1password==0.6.2
ckl-psm==1.2
fasttext==0.9.2
lastpass-python==0.3.2
numpy==1.23.5
pandas==1.5.0
scipy==1.8.0
torch==1.11.0
zxcvbn-python==4.4.24
```

### Data download and preprocess

The first step is to download the 4iQ data. Please see "Get the data" in [the git repo](https://github.com/philipperemy/tensorflow-1.4-billion-password-analysis).

After downloading the data (e.g., by following the "Get the data" step), you will get a dir named "BreachCompilation" where there are data files included. Then you can run by:

```
python3 preprocess/read.py
python3 preprocess/process.py
```

The above running will filter out those passwords that do not meet some specific requirements, e.g., too long or too short, and group passwords based on the username of the emails such that each group represents a user.