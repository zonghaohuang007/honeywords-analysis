#!/usr/bin/env python3
'''
Strategy for generate_from_empty.py:
- loop through the input real passwords
- seperate them into 3 categories: char, digit, special
- with some randomness, lower/upper case the character
- with some randomness, substitute the digit with random digit
- shuffle each block
- shuffle block position
- if sweetword exists in the generated set or equal to real password,
  run the same procedure again
'''

import sys
import random
import string
import numpy as np

from fasttext.FastText import _FastText

ALPHABET = string.ascii_uppercase + string.ascii_lowercase

special_char = '~!#$%^&*()_+-=`<>?,./{}|[]\;\':"'

def tweak_t_position(realpasswd, t):

	if t > len(realpasswd):
		t = len(realpasswd)

	new_pwd = ''
	for i in range(len(realpasswd)):
		char = realpasswd[i]
		if i > len(realpasswd) - t - 1:
			if char in ALPHABET:
				new_pwd += random.choice(ALPHABET)
			elif char in string.digits:
				new_pwd += random.choice(string.digits)
			else:
				new_pwd += random.choice(special_char)
		else:
			new_pwd += char

	return new_pwd


def tweak_digit_t_position(realpasswd, t):

	new_pwd = ''
	digit_position = []
	for i in range(len(realpasswd)):
		char = realpasswd[i]
		if char in string.digits or char in special_char:
			digit_position.append(i)
	digit_position = set(digit_position)
	k = len(realpasswd) - 1
	while len(digit_position) < t:
		digit_position.add(k)
		k = k - 1

	digit_position = list(digit_position)
	for i in range(len(realpasswd)):
		char = realpasswd[i]
		if i in digit_position[-t:]:
			if char in ALPHABET:
				new_pwd += random.choice(ALPHABET)
			elif char in string.digits:
				new_pwd += random.choice(string.digits)
			else:
				new_pwd += random.choice(special_char)
		else:
			new_pwd += char

	return new_pwd


# generate honeywords by tweaking
def chafffing_by_tweak(x):

    symbols = ['!', '#', '$', '%', '&', '"', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?',
               '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', "'"]
    f = 0.03
    p = 0.3
    q = 0.05
    temp = ''
    for i in range(len(x)):
        if x[i] >= "a" and x[i] <= "z":
            if random.random() <= p:
                temp += x[i].upper()
            else:
                temp += x[i]
        elif x[i] >= "A" and x[i] <= "Z":
            if random.random() <= q:
                temp += x[i].lower()
            else:
                temp += x[i]
        elif x[i] >= "0" and x[i] <= "9":
            temp += str(int(random.random() * 10))
        elif x[i] in symbols:
            temp += symbols[int(random.random()*len(symbols))]
    return temp


# generate honeywords by FastText
def chaffing_by_fasttext(x, n_hw):
    model = fasttext_.load_model("model_trained_on_rockyou_500_epochs.bin")
    honeywords=[]
    temp = model.get_nearest_neighbors(x,k=n_hw)
    for element in temp:
        honeywords.append(element[1])
	
    return honeywords


import fasttext
import random


def chaffing_by_model(model,real_password,k,l):
    """
    This function is used in the chaffing with a hybrid model method.
    :param real_password:
    :param k:
    :param l:
    :return:
    """

    # # load honeywords generation model (FastText)
    # print("Loading FastText word embeddings model...")
    # model = fasttext.load_model("/usr/project/xtmp/zh127/Models/model_trained_on_rockyou_500_epochs.bin")
    # print("Word embeddings model loaded.")
    # print()

    # # read target dataset
    # print("Producing l honeywords with a model for each real password found in the target dataset...")
    # #list hosting the generated honeywords
    honeywords=[]
    honeywords.append(real_password)
    temp = model.get_nearest_neighbors(real_password,k=l-1)
    for element in temp:
        honeywords.append(element[1])

    #after having l honeywords (including the real password) return the list
    return honeywords


def chafffing_by_tweaking_only(real_password,k):
    symbols = ['!', '#', '$', '%', '&', '"', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?',
               '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', "'"]

    honeywords={}
    # honeywords[real_password]=1
    duplicates_limit=0
    replaced_symbols=[0 for i in range(len(real_password))]
    p = 0.3
    f = 0.03
    q = 0.05
    while len(honeywords)<k:
        if duplicates_limit > 1000000:
            temp = tweak_t_position(real_password, 3)
        else:
            temp=""
            for i in range(len(real_password)):
                if real_password[i]>='a' and real_password[i]<='z':
                    if random.random()<=f:
                        temp+=real_password[i].upper()
                    else:
                        temp+=real_password[i]
                elif real_password[i]>='A' and real_password[i]<='Z':
                    if random.random()<=p:
                        temp+=real_password[i].lower()
                    else:
                        temp+=real_password[i]
                elif real_password[i]>='0' and real_password[i]<='9':
                    if random.random()<=q:
                        #temp+=str(randint(0, 9))
                        temp+=str(int(random.random()*10))
                    else:
                        temp+=real_password[i]
                elif real_password[i] in symbols:
                    if replaced_symbols[i]=='n':
                        temp+=real_password[i]
                        continue
                    if replaced_symbols[i]==0:
                        if random.random()<=p:
                            #index_symbol =randint(0,31)
                            index_symbol =int(len(symbols)*random.random())
                            temp+=symbols[index_symbol]
                            for j in range(i+1,len(real_password)):
                                if real_password[j]==real_password[i]:
                                    replaced_symbols[j] = symbols[index_symbol]
                        else:
                            for j in range(i+1,len(real_password)):
                                if real_password[j]==real_password[i]:
                                    replaced_symbols[j] = 'n'
                            temp+=real_password[i]
                    else:
                        temp+=replaced_symbols[i]

        if temp in honeywords or temp == real_password:
            duplicates_limit+=1
            if duplicates_limit % 4 == 0:
                p += 0.1
                q += 0.1
                f += 0.1
            
        else:
            honeywords[temp]=1

    #for key,value in honeywords.items():
    #    print(key)
    #exit(0)
    return honeywords


def chaffing_by_tweaking(honeywords,k,l):
    """
    This function is the 2nd stage (tweaking) in the hybrid method.
    :param honeywords: A list containing the produced by chaffing with a model HGT.
    :param k: Number of sweetwords.
    :param l: Number of sweetowrds to be produce by model (including the real password).
    :return: A list of produced sweetwords.
    """

    #how many chaffed with tweaking words per honeyword
    r = k/l

    new_honeywords=[]
    for i in honeywords:
        honeytemps = chafffing_by_tweaking_only(i, r)
        for z in honeytemps:
            new_honeywords.append(z)

    return new_honeywords


def chaffing_with_a_hybrid_model(model, real_password,k,l):
    """
    This function creates k honeyowrds in total. The l parameter denotes the total number of honeywords to be returned
    by the model (including the real password). The rest of them (k-l) are produced with the chaffing by tweaking method. In particular, it tweaks the
    honeywords produced by the chaffing by model method.
    :param real_password: The real-password for which to produce the honeywords.
    :param k: Total sweetwords per user.
    :param l: The honeywords to be returned my the model (including the real-password)
    :return:
    """

    if k%l!=0:
        print("Wrong arguments given!!! Specify other k and l values.")
        exit(0)

    #issue chaffing by model technique
    honeywords = chaffing_by_model(model,real_password,k,l)

    #issue chaffing by tweaking technique
    honeywords = chaffing_by_tweaking(honeywords,k,l)

    #honeywords list has the k sweetwords for the given password you can make whatever you want with them.

    return honeywords