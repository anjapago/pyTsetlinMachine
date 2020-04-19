#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from time import time
import pandas as pd

MAX_NGRAM = 2

NUM_WORDS=5000
INDEX_FROM=2 

FEATURES=5000

print("Downloading dataset...")

# Save np.load
np_load_old = np.load

# Modify the default parameters of np.load
#np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)

# Restore np.load for future normal usage
#np.load = np_load_old


num_train = 5000
num_test = 25000

train_x,train_y = train[0][0:num_train], train[1][0:num_train]
test_x,test_y = test[0][0:num_test], test[1][0:num_test]

print("train_x shape, test_x shape")
print(train_x.shape)
print(test_x.shape)

print("train_y shape, test_y shape")
print(train_y.shape)
print(test_y.shape)

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

print("Producing bit representation...")

# Produce N-grams

id_to_word = {value:key for key,value in word_to_id.items()}

vocabulary = {}
for i in range(train_y.shape[0]):
	terms = []
	for word_id in train_x[i]:
		terms.append(id_to_word[word_id])
	
	for N in range(1,MAX_NGRAM+1):
		grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
		for gram in grams:
			phrase = " ".join(gram)
			
			if phrase in vocabulary:
				vocabulary[phrase] += 1
			else:
				vocabulary[phrase] = 1

# Assign a bit position to each N-gram (minimum frequency 10) 

phrase_bit_nr = {}
bit_nr_phrase = {}
bit_nr = 0
for phrase in vocabulary.keys():
	if vocabulary[phrase] < 10:
		continue

	phrase_bit_nr[phrase] = bit_nr
	bit_nr_phrase[bit_nr] = phrase
	bit_nr += 1

# Create bit representation
X_train = np.zeros((train_y.shape[0], len(phrase_bit_nr)), dtype=np.uint32)
Y_train = np.zeros(train_y.shape[0], dtype=np.uint32)

print("X_train shape, Y_train shape")
print(X_train.shape)
print(Y_train.shape)
for i in range(train_y.shape[0]):
	terms = []
	for word_id in train_x[i]:
		terms.append(id_to_word[word_id])

	for N in range(1,MAX_NGRAM+1):
		grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
		for gram in grams:
			phrase = " ".join(gram)
			if phrase in phrase_bit_nr:
				X_train[i,phrase_bit_nr[phrase]] = 1

	Y_train[i] = train_y[i]

X_test = np.zeros((test_y.shape[0], len(phrase_bit_nr)), dtype=np.uint32)
Y_test = np.zeros(test_y.shape[0], dtype=np.uint32)
print("X_test shape, Y_test shape")
print(X_test.shape)
print(Y_test.shape)

for i in range(test_y.shape[0]):
	terms = []
	for word_id in test_x[i]:
		terms.append(id_to_word[word_id])

	for N in range(1,MAX_NGRAM+1):
		grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
		for gram in grams:
			phrase = " ".join(gram)
			if phrase in phrase_bit_nr:
				X_test[i,phrase_bit_nr[phrase]] = 1				

	Y_test[i] = test_y[i]

print("Selecting features...")

SKB = SelectKBest(chi2, k=FEATURES)
SKB.fit(X_train, Y_train)

selected_features = SKB.get_support(indices=True)
X_train = SKB.transform(X_train)
X_test = SKB.transform(X_test)

print("X_train shape, X_test shape")
print(X_train.shape)
print(X_test.shape)

# test s
s_params = [3, 30, 300]
s_results_df = pd.DataFrame(columns=['rep', 'epoch', 'num_examples', 'accuracy',
								   'noise_level', 's', 'num_clauses', 'num_states'])

noise_lev = 0
#s_param = 27.0
T_param = 80
n_clauses =  10000
state_bits = 8

exp_id = str(time()).split('.')[1]

for s_param in s_params:
	tm = MultiClassTsetlinMachine(n_clauses, T_param, s_param, number_of_state_bits=state_bits, indexed=False)
	print("s param: "+str(s_param))
	print("\nAccuracy over 50 epochs:\n")
	for i in range(50):
		start_training = time()
		tm.fit(X_train, Y_train, epochs=1, incremental=True)
		stop_training = time()

		start_testing = time()
		accuracy = 100*(tm.predict(X_test) == Y_test).mean()
		stop_testing = time()

		print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, accuracy, stop_training-start_training, stop_testing-start_testing))
		row = {'epoch': i,
			   'num_examples': i*num_train,
			   'accuracy': accuracy,
			   'noise_level': noise_lev,
			   's': s_param,
			   'num_clauses': n_clauses,
			   'num_states': 2**state_bits,
			   'time': start_training-stop_training}
		s_results_df = s_results_df.append(row, ignore_index=True)
		s_results_df.to_csv("imdb_sparam_5000examples_"+str(exp_id)+".csv", index = False)



