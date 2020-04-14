from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np
import pandas as pd
import itertools

import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import time
from time import time

num_train = 5000
num_test = 5000
MAX_NGRAM = 2
NUM_WORDS=5000
#INDEX_FROM=2
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

train_x,train_y = train[0][0:num_train], train[1][0:num_train]
test_x,test_y = test[0][0:num_test], test[1][0:num_train]

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

print("train size 1")
print(X_train.shape)

print(X_test.shape)
print("test size 1")

SKB = SelectKBest(chi2, k=FEATURES)
SKB.fit(X_train, Y_train)

selected_features = SKB.get_support(indices=True)
X_train = SKB.transform(X_train)
X_test = SKB.transform(X_test)

print("train size")
print(X_train.shape)

print(X_test.shape)
print("test size")

# Tsetlin Machine -----------------------------------------------------------------------------------
# n_clauses = 10
# nstate_bits = 8
# s = 3.0
noise_levels = [0.3, 0.15, 0]
s_vals = [3, 30, 300]
nstate_bits_vals = [4, 8]
n_clauses_vals = [100, 1000, 10000, 100000]

param_combos = itertools.product(noise_levels, s_vals, nstate_bits_vals, n_clauses_vals)

results_df = pd.DataFrame(columns=['rep', 'epoch', 'num_examples', 'accuracy',
                                   'noise_level', 's', 'num_clauses', 'num_states'])

exp_id = str(time.time())[0:5]

for rep in range(0,10, 1):
    print("REP:********"+str(rep))
    param_combos = itertools.product(noise_levels, s_vals, nstate_bits_vals, n_clauses_vals)
    for (noise_lev, s, nstate_bits, n_clauses) in param_combos:
        print("params: "+ str((noise_lev, s, nstate_bits, n_clauses)))
        T = n_clauses
        #10000, 80, 27.0
        #T = 80
        tm = MultiClassTsetlinMachine(n_clauses, T, s,
                                      boost_true_positive_feedback=0,
                                      number_of_state_bits=nstate_bits)
        Y_train = np.logical_xor(X_train[:,0], X_train[:,1]).astype(dtype=np.uint32)
        Y_train = np.where(np.random.rand(num_train) <= noise_lev, 1-Y_train, Y_train) # Adds noise

        print("fit:"+str(noise_lev))
        accuracies = []
        for i in range(5000):
            tm.fit(X_train, Y_train, epochs=1, incremental = True)
            accuracy = 100*(tm.predict(X_test) == Y_test).mean()
            print("Accuracy:", accuracy)
            accuracies.append(accuracy)
            row = {'rep': rep,
                   'epoch': i,
                   'num_examples': i*num_train,
                   'accuracy': accuracy,
                   'noise_level': noise_lev,
                   's': s,
                   'num_clauses': n_clauses,
                   'num_states': 2**nstate_bits}
            results_df = results_df.append(row, ignore_index=True)
    #results_df[noise_lev] = accuracies

    results_df.to_csv("imdb_noiselevels_"+str(exp_id)+".csv", index = False)

