import numpy as np
import keras
import sys
import os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from time import time

MAX_NGRAM = 2
NUM_WORDS = 5000
INDEX_FROM = 2
FEATURES = 1000
METHODS = ['skb']#['skb', 'svc']


def load_imdb_data(train, test, train_size = 'full', start_index = 0):
    train_x_full, train_y_full = train
    test_x, test_y = test

    if train_size == 'full':
        train_x = train_x_full[start_index:] # add a loop to go through these 5x
        train_y = train_y_full[start_index:]
    else:
        end_index = start_index+train_size
        train_x = train_x_full[start_index:end_index] # add a loop to go through these 5x
        train_y = train_y_full[start_index:end_index]

    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k: (v + INDEX_FROM) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2

    print("Producing bit representation...")

    # Produce N-grams
    id_to_word = {value: key for key, value in word_to_id.items()}

    vocabulary = {}
    for i in range(train_y.shape[0]):
        terms = []
        for word_id in train_x[i]:
            terms.append(id_to_word[word_id])

        for N in range(1, MAX_NGRAM + 1):
            grams = [terms[j:j + N] for j in range(len(terms) - N + 1)]
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
    X_train = np.zeros((train_y.shape[0], len(phrase_bit_nr)), dtype=np.uint8)
    print(sys.getsizeof(X_train)) #uint32: 7178300112, uint8: 1794575112
    Y_train = np.zeros(train_y.shape[0], dtype=np.uint8)

    X_test = np.zeros((test_y.shape[0], len(phrase_bit_nr)), dtype=np.uint8)
    print(sys.getsizeof(X_test))
    Y_test = np.zeros(test_y.shape[0], dtype=np.uint8)

    for i in range(train_y.shape[0]):
        terms = []
        for word_id in train_x[i]:
            terms.append(id_to_word[word_id])

        for N in range(1, MAX_NGRAM + 1):
            grams = [terms[j:j + N] for j in range(len(terms) - N + 1)]
            for gram in grams:
                phrase = " ".join(gram)
                if phrase in phrase_bit_nr:
                    X_train[i, phrase_bit_nr[phrase]] = 1

        Y_train[i] = train_y[i]

    for i in range(test_y.shape[0]):
        terms = []
        for word_id in test_x[i]:
            terms.append(id_to_word[word_id])

        for N in range(1, MAX_NGRAM + 1):
            grams = [terms[j:j + N] for j in range(len(terms) - N + 1)]
            for gram in grams:
                phrase = " ".join(gram)
                if phrase in phrase_bit_nr:
                    X_test[i, phrase_bit_nr[phrase]] = 1

        Y_test[i] = test_y[i]

    return X_train, Y_train, X_test, Y_test

def feat_selection(X_train, Y_train, X_test, nfeat=FEATURES, method='skb'):
    if method == 'skb':
        print("Selecting SKB features...")

        SKB = SelectKBest(chi2, k=nfeat)
        SKB.fit(X_train, Y_train)
        selected_features = SKB.get_support(indices=True)
        X_train = SKB.transform(X_train)
        print(X_train.shape)
        X_test = SKB.transform(X_test)
        print(X_test.shape)
    elif method == 'svc':
        print("Selecting SVC features...")
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, Y_train)
        model = SelectFromModel(lsvc, prefit=True, threshold=-np.inf, max_features=nfeat)
        X_train = model.transform(X_train)
        print(X_train.shape)
        X_test = model.transform(X_test)
        print(X_test.shape)

    return X_train, X_test

# for each method and training size, save output
def save_feat_data(train, test, train_size = 'full', start_index = 0, basepath = 'imdb_processed_data/',
                   methods = ['skb', 'svc'], nfeat=FEATURES):
    print(train_size)
    if not os.path.isdir(basepath):
        os.mkdir(basepath)
    print('format imdb data')
    X_train, Y_train, X_test, Y_test = load_imdb_data(train, test, train_size=train_size, start_index=start_index)
    for method in methods:
        X_train, X_test = feat_selection(X_train, Y_train, X_test, nfeat=nfeat, method=method)
        filename = method+'_trainsize_'+str(train_size)+'_'+str(start_index)+'_NFEAT_'+str(nfeat)
        np.savez(basepath+filename+'.npz', X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)

if __name__ =='__main__':
    print("Downloading dataset...")

    np.load.__defaults__ = (None, True, True, 'ASCII')
    train, test = imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)
    np.load.__defaults__ = (None, False, True, 'ASCII')



    train_size = 5000
    #ncv = int(train[0].shape[0]/train_size)
    ncv=1
    for cv in range(ncv):
        print('cv: '+str(cv))
        start_index = cv*train_size
        save_feat_data(train, test, train_size=train_size, start_index=start_index,
                       basepath='imdb_processed_data/',
                       methods=METHODS, nfeat=FEATURES)