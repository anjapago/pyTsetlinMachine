# import sys
# sys.path.append('~/ClionProjects/pyTsetlinMachine')
# print(sys.path)

from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from time import time
import pandas as pd
import numpy as np
import os
import sys
import argparse
import tracemalloc
import pickle
import cProfile

def fit_tm(tm, X_train_skb, Y_train):
    tm.fit(X_train_skb, Y_train, epochs=1, incremental=True)

def run_tsetlin(basepath, datafile, n_epochs = 50):
    try:
        tracemalloc.start()
        with np.load(datafile) as data:
            print(sys.getsizeof(data))
            X_train_skb = data['X_train']
            Y_train = data['Y_train']
            X_test_skb = data['X_test']
            Y_test = data['Y_test']
            print(sys.getsizeof(X_train_skb))

        tm = MultiClassTsetlinMachine(10000, 80, 27.0)

        results_df = pd.DataFrame(index = range(n_epochs), columns = ['accuracy', 'TP',
                                                                      'FP', 'TN', 'FN',
                                                                      'test_time',
                                                                      'train_time'])

        snapshots = []

        print("\nAccuracy over epochs:\n"+str(n_epochs))
        for i in range(n_epochs):
            snapshots.append(tracemalloc.take_snapshot())

            start_training = time()
            #tm.fit(X_train_skb, Y_train, epochs=1, incremental=True)
            cProfile.runctx("fit_tm(tm, X_train_skb, Y_train)", globals(),
                            {'tm':tm, 'X_train_skb':X_train_skb, 'Y_train': Y_train},
                            filename = 'fit_profile_'+str(i))
            stop_training = time()

            start_testing = time()
            predicted_class = tm.predict(X_test_skb)
            result = 100*(predicted_class == Y_test).mean()
            stop_testing = time()
            test_labels = Y_test
            tp = sum([pred if pred == test_labels[idx] else 0 for idx, pred in enumerate(predicted_class)])
            fp = sum([pred if pred != test_labels[idx] else 0 for idx, pred in enumerate(predicted_class)])
            tn = sum([(1 - pred) if pred == test_labels[idx] else 0 for idx, pred in enumerate(predicted_class)])
            fn = sum([(1 - pred) if pred != test_labels[idx] else 0 for idx, pred in enumerate(predicted_class)])
            results_df.loc[i, 'accuracy'] = result
            results_df.loc[i, ['TP', 'FP', 'TN', 'FN']] = [tp, fp, tn, fn]
            results_df.loc[i, ['test_time', 'train_time']] = [stop_testing-start_testing, stop_training-start_training]
            if '/' in datafile:
                results_df.to_csv(basepath+datafile.split('/')[1].split('.')[0]+'_results_df.csv')
                np.save(basepath + datafile.split('/')[1].split('.')[0] + '_epoch_' + str(i) + '_predictions.npy', predicted_class)
            else:
                results_df.to_csv(basepath + datafile.split('.')[0] + '_results_df.csv')
                np.save(basepath+datafile.split('.')[0]+'_epoch_'+str(i)+'_predictions.npy',predicted_class)

            print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result,
                                                                           stop_training-start_training,
                                                                           stop_testing-start_testing))
            pickle.dump([snapshots], open( "snapshots.p", "wb"))
    except Exception as e:
        print("error with file:"+str(datafile))
        print(e)

if __name__ == '__main__':

    resultspath = 'tsetlin_results/'
    if not os.path.isdir(resultspath):
        os.mkdir(resultspath)

    parser = argparse.ArgumentParser()

    parser.add_argument("--file", "-f", type=str, required=True)
    args = parser.parse_args()

    datafile = args.file #'skb_trainsize_full_0_NFEAT_5000.npz'
    if 'npz' not in datafile:
        print('running on every file in :'+datafile)
        # run for every file in that folder
        for filen in os.listdir(datafile):
            exists = False
            if ('trainsize_5000' in filen) and ('flip_ids' not in filen):
                print('computing file: '+filen)
                existing_results = os.listdir(resultspath)

                run_tsetlin(resultspath, datafile+'/'+filen, n_epochs=10)
                #cProfile.run("run_tsetlin(resultspath, datafile+'/'+filen, n_epochs=10)", "profile")
                # for result_file in existing_results:
                #     if filen.split('.')[0] in result_file:
                #         exists = True
                #         print("datafile already computed: "+str(filen))
                # if not exists:
                #     try:
                #         run_tsetlin(resultspath, datafile+'/'+filen)
                #     except Exception as e:
                #         print("failed for file: "+str(filen))
                #         print(str(e))
                #         pass
    else:
        # run directly on that file
        run_tsetlin(resultspath, datafile)