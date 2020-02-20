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

if __name__ == '__main__':

    datafile = "imdb_processed_data/skb_trainsize_5000_0_NFEAT_1000.npz"
    with np.load(datafile) as data:
        print(sys.getsizeof(data))
        X_train_skb = data['X_train']
        Y_train = data['Y_train']
        X_test_skb = data['X_test']
        Y_test = data['Y_test']
        print(sys.getsizeof(X_train_skb))

    # num clauses=1000, T=80 (num tsetlin states??), s=27 (the s para in the chart??)
    tm = MultiClassTsetlinMachine(10000, 1, 27.0, dlri=False, indexed=False)
    # len ta_states: 5040000 if 10000 clauses, 80T --> 85% acc
    # len ta_states: 504000 if 1000 clauses, 80T --> accuracy goes to 79%
    # len ta_state: 5040000 if 10000 clauses, 40T and 1T --> 85% acc

    tm.fit(X_train_skb, Y_train, epochs=1, incremental=True)

    print("ta state: **************************")
    ta_states = tm.get_state()
    print(ta_states)
    print(len(ta_states[0]))
    print(ta_states[0][0])

    print("ta actions: **************************")
    ta_actions = tm.get_action()
    print(ta_actions)
    print(len(ta_actions[0]))
    print(ta_actions[0][0])

    predicted_class = tm.predict(X_test_skb)
    accuracy = 100*(predicted_class == Y_test).mean()
    print(accuracy)