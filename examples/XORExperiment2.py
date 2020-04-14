from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np
import pandas as pd
import itertools

number_of_features = 20
#noise = 0.1
#num_train = 10000
num_train = 2000
num_test = 5000

X_train = np.random.randint(0, 2, size=(num_train, number_of_features), dtype=np.uint32)
# Y_train = np.logical_xor(X_train[:,0], X_train[:,1]).astype(dtype=np.uint32)
# Y_train = np.where(np.random.rand(num_train) <= noise, 1-Y_train, Y_train) # Adds noise

X_test = np.random.randint(0, 2, size=(num_test, number_of_features), dtype=np.uint32)
Y_test = np.logical_xor(X_test[:,0], X_test[:,1]).astype(dtype=np.uint32)

n_clauses = 10
nstate_bits = 8
s = 3.0
T = 10
#T = 2**nstate_bits

# tm = MultiClassTsetlinMachine(n_clauses, 15, 3.0,
#                               boost_true_positive_feedback=0,
#                               number_of_state_bits=nstate_bits)

noise_levels = [0.3, 0.15]
s_vals = [3, 10, 100]
nstate_bits_vals = [4, 8, 16]
n_clauses_vals = [4, 10, 100]


param_combos = itertools.product(noise_levels, s_vals, nstate_bits_vals, n_clauses_vals)

results_df = pd.DataFrame(columns=['rep', 'epoch', 'num_examples', 'accuracy',
                                   'noise_level', 's', 'num_clauses', 'num_states'])

for rep in range(0,100, 1):
    print("REP:********"+str(rep))
    param_combos = itertools.product(noise_levels, s_vals, nstate_bits_vals, n_clauses_vals)
    for (noise_lev, s, nstate_bits, n_clauses) in param_combos:
        T = n_clauses
        tm = MultiClassTsetlinMachine(n_clauses, T, s,
                                      boost_true_positive_feedback=0,
                                      number_of_state_bits=nstate_bits)
        Y_train = np.logical_xor(X_train[:,0], X_train[:,1]).astype(dtype=np.uint32)
        Y_train = np.where(np.random.rand(num_train) <= noise_lev, 1-Y_train, Y_train) # Adds noise

        print("fit:"+str(noise_lev))
        accuracies = []
        for i in range(50):
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

results_df.to_csv("noiselevels.csv", index = False)


# print("\nClass 0 Positive Clauses:\n")
# class_num = 0
# for j in range(0, n_clauses, 2):
#     #for j in range(0, 4, 1):
#     print("Clause #%d: " % (j))
#     l = []
#     for k in range(number_of_features*2):
#         action = tm.ta_action(class_num, j, k)#, 0, 0)
#         if action == 1:
#             if k < number_of_features:
#                 l.append(" x%d" % (k))
#             else:
#                 l.append("¬x%d" % (k-number_of_features))
#     print(" ∧ ".join(l))
#
# print("\nClass 0 Negative Clauses:\n")
# class_num = 0
# for j in range(1, n_clauses, 2):
#     #for j in range(0, 4, 1):
#     print("Clause #%d: " % (j))
#     l = []
#     for k in range(number_of_features*2):
#         action = tm.ta_action(class_num, j, k)#, 0, 0)
#         if action == 1:
#             if k < number_of_features:
#                 l.append(" x%d" % (k))
#             else:
#                 l.append("¬x%d" % (k-number_of_features))
#     print(" ∧ ".join(l))

# print("\nClass 1 Positive Clauses:\n")
# class_num = 1
# for j in range(0, n_clauses, 2):
#     #for j in range(0, 4, 1):
#     print("Clause #%d: " % (j))
#     l = []
#     for k in range(number_of_features*2):
#         action = tm.ta_action(class_num, j, k)#, 0, 0)
#         if action == 1:
#             if k < number_of_features:
#                 l.append(" x%d" % (k))
#             else:
#                 l.append("¬x%d" % (k-number_of_features))
#     print(" ∧ ".join(l))
#
# print("\nClass 1 Negative Clauses:\n")
# class_num = 1
# for j in range(1, n_clauses, 2):
#     #for j in range(0, 4, 1):
#     print("Clause #%d: " % (j))
#     l = []
#     for k in range(number_of_features*2):
#         action = tm.ta_action(class_num, j, k)#, 0, 0)
#         if action == 1:
#             if k < number_of_features:
#                 l.append(" x%d" % (k))
#             else:
#                 l.append("¬x%d" % (k-number_of_features))
#     print(" ∧ ".join(l))
#
# print("num clause chunks: "+str(tm.number_of_clause_chunks))
# print("num clauses: "+str(tm.number_of_clauses))
# print("num ta chunks: "+str(tm.number_of_ta_chunks))
# print("num ta state bits: "+str(tm.number_of_state_bits))
# #print("ta state: **************************")
# #ta_states = tm.get_state()
# #print(ta_states)
# #print(len(ta_states[0]))
#
# #predict point by points:
# print(tm.predict(X_test[0:10]))
# print(Y_test[0:10])
#
# print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())
