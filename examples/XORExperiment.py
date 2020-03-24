from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np

# number_of_features = 16
# noise = 0.1
# num_exs_train = 5000
# num_exs_test = 5000

number_of_features = 3
noise = 0
num_exs_train = 8000000
num_exs_test = 2000

X_train = np.random.randint(0, 2, size=(num_exs_train, number_of_features), dtype=np.uint32)
Y_train = np.logical_xor(X_train[:,0], X_train[:,1]).astype(dtype=np.uint32)
Y_train = np.where(np.random.rand(num_exs_train) <= noise, 1-Y_train, Y_train) # Adds noise

X_test = np.random.randint(0, 2, size=(num_exs_test, number_of_features), dtype=np.uint32)
Y_test = np.logical_xor(X_test[:,0], X_test[:,1]).astype(dtype=np.uint32)

# n_clauses = 10
# nstate_bits = 16
# T= 15
# s=3.0

n_clauses = 10
nstate_bits = 14
T = 15
s = 300.0
tm = MultiClassTsetlinMachine(n_clauses, T, s, dlri = True, indexed=False,
                              number_of_state_bits=nstate_bits, boost_true_positive_feedback=0)

print("fit:")
tm.fit(X_train, Y_train, epochs=1)

print("\nClass 0 Positive Clauses:\n")
class_num = 0
for j in range(0, n_clauses, 2):
#for j in range(0, 4, 1):
    print("Clause #%d: " % (j))
    l = []
    for k in range(number_of_features*2):
        action = tm.ta_action(class_num, j, k, 0, 0)
        if action == 1:
            if k < number_of_features:
                l.append(" x%d" % (k))
            else:
                l.append("¬x%d" % (k-number_of_features))
    print(" ∧ ".join(l))

print("\nClass 0 Negative Clauses:\n")
class_num = 0
for j in range(1, n_clauses, 2):
    #for j in range(0, 4, 1):
    print("Clause #%d: " % (j))
    l = []
    for k in range(number_of_features*2):
        action = tm.ta_action(class_num, j, k, 0, 0)
        if action == 1:
            if k < number_of_features:
                l.append(" x%d" % (k))
            else:
                l.append("¬x%d" % (k-number_of_features))
    print(" ∧ ".join(l))

print("\nClass 1 Positive Clauses:\n")
class_num = 1
for j in range(0, n_clauses, 2):
    #for j in range(0, 4, 1):
    print("Clause #%d: " % (j))
    l = []
    for k in range(number_of_features*2):
        action = tm.ta_action(class_num, j, k, 0, 0)
        if action == 1:
            if k < number_of_features:
                l.append(" x%d" % (k))
            else:
                l.append("¬x%d" % (k-number_of_features))
    print(" ∧ ".join(l))

print("\nClass 1 Negative Clauses:\n")
class_num = 1
for j in range(1, n_clauses, 2):
    #for j in range(0, 4, 1):
    print("Clause #%d: " % (j))
    l = []
    for k in range(number_of_features*2):
        action = tm.ta_action(class_num, j, k, 0, 0)
        if action == 1:
            if k < number_of_features:
                l.append(" x%d" % (k))
            else:
                l.append("¬x%d" % (k-number_of_features))
    print(" ∧ ".join(l))

print("num clause chunks: "+str(tm.number_of_clause_chunks))
print("num clauses: "+str(tm.number_of_clauses))
print("num ta chunks: "+str(tm.number_of_ta_chunks))
print("num ta state bits: "+str(tm.number_of_state_bits))
#print("ta state: **************************")
#ta_states = tm.get_state()
#print(ta_states)
#print(len(ta_states[0]))

#predict point by points:
print(tm.predict(X_test[0:10], 1))
print(Y_test[0:10])

print("Accuracy:", 100*(tm.predict(X_test, 1) == Y_test).mean())
print("Accuracy:", 100*(tm.predict(X_test, 1) == Y_test).mean())
print("Accuracy:", 100*(tm.predict(X_test, 1) == Y_test).mean())
print("Accuracy:", 100*(tm.predict(X_test, 1) == Y_test).mean())
print("Accuracy:", 100*(tm.predict(X_test, 1) == Y_test).mean())
print("TA Accuracy:", 100*(tm.predict(X_test, 0) == Y_test).mean())

# compare actions from regular with actions from the dlri
mc_tm_class = 1
print_output=False
print("\nClass 1 Negative Clauses:\n")
for clause in range(1, n_clauses, 2):
    for ta in range(number_of_features*2):
        dlri_actions = []
        for i in range(0, 100):
            dlri_actions.append(tm.ta_action(mc_tm_class, clause, ta, print_output, True))

        # check that state prbability is similar to the p(choosing action1) from the dlir action
        ta_state = tm.ta_state(mc_tm_class, clause, ta)
        print("State: "+str(ta_state)+" state prob: "+str(ta_state/2**nstate_bits))
        print(" avg dlri action: "+str(sum(dlri_actions)/len(dlri_actions)))
        print(" non dlri action: "+str(tm.ta_action(mc_tm_class, clause, ta, print_output, False)))



# print("\nClass 1 Positive Clauses:\n")
# for clause in range(0, n_clauses, 2):
#     for ta in range(number_of_features*2):
#         tm.ta_action(mc_tm_class, clause, ta, print_output, True)
#         tm.ta_action(mc_tm_class, clause, ta, print_output, False)

# print("\nClass 0 Negative Clauses:\n")
# class_num=0
# for j in range(1, n_clauses, 2):
#     print("Clause #%d: " % (j), end=' ')
#     l = []
#     for k in range(number_of_features*2):
#         if tm.ta_action(class_num, j, k) == 1:
#             if k < number_of_features:
#                 l.append(" x%d" % (k))
#             else:
#                 l.append("¬x%d" % (k-number_of_features))
#     print(" ∧ ".join(l))
#
# print("\nClass 1 Positive Clauses:\n")
# class_num=1
# for j in range(0, n_clauses, 2):
#     print("Clause #%d: " % (j))#, end=' ')
#     l = []
#     for k in range(number_of_features*2):
#         print("state: "+str(tm.ta_state(class_num, j, k)))
#         if tm.ta_action(class_num, j, k) == 1:
#             if k < number_of_features:
#                 l.append(" x%d" % (k))
#             else:
#                 l.append("¬x%d" % (k-number_of_features))
#     print(" ∧ ".join(l))
#
# print("\nClass 1 Negative Clauses:\n")
# class_num=1
# for j in range(1, n_clauses, 2):
#     print("Clause #%d: " % (j), end=' ')
#     l = []
#     for k in range(number_of_features*2):
#         if tm.ta_action(class_num, j, k) == 1:
#             if k < number_of_features:
#                 l.append(" x%d" % (k))
#             else:
#                 l.append("¬x%d" % (k-number_of_features))
#     print(" ∧ ".join(l))