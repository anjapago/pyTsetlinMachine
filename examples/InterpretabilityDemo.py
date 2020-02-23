from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np

number_of_features = 16
noise = 0.1
#num_exs = 5000
num_exs_train = 5000
num_exs_test = 5000

X_train = np.random.randint(0, 2, size=(num_exs_train, number_of_features), dtype=np.uint32)
Y_train = np.logical_xor(X_train[:,0], X_train[:,1]).astype(dtype=np.uint32)
Y_train = np.where(np.random.rand(num_exs_train) <= noise, 1-Y_train, Y_train) # Adds noise

X_test = np.random.randint(0, 2, size=(num_exs_test, number_of_features), dtype=np.uint32)
Y_test = np.logical_xor(X_test[:,0], X_test[:,1]).astype(dtype=np.uint32)

n_clauses = 10
tm = MultiClassTsetlinMachine(n_clauses, 15, 3.0, dlri = False, indexed=False,
                              number_of_state_bits=16, boost_true_positive_feedback=0)

print("fit:")
tm.fit(X_train, Y_train, epochs=1)

# print("\nClass 0 Positive Clauses:\n")
# class_num = 0
# #for j in range(0, n_clauses, 2):
# for j in range(0, 4, 1):
#     print("Clause #%d: " % (j))
#     l = []
#     for k in range(number_of_features*2):
#         print("Get state for TA="+str(k)+"*************************************")
#         state = tm.ta_state(class_num, j, k)
#         print("state: "+str(state))
#         action = tm.ta_action(class_num, j, k, 0)
#         print("action: "+str(action))
#     #     if action == 1:
#     #         if k < number_of_features:
#     #             l.append(" x%d" % (k))
#     #         else:
#     #             l.append("¬x%d" % (k-number_of_features))
#     # print(" ∧ ".join(l))

print("num clause chunks: "+str(tm.number_of_clause_chunks))
print("num clauses: "+str(tm.number_of_clauses))
print("num ta chunks: "+str(tm.number_of_ta_chunks))
print("num ta state bits: "+str(tm.number_of_state_bits))
#print("ta state: **************************")
#ta_states = tm.get_state()
#print(ta_states)
#print(len(ta_states[0]))

print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())

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