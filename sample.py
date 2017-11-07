# coding: utf-8

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from label_propagation import LGC, HMN, PARW, OMNI, CAMLP
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# 다른 비교 모델
# https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.asyn_lpa.asyn_lpa_communities.html#networkx.algorithms.community.asyn_lpa.asyn_lpa_communities

G = nx.karate_club_graph()
nx.draw(G, with_labels=True)
plt.show()
labels = {'Officer': 0, 'Mr. Hi': 1}
nodes = np.array([(n, labels[attr['club']]) for n, attr in G.nodes(data=True)])
x = nodes[:, 0]
y = nodes[:, 1]
A = nx.to_scipy_sparse_matrix(G, nodelist=x)     # adjacency matrix

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

methods = [('HMN', HMN(), {'graph': [A]}),
           ('LGC', LGC(), {'graph': [A], 'alpha':[
            0.01, 0.05, 0.1, 0.5, 0.99]}),
           # ('PARW', PARW(), {'graph':[A], 'lamb':[0.01, 0.05, 0.01, 0.5, 0.99]}),
           ('OMNI', OMNI(), {'graph': [A], 'lamb':[
            0.01, 0.1, 1.0, 10.0, 100.0]}),
           ('CAMLP', CAMLP(), {'graph': [A], 'beta':[
            0.01, 0.1, 1.0, 10.0, 100.0], 'H':[np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])]})]

for name, clf, params in methods:
    print("=========================")
    print(name)
    gs = GridSearchCV(clf, params, cv=5)
    gs.fit(x_train, y_train)

    model = gs.best_estimator_
    model.fit(x_train, y_train)

    predicted = model.predict(x_test)
    acc = (predicted == y_test).mean()
    print("\nAccuracy: %s\n" % acc)
