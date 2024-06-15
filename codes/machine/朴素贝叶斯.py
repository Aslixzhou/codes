import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import load_iris

if __name__ == '__main__':
    print("-----------三种朴素贝叶斯实现形式-----------")
    data, target = load_iris(return_X_y=True)
    data2 = data[:, 2:].copy()
    print(data2,target)
    gs_nb = GaussianNB()
    gs_nb.fit(data2, target).score(data2, target)
    y_pred = gs_nb.predict([[1.2,2.3],[4.6,5.0]])
    print("y_pred: ",y_pred)
    mu_nb = MultinomialNB()
    mu_nb.fit(data2, target).score(data2, target)
    y_pred2 = mu_nb.predict([[1.2,2.3],[4.6,5.0]])
    print("y_pred2: ", y_pred2)
    be_nb = BernoulliNB()
    be_nb.fit(data2, target).score(data2, target)
    y_pred3 = be_nb.predict([[1.2,2.3],[4.6,5.0]])
    print("y_pred3: ", y_pred3)