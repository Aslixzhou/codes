import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

if __name__ == '__main__':
    digits = load_digits()
    data = digits['data']
    target = digits['target']
    print(data,target)
    feature_names = digits['feature_names']
    print(feature_names)
    target_names = digits['target_names']
    print(target_names)
    images = digits['images']
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    print("y_test: " , y_test)
    lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=3000, n_jobs=-1) # 迭代次数 CPU核数 C正则化强度 越小越强 solver='lbfgs'优化问题算法 拟牛顿法 sag saga newton-cg
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    print("y_pred" , y_pred)
    print(lr.score(x_train, y_train))
    print(lr.score(x_test, y_test))

    print("-----------非线性逻辑回归-----------")
    poly_reg = PolynomialFeatures(degree=2)
    x_poly = poly_reg.fit_transform(x_train)
    logistic = LogisticRegression()
    logistic.fit(x_poly,y_train)
    print(logistic.score(x_poly,y_train))
    print(logistic.score(poly_reg.fit_transform(x_test),y_test))


