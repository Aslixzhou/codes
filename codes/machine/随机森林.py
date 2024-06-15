import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    data, target = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    rfc = RandomForestClassifier(n_estimators=10, max_features=3, max_samples=100, bootstrap=True)
    rfc.fit(x_train, y_train)
    print(rfc.feature_importances_) # 特征重要性
    for m in rfc.estimators_: # 遍历子决策树
        y_ = m.predict(x_test)
        print(y_)
    print(rfc.predict(x_test))
    print(rfc.score(x_train, y_train))
    print(rfc.score(x_test, y_test))