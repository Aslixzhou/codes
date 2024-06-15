import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV

if __name__ == '__main__':
    iris = load_iris()
    print(iris)
    data = iris['data']
    target = iris['target']
    target_names = iris['target_names']
    feature_names = iris['feature_names']
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    print("y_test: ", y_test)
    param_dict = [{"n_neighbors": [1, 3, 7, 9]}] # 网格搜索k参数
    knn = KNeighborsClassifier(n_neighbors=10, algorithm="kd_tree") # algorithm="auto"
    knn = GridSearchCV(knn, param_grid=param_dict,cv=6,verbose=1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("y_pred: ", y_pred)
    print(knn.score(X_test, y_test))
    print("最好的参数模型：\n", knn.best_estimator_)
    print("每次交叉验证后的准确率结果：\n", knn.cv_results_)

    cv_results = knn.cv_results_
    for i in range(6):  # 假设cv=6
        print(f"第{i + 1}折的结果:")
        print("验证集得分:", cv_results[f"split{i}_test_score"])
        print("\n")