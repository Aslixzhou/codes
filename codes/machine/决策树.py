import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


if __name__ == '__main__':
    data, target = load_iris(return_X_y=True)
    # 决策树默认使用CART算法
    tree = DecisionTreeClassifier(min_samples_split=4, min_samples_leaf=4,criterion="gini") # criterion="entropy" 信息增益
    '''
    max_depth：决策树的最大深度，用于控制树的最大层级。
    min_samples_split：节点分裂所需的最小样本数。
    min_samples_leaf：叶节点所需的最小样本数。
    max_features：寻找最佳分割点时需要考虑的特征数量。
    max_leaf_nodes：最大叶节点数量。
    '''
    print(tree.fit(data, target).score(data, target))
    print(tree.predict([[1, 2, 3, 4]]))
    print(tree.feature_importances_) # 特征重要性