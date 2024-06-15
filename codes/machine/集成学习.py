import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    iris = load_iris()
    print(iris)
    data = iris['data']
    target = iris['target']
    target_names = iris['target_names']
    feature_names = iris['feature_names']
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    print("y_test: ", y_test)
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("y_pred: ", y_pred)
    print(knn.score(X_test, y_test))

    print("-----------集成学习Bagging-----------")
    bc = BaggingClassifier(knn,n_estimators=100)
    bc.fit(X_train,y_train)
    y_pred = bc.predict(X_test)
    print("y_pred: ", y_pred)
    print(bc.score(X_test, y_test))

    print("-----------集成学习AdaBoost-----------")
    '''
    estimator: 这是基本估计器的参数，用于指定弱分类器的类型。如果不指定，则使用默认的决策树分类器。
    n_estimators: 这是指基础分类器的数量，也就是说，在AdaBoost中要集成的弱分类器的数量。默认值为50，通常通过交叉验证来调整这个参数。
    learning_rate: 这是学习率的参数，它控制每个弱分类器的权重更新幅度。较小的学习率意味着每个分类器对最终模型的贡献会减小，可以防止过拟合。默认值为1.0。
    algorithm: 这是用于计算加权错误率的算法，有两种选择，"SAMME"和"SAMME.R"。默认值为"SAMME.R"，它表示使用了概率估计来更新加权错误率，而"SAMME"则是使用了分类器预测值。
    random_state: 这个参数用于控制随机数生成器，保证结果的可复现性。设置一个固定的值，可以让每次运行时得到相同的结果。
    '''
    base_classifier = DecisionTreeClassifier(max_depth=4)
    ada = AdaBoostClassifier(base_classifier,n_estimators=100) # 基于弱学习元的估计器
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("y_pred: ", y_pred)
    print(ada.score(X_test, y_test))

    print("-----------集成学习Stacking-----------")
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    estimators = [ # 基本分类器
        ('knn', knn),
        ('lr', lr),
        ('ada', ada),
        ('bc',bc)
    ]
    logistic = LogisticRegression() # 次级分类器
    clf = StackingClassifier(estimators=estimators,final_estimator=logistic)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("y_pred: ", y_pred)
    print(ada.score(X_test, y_test))

    print("-----------集成学习Voting-----------")
    vc = VotingClassifier([ # 基本分类器
        ('knn', knn),
        ('lr', lr),
        ('ada', ada),
        ('bc',bc)
    ])
    vc.fit(X_train, y_train)
    y_pred = vc.predict(X_test)
    print("y_pred: ", y_pred)
    print(ada.score(X_test, y_test))

    for c,label in zip([knn,lr,ada,bc],
                       ['knn','lr','ada','bc']):
        score = model_selection.cross_val_score(c,X_train,y_train,cv=3,scoring='accuracy')
        print("Accuracy: %0.2f [%s]" % (score.mean(),label))

