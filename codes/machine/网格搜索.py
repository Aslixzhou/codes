from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # 获取鸢尾花数据集
    iris = load_iris()
    print("鸢尾花数据集的返回值：\n", iris)
    print("鸢尾花的特征值:\n", iris["data"])
    print("鸢尾花的⽬标值：\n", iris.target)
    print("鸢尾花特征的名字：\n", iris.feature_names)
    print("鸢尾花⽬标值的名字：\n", iris.target_names)
    print("鸢尾花的描述：\n", iris.DESCR)
    iris_d = pd.DataFrame(iris['data'], columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
    iris_d['Species'] = iris.target
    print(iris_d)
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=6)

    # 特征⼯程：标准化
    transfer = StandardScaler()
    print("x_train: ",x_train)
    print("x_test: ",x_test)
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    print("x_train: ",x_train)
    print("x_test: ",x_test)

    estimator = KNeighborsClassifier()
    param_dict = {"n_neighbors": [1, 3, 5, 9]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
    estimator.fit(x_train, y_train)
    print("在交叉验证中验证的最好结果：\n", estimator.best_score_)
    print("最好的参数模型：\n", estimator.best_estimator_)
    print("每次交叉验证后的准确率结果：\n", estimator.cv_results_)

    # 机器学习(模型训练)
    # estimator = KNeighborsClassifier(n_neighbors=9)
    # estimator.fit(x_train, y_train)

    # 模型评估
    # ⽐对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("预测结果为:\n", y_predict)
    print("⽐对真实值和预测值：\n", y_predict == y_test)

    # 直接计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)



