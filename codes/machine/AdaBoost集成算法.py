from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    # 加载数据集
    iris = load_iris()
    X, y = iris.data, iris.target
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 构建AdaBoost分类器
    ada_clf = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
    # 训练模型
    ada_clf.fit(X_train, y_train)
    # 预测并计算准确率
    y_pred = ada_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    # 输出AdaBoost模型中的所有弱分类器
    for estimator in ada_clf.estimators_:
        print(estimator)