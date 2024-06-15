from sklearn import svm
from sklearn.model_selection import train_test_split

'''
C: 正则化参数，浮点数，默认为1.0。C越大，表示对误分类样本的惩罚越重，可以降低训练误差但可能增加测试误差。
kernel: 核函数类型，字符串，默认为"rbf"（径向基核函数）。其他常用的核函数还包括："linear"（线性核函数）和"poly"（多项式核函数）等。
degree: 多项式核函数的次数，整数，默认为3。只有当kernel为"poly"时才起作用。
gamma: 核函数的系数，可以是浮点数或者字符串，默认为"scale"。如果是浮点数，它定义了核函数的系数，如果是"scale"，则会根据特征数量自动计算。另外还有"auto"选项，表示使用1 / n_features。
coef0: 核函数的独立系数，在"poly"和"sigmoid"核函数中起作用。默认为0.0。
shrinking: 是否使用启发式方法来加速训练，默认为True。在训练数据较大时，设置为True可以加快训练速度。
probability: 是否启用概率估计，默认为False。如果设置为True，则会启用概率估计，并在训练完成后可以调用predict_proba方法来获取每个类别的概率值。
tol: 求解器收敛的容差，浮点数，默认为1e-3。当模型参数的变化小于容差时，认为模型已经收敛。
cache_size: 内核缓存大小，单位为MB，默认为200。指定内核缓存的大小可以加快运行速度。
class_weight: 类别权重，字典或者"balanced"，默认为None。给每个类别分配不同的权重，可以用于处理类别不平衡的情况。
verbose: 是否输出详细的日志信息，默认为False。如果设置为True，会输出训练过程中的详细信息。
max_iter: 最大迭代次数，整数，默认为-1，表示没有限制。如果solver是"lbfgs"，"newton-cg"，"sag"或"saga"，那么这个参数表示的是最大迭代次数；如果solver是"liblinear"，那么这个参数表示的是训练迭代停止的条件。
decision_function_shape: 决策函数的类型，字符串，默认为"ovr"（one-vs-rest）。可选值还包括"ovo"（one-vs-one），在多分类问题中决定使用哪种方式进行分类。
break_ties: 是否在预测时打破平局，默认为False。当为True时，decision_function_shape必须为"ovr"。
'''

if __name__ == '__main__':
    X = [[0, 0], [1, 1],[2,2],[3,4],[1,3],[1,2]]
    y = [0, 1,0,2,3,2]
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)
    print(clf.support_vectors_)
    print(clf.support_)
    print("-------------------")
    print(clf.coef_)
    print("-------------------")
    print(clf.intercept_)
    print("-------------------")
    print(clf.predict([[2., 2.],[0,2]]))

    '''支持向量机的鸢尾花分类问题'''
    from sklearn.datasets import load_iris
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)
    print(x_train.shape, y_train.shape)
    print(x_train,y_train)
    svm = svm.SVC(kernel='rbf', C=1, gamma='auto')
    svm.fit(x_train, y_train)
    print("train score:", svm.score(x_train,y_train))
    print("test score:", svm.score(x_test,y_test))
    print("predict: ", svm.predict(x_test))

    '''支持向量机的回归问题'''
    from sklearn.svm import SVR
    import numpy as np
    import matplotlib.pyplot as plt
    # 创建示例数据集
    np.random.seed(42)
    X = 5 * np.random.rand(100, 1)
    y = np.sin(X).ravel() + 0.1 * np.random.randn(100)
    # 创建 SVR 模型
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    # 在训练集上训练模型
    svr.fit(X, y)
    # 生成预测结果
    X_test = np.linspace(0, 5, 100).reshape(-1, 1)
    y_pred = svr.predict(X_test)
    print(svr.score(X_test,y_pred))
    # 可视化结果
    plt.scatter(X, y, color='darkorange', label='data')
    plt.plot(X_test, y_pred, color='navy', lw=2, label='SVR')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
