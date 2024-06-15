import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    print("-----------感知机的二分类问题------------")
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1, random_state=42)
    print("X: \n",X)
    print("y: \n",y)
    # 绘制样本数据
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k')

    # 初始化感知机分类器
    clf = Perceptron()
    # 训练分类器
    clf.fit(X, y)
    # 获取权重和偏差
    w = clf.coef_[0]
    b = clf.intercept_[0]
    print("w: \n",w)
    print("b: \n",b)

    # 绘制决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 # 计算第一个特征的最小值和最大值，并在两端各扩展1个单位，以确保决策边界在图中完全可见。
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 # 计算第二个特征的最小值和最大值，并在两端各扩展1个单位。
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1)) # 生成网格点，用于在特征空间中创建一个网格，以便绘制决策边界。
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # 对网格中的每个点进行预测，以确定其所属的类别。
    Z = Z.reshape(xx.shape)  # 将预测结果重新调整为与网格相同的形状，以便正确绘制。
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.Paired) # 使用contourf函数绘制填充决策边界的区域，其中alpha参数指定填充区域的透明度，cmap参数指定填充区域的颜色映射。
    # 绘制决策边界
    plt.plot([x_min, x_max], [-(w[0] * x_min + b) / w[1], -(w[0] * x_max + b) / w[1]], 'k--', lw=2) # w_1 x_1 + w_2 x_2 + b = 0  根据x_1 求x_2 绘制感知机决策边界

    # 设置图例
    plt.title('Perceptron Classification')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

    '''
    对于有多个类别的问题，可以为每个类别训练一个感知机模型。
    在训练阶段，对于每个类别，将该类别的样本标记为正例，而将其他所有类别的样本标记为负例。
    这样，就可以得到多个感知机模型，每个模型用于区分一个类别和其他所有类别。
    在预测阶段，将测试样本输入到所有的感知机模型中，并选择得分最高的类别作为最终的预测结果。
    '''
    # 加载垃圾邮件数据集，这里使用的是20个新闻组数据集中的垃圾邮件子集
    data = fetch_20newsgroups(subset='all',
                              categories=['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med'],
                              shuffle=True, random_state=42)

    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    # 使用TF-IDF向量化器将文本转换为特征向量
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    print("Number of features:", len(vectorizer.vocabulary_))
    print("X_train: \n",X_train)
    print("y_train: \n",y_train)
    print("X_test: \n",X_test)
    print("y_test: \n",y_test)
    # 初始化感知机模型
    perceptron = Perceptron()
    # 拟合模型
    perceptron.fit(X_train, y_train)
    # 进行预测
    y_pred = perceptron.predict(X_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    # 输出分类报告
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    print("perceptron.coef_.shape[0]: ", perceptron.coef_[0].shape[0])
    print("perceptron.coef_:\n",perceptron.coef_[0])
    print("perceptron_intercept_",perceptron.intercept_[0])
