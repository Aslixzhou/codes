from scipy.stats import spearmanr,pearsonr
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''
低方差过滤：
    from sklearn.feature_selection import VarianceThreshold
    data = pd.read_csv("factor_returns.csv")
    transfer = VarianceThreshold(threshold=1)
    data = transfer.fit_transform(data.iloc[:, 1:10])
    print("删除低⽅差特征的结果：\n", data)
    print("形状：\n", data.shape)
'''

'''特征选择后做特征降维处理'''

if __name__ == '__main__':
    x1 = [12.5, 15.3, 23.2, 26.4, 33.5, 34.4, 39.4, 45.2, 55.4, 60.9]
    x2 = [21.2, 23.9, 32.9, 34.1, 42.5, 43.2, 49.0, 52.8, 59.4, 63.5]
    print(spearmanr(x1, x2)) # 斯皮尔曼相关系数
    print(pearsonr(x1,x2)) # 皮尔逊相关系数

    print("--------------主成分分析PCA----------------") # 无监督
    data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]
    # ⼩数——保留多少信息
    transfer = PCA(n_components=0.9)
    data1 = transfer.fit_transform(data)
    print("保留90%的信息，降维结果为：\n", data1)
    # 整数——指定降维到的维数
    transfer2 = PCA(n_components=3)
    data2 = transfer2.fit_transform(data)
    print("降维到3维（列）的结果：\n", data2)

    print("pd.DataFrame(data):\n", pd.DataFrame(data))
    print("pd.DataFrame(data2):\n", pd.DataFrame(data2))

    print("--------------线性判别分析LDA----------------") # 有监督
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, random_state=66)
    lda = LinearDiscriminantAnalysis(n_components=2) # 只能指定整数
    lda_train = lda.fit_transform(x_train,y_train)
    print(lda_train)


    print("--------------PCA之前的数据规范化--------------")
    '''
    Z-score 标准化：将数据的每个特征值减去其均值，然后除以其标准差，使得每个特征的均值为0，标准差为1。
    Min-Max 标准化：将数据的每个特征值线性缩放到一个特定的范围，通常是[0, 1]或[-1, 1]。
    范数标准化：将每个样本的特征向量除以其范数，使得每个样本的特征向量具有单位范数。
    Robust 标准化：使用中位数和四分位距来规范化数据，而不是均值和标准差，以提高对异常值的鲁棒性。
    '''
    # 数据规范化
    data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    print(data,"\n",data_normalized)

    # 小数——保留多少信息
    transfer = PCA(n_components=0.9)
    data1 = transfer.fit_transform(data_normalized)
    print("保留90%的信息，降维结果为：\n", data1)

    # 整数——指定降维到的维数
    transfer2 = PCA(n_components=3)
    data2 = transfer2.fit_transform(data_normalized)
    print("降维到3维（列）的结果：\n", data2)


    print("--------------利用累计方差贡献率确定主成分分析的特征数--------------")
    # 鸢尾花数据集4个特征
    data = datasets.load_iris().data
    # 数据规范化
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    # 主成分分析
    pca = PCA()
    pca.fit(data_normalized)
    print(data_normalized)
    # 绘制累计方差贡献率曲线
    # 每个主成分解释的方差百分比
    print("每个主成分解释的方差百分比：", pca.explained_variance_ratio_)
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    print("前n个主成分解释的方差之和占总方差的比例：", cumulative_variance_ratio)
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Ratio')
    plt.title('Cumulative Variance Ratio vs. Number of Components')
    plt.grid(True)
    plt.show()