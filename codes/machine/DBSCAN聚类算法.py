import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.metrics import calinski_harabasz_score

'''
DBSCAN（Density-Based Spatial Clustering of Applications with Noise）是一种基于密度的聚类算法，它可以发现任意形状的聚类，并且在处理噪声和异常值方面表现良好。DBSCAN算法的核心思想是将密度较高的点视为聚类的一部分，并通过设置距离阈值和最小点数来确定聚类的形成。
下面是DBSCAN算法的主要步骤：
定义参数：
ε（epsilon）：邻域半径，用于确定一个点的邻域范围。
MinPts：最小点数，用于确定一个核心点。
寻找核心点：
对于数据集中的每个点，计算其ε-邻域内的点的数量。如果该数量大于或等于MinPts，则将该点标记为核心点。
构建聚类：
对于每个核心点或核心点的连通组件（即密度可达的点集合），将其分配给一个新的聚类。
对于核心点ε-邻域内的非核心点，将其分配给与其相邻的核心点所属的聚类。
处理噪声：
将不属于任何聚类的点标记为噪声点（noise）。
DBSCAN算法的优点包括：
不需要预先指定聚类数量，可以自动发现聚类。
能够识别任意形状的聚类，并对噪声数据进行鲁棒处理。
相对于K-means等算法，对参数的敏感度较低。
但是，DBSCAN算法也有一些缺点：
对于密度不均匀的数据集，可能会难以设置合适的ε和MinPts参数。
对于高维数据，由于“维度灾难”的影响，需要谨慎选择距离度量方法和参数设置。
'''

if __name__ == '__main__':
    X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.2],
                      random_state=9)
    plt.scatter(X[:, 0], X[:, 1], marker='o')
    plt.show()
    y_pred = DBSCAN(eps=0.1,min_samples=4).fit_predict(X) # 两个点之间的最大距离 在某个点的邻域内最小的样本数
    # 聚类效果n_clusters=2\3\4
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.show()
    # Calinski-Harabasz Index评估的聚类分数
    print(calinski_harabasz_score(X, y_pred)) # 越大越好

