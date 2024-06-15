import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

if __name__ == '__main__':
    X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.2],
                      random_state=9)
    plt.scatter(X[:, 0], X[:, 1], marker='o')
    plt.show()
    # y_pred = KMeans(n_clusters=4, random_state=9).fit_predict(X)
    y_pred = MiniBatchKMeans(n_clusters=4, random_state=9).fit_predict(X)
    # 聚类效果n_clusters=2\3\4
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.show()
    # Calinski-Harabasz Index评估的聚类分数
    print(calinski_harabasz_score(X, y_pred)) # 越大越好

    '''使用层次聚类'''
    model = AgglomerativeClustering(n_clusters=4)  # 设置聚类数为3
    # 拟合数据并进行聚类
    labels = model.fit_predict(X)
    # 计算轮廓系数
    silhouette_avg = silhouette_score(X, labels)
    print("聚类结果的轮廓系数为:", silhouette_avg)
    # Calinski-Harabasz Index评估的聚类分数
    print(calinski_harabasz_score(X, labels))  # 越大越好
    # 画出聚类树状图
    linkage_matrix = linkage(X, method='ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.show()

