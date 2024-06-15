from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

if __name__ == '__main__':
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data  # 特征数据
    y = iris.target  # 真实类别标签
    print(X)
    print(y)
    # 创建一个层次聚类模型
    model = AgglomerativeClustering(n_clusters=3)  # 设置聚类数为3
    # 拟合数据并进行聚类
    labels = model.fit_predict(X)
    print(labels)
    print(accuracy_score(y,labels))
    # 绘制聚类结果
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title('Hierarchical Clustering of Iris Dataset')
    plt.show()
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
