import networkx as nx
from gensim.models import Word2Vec
import random
import numpy as np
import matplotlib.pyplot as plt

# 构建一个简单的图
from sklearn.cluster import KMeans

G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4), (4, 5)])

# 绘制图
pos = nx.spring_layout(G)  # 定义节点排列方式
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1000, font_size=10, font_color='black')
plt.title('Simple Graph')
plt.show()

# 执行随机游走
walks = []
num_walks = 10
walk_length = 5
for _ in range(num_walks):
    for node in G.nodes():
        walk = [node]
        for _ in range(walk_length):
            neighbors = list(G.neighbors(walk[-1]))
            if len(neighbors) > 0:
                walk.append(random.choice(neighbors))
        walks.append([str(node) for node in walk])

# 使用 Word2Vec 模型学习节点表示
model = Word2Vec(walks, vector_size=128, window=5, min_count=0, sg=1, workers=2, epochs=100)

# 查看节点的向量表示
print(model.wv['1'])  # 输出节点 1 的向量表示

# 假设你已经有了节点向量，用 node_vectors 表示
node_vectors = {
    1: np.array(model.wv['1']),  # 第一个节点的向量表示
    2: np.array(model.wv['2']),
    3: np.array(model.wv['3']),
    4: np.array(model.wv['4']),
    5: np.array(model.wv['5']),
    # 其他节点的向量表示
}

# 将节点向量转换为数组
X = np.array(list(node_vectors.values()))

# 使用K均值算法进行聚类
k = 2
kmeans = KMeans(n_clusters=k, random_state=0)  # 假设要分成3个簇
kmeans.fit(X)

# 获取聚类结果
cluster_labels = kmeans.labels_

# 绘制图，并根据聚类结果标记不同颜色
# G = nx.Graph()  # 创建一个简单的图
# # 添加图的边
# # ...
# G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4), (4, 5)])

# pos = nx.spring_layout(G)  # 定义节点排列方式

cluster_colors = ['#'+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(k)]
# 根据聚类结果标记不同颜色
colors = [cluster_colors[label] for label in cluster_labels]
# colors = ['r' if label == 0 else 'b' if label == 1 else 'g' for label in cluster_labels]  # 不同簇使用不同颜色
nx.draw(G, pos, with_labels=True, node_color=colors, node_size=1000, font_size=10, font_color='black')
plt.title('Graph with Node Clustering using K-means')
plt.show()

