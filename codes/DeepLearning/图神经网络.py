import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 创建一个简单的图
    G = nx.Graph()
    G.add_edge('A', 'B')
    G.add_edge('A', 'C')
    G.add_edge('A', 'D')
    G.add_edge('B', 'C')
    G.add_edge('C', 'F')
    G.add_edge('C', 'E')
    G.add_edge('F', 'E')
    # 重新标记节点为0, 1, 2, 3
    # mapping = {node: i for i, node in enumerate(G.nodes())}
    # G = nx.relabel_nodes(G, mapping)
    # 将图转换为邻接矩阵
    adj_matrix = nx.to_numpy_array(G)
    # 打印邻接矩阵
    print("邻接矩阵：")
    print(adj_matrix)

    # 计算连接数矩阵
    degree_matrix = np.diag(list(dict(G.degree()).values()))
    # 打印连接数矩阵
    print("连接数矩阵：")
    print(degree_matrix)

    # 绘制图形
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=1000, font_size=20, font_weight='bold')
    plt.show()

    
