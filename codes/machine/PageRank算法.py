import networkx as nx
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

'''
PageRank算法：解决节点重要性排序问题
当所有边的权重都设置为1时，算法会认为所有连接对节点重要性的影响是相同的，因此节点的重要性主要取决于连接到该节点的其他节点的数量和重要性。
'''

if __name__ == '__main__':
    plt.rcParams['font.sans-serif']=['SimHei'] # 正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False # 正常显示负号
    df = pd.read_csv('data/xi/triples.csv')
    print(df)
    edges = [edge for edge in zip(df['head'],df['tail'])]
    print(edges)
    G = nx.DiGraph()
    G.add_edges_from(edges)
    print(G)
    print(G.edges)
    print(G.nodes)
    print(len(G))
    # 可视化
    # plt.figure(figsize=(15,14))
    pos = nx.spring_layout(G, iterations=3, seed=5)
    # nx.draw(G, pos, with_labels=True)
    # plt.show()
    pagerank = nx.pagerank(G,
                           alpha=0.85,
                           personalization=None,
                           max_iter=100,
                           tol=1e-06,
                           nstart=None,
                           dangling=None,
                           )
    print(pagerank)
    # 从高到低排序
    print(sorted(pagerank.items(), key=lambda x: x[1], reverse=True))
    # 节点尺寸
    node_sizes = (np.array(list(pagerank.values())) * 8000).astype(int)
    # 节点颜色
    M = G.number_of_edges()
    edge_colors = range(2, M+2)
    plt.figure(figsize=(15,14))
    # 绘制节点
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_sizes)
    # 绘制连接
    edges = nx.draw_networkx_edges(
        G,
        pos,
        node_size=node_sizes,
        arrowstyle='->',
        arrowsize=20,
        edge_color=edge_colors,
        edge_cmap=plt.cm.plasma,
        width=4,
    )
    # 设置每个连接的透明度
    edge_alphas = [(5+i)/(M+4) for i in range(M)]
    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()

    '''
    带有向边权重的简单例子
    '''
    # 创建带权重的有向图
    G = nx.DiGraph()
    edges = [("A", "B", 0.5), ("A", "C", 0.4), ("B", "C", 0.3),  ("C", "A", 0.8), ("D", "C", 0.7), ("D", "A", 0.2),]
    G.add_weighted_edges_from(edges)

    # 计算 PageRank 值
    pagerank = nx.pagerank(G, weight='weight')

    # 对节点按 PageRank 值进行排序
    sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

    # 输出节点的重要性排序
    print("节点的重要性排序:")
    for node, importance in sorted_nodes:
        print(f"{node}: {importance}")

    # 绘制图形
    pos = nx.circular_layout(G)  # 使用圆形布局
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color="skyblue")  # 绘制节点
    edge_labels = {(u, v): f"{w}" for u, v, w in G.edges(data='weight')}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', label_pos=0.5)  # 绘制边权重，并调整位置
    plt.title("带权重的有向图")
    plt.show()