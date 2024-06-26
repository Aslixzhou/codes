import numpy as np
import pandas as pd
import networkx as nx

edges = pd.DataFrame()
edges['sources'] = ['A',1,2,3,4,4,6,7,7,9,1,4,4,4,6,7,5,8,9,8]
edges['targets'] = [1,4,4,4,6,7,5,8,9,8,'A',1,2,3,4,4,6,7,7,9]
edges['weights'] = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

G = nx.from_pandas_edgelist(edges, source='sources',target='targets',edge_attr="weights")

# degree
print(nx.degree(G))

# 连通分量
print(list(nx.connected_components(G)))

# 图直径
print(nx.diameter(G))

# 度中心性
print('度中心性',nx.degree_centrality(G))

# 特征向量中心性
print('特征向量中心性',nx.eigenvector_centrality(G))

# betweenness
print('betweenness',nx.betweenness_centrality((G)))

# closeness
print('closeness',nx.closeness_centrality(G))

# pagerank
print('pagerank',nx.pagerank(G))

# HITS
print('HITS',nx.hits(G,tol=0.00001))