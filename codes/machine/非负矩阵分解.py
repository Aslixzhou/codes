from sklearn.decomposition import NMF
import numpy as np

'''
非负矩阵分解：将非负的大矩阵分解成两个非负的小矩阵。它使分解后的所有分量均为非负值(要求纯加性的描述)，并且同时实现非线性的维数约减。
'''

if __name__ == '__main__':
    # 创建一个矩阵
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    print("Original matrix A: \n", A)
    print()

    # 构建 NMF 模型，设置 2 个主题
    topics = 2
    nmf_model = NMF(n_components=topics)

    # 训练并转换矩阵
    W = nmf_model.fit_transform(A)
    H = nmf_model.components_

    print("W matrix (Left Non-negative Matrix):")
    print(W)
    print()

    print("H matrix (Right Non-negative Matrix):")
    print(H)
    print()

    # 重构原始矩阵
    A_reconstructed = np.dot(W, H)
    # 由于 NMF 是一种近似分解方法，因此重构后的矩阵不会完全等于原始矩阵。
    print("Reconstructed matrix A: \n", A_reconstructed)
