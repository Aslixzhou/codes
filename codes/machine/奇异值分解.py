import numpy as np

if __name__ == '__main__':
    # 创建一个矩阵
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    print("Original matrix A: \n",A)
    print()

    # 对矩阵进行奇异值分解
    U, S, V = np.linalg.svd(A)

    # 将奇异值数组S表示为对角矩阵
    S_matrix = np.diag(S)
    # 打印奇异值数组S的矩阵形式
    print("Singular values matrix (S):")
    print(S_matrix)
    print()

    # U是左奇异矩阵
    print("Left singular vectors (U):")
    print(U)
    print()

    # V是右奇异矩阵的转置
    print("Right singular vectors (V^T):")
    print(V)


    print("--------------------------------------")

    '''
    紧奇异值分解是与原始矩阵等秩的奇异值分解，
    截断奇异值分解是比原始矩阵低秩的奇异值分解。
    '''
    # 紧奇异值分解
    # 创建一个矩阵
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    # 进行紧奇异值分解
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # 重构原始矩阵
    reconstructed_A = np.dot(U, np.dot(np.diag(S), Vt))

    print("Original matrix A:")
    print(A)
    print()

    print("Reconstructed matrix from SVD:")
    print(reconstructed_A)
    print()

    # 将奇异值数组S表示为对角矩阵
    S_matrix = np.diag(S)
    # 打印奇异值数组S的矩阵形式
    print("Singular values matrix (S):")
    print(S_matrix)
    print()

    # U是左奇异矩阵
    print("Left singular vectors (U):")
    print(U)
    print()

    # V是右奇异矩阵的转置
    print("Right singular vectors (V^T):")
    print(Vt)

    print("--------------------------------------")
    # 截断奇异值分解
    # 创建一个矩阵
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    # 指定截断奇异值的数量
    k = 2

    # 进行截断奇异值分解
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    U_truncated = U[:, :k]
    S_truncated = np.diag(S[:k])
    Vt_truncated = Vt[:k, :]
    # 重构原始矩阵
    reconstructed_A = np.dot(U_truncated, np.dot(S_truncated, Vt_truncated))

    print("Original matrix A:")
    print(A)
    print()

    print("Reconstructed matrix from truncated SVD:")
    print(reconstructed_A)
    print()

    # 将奇异值数组S表示为对角矩阵
    S_matrix = np.diag(S)
    # 打印奇异值数组S的矩阵形式
    print("Singular values matrix (S):")
    print(S_matrix)
    print()

    # U是左奇异矩阵
    print("Left singular vectors (U):")
    print(U)
    print()

    # V是右奇异矩阵的转置
    print("Right singular vectors (V^T):")
    print(Vt)


