from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

''' 基于决策树的Boosting
    loss: 损失函数，用于优化的损失函数。通常可选值包括"deviance"（对数似然损失，适用于分类问题）和"exponential"（指数损失，通常用于Adaboost）等。
    learning_rate: 学习率，控制每棵树的贡献。较低的学习率意味着需要更多的树来构建模型，但往往会获得更好的泛化能力。
    n_estimators: 基础决策树的数量，也就是要构建的树的数量。
    subsample: 在训练每棵树时使用的子样本比例。如果小于1.0，意味着使用随机梯度提升（Stochastic Gradient Boosting），从而防止过拟合。
    criterion: 衡量分裂质量的函数，例如均方误差（mse）或者对数似然损失（deviance）等。
    min_samples_split: 内部节点再划分所需的最小样本数。这个值越大，可以防止过拟合。
    min_samples_leaf: 叶节点上的最小样本数。类似于min_samples_split，但是应用于叶节点。
    min_weight_fraction_leaf: 叶节点上的所有样本的最小加权分数总和的要求。
    max_depth: 每棵树的最大深度。控制树的复杂度，避免过拟合。
    min_impurity_decrease: 如果分裂导致杂质减少大于或等于这个值，才会分裂。
    init: 初始化的提升树模型，可以用来提供一个初始估计。
    random_state: 随机数种子，用于控制随机性。
    max_features: 寻找最佳分割时考虑的特征数量。
    verbose: 控制输出的详细程度。
    max_leaf_nodes: 每棵树上最大叶节点数。
    warm_start: 当设置为True时，使用之前的解决方案以适应并且添加更多的估计器，否则，只需拟合一组估计器。
    validation_fraction: 用于早期停止的验证集的比例。
    n_iter_no_change: 连续n次迭代损失未发生改变时，训练将在第n次发生改变时停止。
    tol: 早期停止的终止标准。
    ccp_alpha: 用于最小成本复杂度修剪（Minimal Cost-Complexity Pruning）的复杂度参数。
    '''

if __name__ == '__main__':
    # 加载数据集
    iris = load_iris()
    X, y = iris.data, iris.target
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 创建梯度提升树分类器
    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    # 训练模型
    gb_clf.fit(X_train, y_train)
    # 预测并计算准确率
    y_pred = gb_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)