from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, random_state=66)
    gbdt = GradientBoostingClassifier()
    gbdt.fit(x_train,y_train)
    print("特征重要性: ", gbdt.feature_importances_) # 可以用于特征选择
    print(gbdt.score(x_train, y_train))
    y_pred = gbdt.predict(x_test)
    print("y_test:", y_test)
    print("y_pred:", y_pred)
    print(gbdt.score(x_test,y_test))

    print("-------特征选择----------")
    '''
    REF: 递归训练不断消除每个不重要的特征直到达到需要的特征个数为止
    '''
    feature_names = data['feature_names']
    print("feature_names: ", feature_names)
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    # 初始化模型
    lr = LogisticRegression(max_iter=2000)
    # 初始化RFE并选择特征数量
    rfe = RFE(estimator=lr, n_features_to_select=2)
    # 拟合RFE
    rfe.fit(x_train, y_train)
    # 打印选择的特征
    print('Selected features:', rfe.support_)

    # 初始化模型
    gbdt = GradientBoostingClassifier()
    # 初始化RFE并选择特征数量
    rfe = RFE(estimator=gbdt, n_features_to_select=3)
    # 拟合RFE
    rfe.fit(x_train, y_train)
    # 打印选择的特征
    print('Selected features:', rfe.support_)

    '''重要性阈值特征选择'''
    from sklearn.feature_selection import SelectFromModel
    print(SelectFromModel(estimator=gbdt, threshold=0.2).fit_transform(x_train, y_train))
