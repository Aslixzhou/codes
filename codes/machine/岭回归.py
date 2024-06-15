from sklearn import linear_model
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

if __name__ == '__main__':
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, random_state=66)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    alphas = np.linspace(0.001,1)
    # estimator = linear_model.Ridge(alpha=1.0, fit_intercept=True, solver="auto", normalize=False)
    estimator = linear_model.RidgeCV(alphas=alphas,cv=10) # 加交叉验证
    estimator.fit(x_train, y_train)

    print(estimator.best_score_)
    print(estimator.alpha_)

    y_predict = estimator.predict(x_test)
    print(estimator.score(x_train,y_train))
    print(estimator.score(x_test, y_test)) # 测试得分
    print("预测值为:\n", y_predict)
    print("模型中的系数为:\n", estimator.coef_)
    print("模型中的偏置为:\n", estimator.intercept_)
    error = mean_squared_error(y_test, y_predict)
    print("均方误差为:\n", error)


    print("\n--------------Lasso-------------")
    estimator = linear_model.LassoCV()
    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    print(estimator.score(x_train,y_train))
    print(estimator.score(x_test, y_test)) # 测试得分
    print("预测值为:\n", y_predict)
    print("模型中的系数为:\n", estimator.coef_)
    print("模型中的偏置为:\n", estimator.intercept_)
    error = mean_squared_error(y_test, y_predict)
    print("均方误差为:\n", error)

    print("\n--------------ElasticNet-------------")
    estimator = linear_model.ElasticNetCV()
    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    print(estimator.score(x_train, y_train))
    print(estimator.score(x_test, y_test))  # 测试得分
    print("预测值为:\n", y_predict)
    print("模型中的系数为:\n", estimator.coef_)
    print("模型中的偏置为:\n", estimator.intercept_)
    error = mean_squared_error(y_test, y_predict)
    print("均方误差为:\n", error)