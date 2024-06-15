from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

if __name__ == '__main__':
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, random_state=22)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    estimator = SGDRegressor(max_iter=1000,learning_rate="constant",eta0=0.1)
    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    print(estimator.score(x_train,y_train))
    print(estimator.score(x_test, y_test)) # 测试得分
    print("预测值为:\n", y_predict)
    print("模型中的系数为:\n", estimator.coef_)
    print("模型中的偏置为:\n", estimator.intercept_)
    error = mean_squared_error(y_test, y_predict)
    print("均方误差为:\n", error)

