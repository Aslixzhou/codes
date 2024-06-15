import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

if __name__ == '__main__':
    diabetes = load_diabetes()
    data = diabetes['data']
    target = diabetes['target']
    feature_names = diabetes['feature_names']
    df = pd.DataFrame(data, columns=feature_names)
    print(df)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    linear = LinearRegression()
    linear.fit(x_train, y_train)
    y_pred = linear.predict(x_test)
    print(y_pred,y_test)
    print(linear.score(x_train,y_train))
    print(linear.score(x_test, y_test)) # 测试得分
    print(mse(y_test, y_pred)) # 均方误差
    print(linear.coef_)
    print(linear.intercept_)