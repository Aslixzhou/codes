import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

if __name__ == '__main__':
    import warnings
    from matplotlib.cbook import MatplotlibDeprecationWarning
    # 屏蔽特定类型的警告
    warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
    x_train = np.array([1,2,3,4,5,6,7,8,9,10])
    x_train = x_train[:, np.newaxis]
    print(x_train) # 转为二维数组
    y_train = [45000,50000,60000,80000,110000,150000,200000,300000,500000,1000000]
    print("-------------多项式回归--------------")
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=5) # 幂次
    poly.fit(x_train, y_train)
    x_poly = poly.transform(x_train) # 特征处理
    print(x_poly)
    linear = LinearRegression()
    linear.fit(x_poly, y_train)
    print("得分：" ,linear.score(x_poly, y_train))
    print("权重：" ,linear.coef_)
    print("偏置：" ,linear.intercept_)
    plt.plot(x_train,y_train,'b.')
    plt.plot(x_train,linear.predict(poly.fit_transform(x_train)),c='r')
    plt.show()
