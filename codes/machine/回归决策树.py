import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model

if __name__ == '__main__':
    x = np.array(list(range(1, 11))).reshape(-1, 1)
    y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])
    print("x: ",x)
    print("y: ",y)
    model = DecisionTreeRegressor(max_depth=3)
    model.fit(x,y)
    X_test = np.arange(0.0, 10.0, 0.01).reshape(-1, 1)  # ⽣成1000个数,⽤于预测模型
    y_ = model.predict(X_test)
    #  结果可视化
    plt.figure(figsize=(10, 6), dpi=100)
    plt.scatter(x, y, label="data")
    plt.plot(X_test, y_,label="max_depth=3")
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()