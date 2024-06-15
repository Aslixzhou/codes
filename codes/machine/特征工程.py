import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer

if __name__ == '__main__':
    data = [[180, 75, 25], [175, 80, 19], [159, 50, 40], [160, 60, 32]]
    print("-----------区间缩放法-----------")
    scaler = MinMaxScaler(feature_range=(0,2)) #指定归一化区间
    result = scaler.fit_transform(data)
    print(pd.DataFrame(result))

    print("-----------标准化-----------")
    scaler = StandardScaler()
    result = scaler.fit_transform(data)
    print(pd.DataFrame(result))

    print("-----------缺失值处理--------")
    data = np.array([[1, 2, np.nan],
                     [4, np.nan, 6],
                     [np.nan, 8, 9]])
    imputer = SimpleImputer(strategy='mean') # 均值填充
    data = imputer.fit_transform(data)
    print("填充缺失值后的数据：\n", data)

    print("-----------归一化-----------")
    scaler = Normalizer()
    result = scaler.fit_transform(data)
    print(pd.DataFrame(result))

    data = [[180, 75, 25], [175, 80, 19], [159, 50, 40], [160, 60, 32]]
    print("-------定量特征二值化--------")
    scaler = Binarizer(threshold=65) # 阈值65
    result = scaler.fit_transform(data)
    print(pd.DataFrame(data))
    print(pd.DataFrame(result))

    from sklearn.preprocessing import LabelEncoder
    print("----------标签编码----------")
    a = ["male", "female", "male", "error"]
    le = LabelEncoder()
    print(le.fit_transform(a))

    from sklearn.preprocessing import OneHotEncoder
    print("----------独热编码----------")
    # 创建一个包含分类变量的DataFrame
    data = pd.DataFrame({'color': ['红色', '绿色', '蓝色', '绿色']})
    print("data:\n",data)
    # 使用pd.get_dummies进行独热编码
    one_hot_encoded = pd.get_dummies(data['color'])
    print(one_hot_encoded)
    print(OneHotEncoder().fit_transform(data).toarray())

    colors = ['红色', '绿色', '蓝色', '绿色', '红色']
    le = LabelEncoder()
    labels = le.fit_transform(colors)
    print("labels: " ,labels)
    print(OneHotEncoder().fit_transform(labels.reshape(-1,1)).toarray())

    from sklearn.datasets import load_iris
    data, target = load_iris(return_X_y=True)
    from sklearn.preprocessing import label_binarize
    print(target)
    print(label_binarize(target, classes=[0,1,2,3]))

    '''
    标准化：列方向转换 符合正态分布
    区间缩放：列方向转换 没有负值
    归一化：行方向转换 强调数据内部的组成
    '''

