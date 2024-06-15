import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataPath = "./波士顿房价数据集/boston.csv"
df = pd.read_csv(dataPath)
print(df.shape)

x_data = df.iloc[:, :12].values # 选取 DataFrame 前12列的方法
y_data = df.MEDV.values
print(x_data.shape)
print(y_data.shape)

# 数据标准化
scaler = StandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=56)

# 转换为张量
train_xt = torch.from_numpy(X_train.astype(np.float32))
train_yt = torch.from_numpy(y_train.astype(np.float32))
test_xt = torch.from_numpy(X_test.astype(np.float32))
test_yt = torch.from_numpy(y_test.astype(np.float32))

# 数据集加载器
train_data = Data.TensorDataset(train_xt, train_yt)
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=32,
                               shuffle=True,
                               drop_last=True)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=12, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1)
        )
    def forward(self, x):
        return self.fc(x)

model = Model().to(device)
print(model)

def train():
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    loss_fun = nn.MSELoss()
    model.train()
    for epoch in range(80):
        for step, (x, y) in enumerate(train_data):
            out = model(x.to(device))
            loss = loss_fun(torch.squeeze(out.to("cpu")), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if epoch % 5 == 0:
            print(f"epoch:{epoch}, 损失：{loss.item()}")
    torch.save(model, "./波士顿房价预测.model")

@torch.no_grad()
def test(test_xt):
    model = torch.load("./波士顿房价预测.model").to("cpu")
    model.eval()
    out = model(test_xt)
    return out

train()
pred = test(test_xt)
pred = pred.squeeze() # 转为一维
# new_shape = (1, pred.shape[0])  # 将形状设定为 (1, n)，其中 n 是原始张量的行数
# pred = pred.reshape(new_shape) # (1,50)
# pred = pred.view(-1)  # -1 表示自动推断该维度的大小
print(pred,pred.shape) # (50,)
print(test_yt,test_yt.shape) # (50,)


