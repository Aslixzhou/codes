import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.utils.data as Data
import random
import matplotlib .pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getXyData():
    x = random.random()
    noiseData = random.random() * 0.5
    y = 5 * x + 3 + noiseData
    return x, y

class Dataset(Data.Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return 500
    def __getitem__(self, i):
        x, y = getXyData()
        xData = torch.FloatTensor([x])
        yData = torch.FloatTensor([y])
        return xData, yData

# 查看数据集
dataset = Dataset()
print(len(dataset))
print(dataset[0])

# 散点图
tempxList = list()
tempyList = list()
for i in range(100):
    x, y = getXyData()
    tempxList.append(x)
    tempyList.append(y)
plt.scatter(tempxList, tempyList)
plt.show()

# 定义数据集加载器
loader = Data.DataLoader(dataset=dataset,
                         batch_size=64,
                         shuffle=True,
                         drop_last=True)
print(len(loader))
x,y = next(iter(loader))
print(x.type,x.shape)
print(y.type,y.shape)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=1, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )

    def forward(self, x):
        return self.fc(x)

model = Model().to(device)
print(model)

epochList = list()
lossList = list()
def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fun = nn.MSELoss()
    model.train()
    for epoch in range(1000):
        for x, y in loader:
            out = model(x.to(device))
            loss = loss_fun(out.to("cpu"), y)
            epochList.append(epoch)
            lossList.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if epoch % 5 == 0:
            print(f"epoch:{epoch}, 损失：{loss.item()}")
    torch.save(model, "./单变量线性回归.model")

@torch.no_grad()
def test():
    model = torch.load("./单变量线性回归.model").to("cpu")
    model.eval()
    xData, yData = next(iter(loader))
    out = model(xData)
    return xData, yData, out

train()
xData, yData, y_pred = test()
print(xData.shape)
print(yData.shape)
print(y_pred.shape)
x1List = xData.numpy() # 张量转数组
# x 是一个需要梯度的张量，可通过 y = x.detach() 来创建一个新的张量 y，具有与 x 相同的数值，但不再与计算图相关联，因此不会影响梯度的传播。
yPreList = y_pred.detach().numpy()
plt.scatter(x1List, yData)
plt.plot(x1List, yPreList, color="r")
plt.show()