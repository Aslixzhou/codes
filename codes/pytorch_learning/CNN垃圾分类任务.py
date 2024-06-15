import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loadData():
    xData = list()
    yData = list()
    labelNameDict = dict()
    dataPath = "./垃圾分类数据集/GarbageClassification/GarbageClassification/"
    listdir = os.listdir(dataPath)
    for label, labelName in enumerate(listdir):
        labelNameDict[label] = labelName
        for filename in os.listdir(dataPath + labelName):
            if not filename.endswith('.jpg'):
                continue
            imageSize = (32, 32)
            x = PIL.Image.open(dataPath + labelName + '/' + filename).resize(imageSize)
            x = torch.FloatTensor(np.array(x)) / 255
            # print(x.shape)
            # [32, 32, 3] -> [3, 32, 32]
            x = x.permute(2, 0, 1)
            xData.append(x)
            yData.append(label)
    return xData, yData, labelNameDict

xData, yData, labelNameDict = loadData()
# print(len(xData), len(yData), xData[0].shape, yData[0]) # 15515 15515 torch.Size([3, 32, 32]) 0

class Dataset(Data.Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return len(xData)
    def __getitem__(self, i):
        return xData[i], yData[i]

dataset = Dataset()
# 数据集加载器
loader = Data.DataLoader(dataset=dataset,
                         batch_size=8,
                         shuffle=True,
                         drop_last=True)
x, y = next(iter(loader))
# print(len(loader), x.shape, y)

# 搭建分类模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels=3,
                              out_channels=16,
                              kernel_size=5,
                              stride=2,
                              padding=0)
        self.cnn2 = nn.Conv2d(in_channels=16,
                              out_channels=32,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.cnn3 = nn.Conv2d(in_channels=32,
                              out_channels=128,
                              kernel_size=7,
                              stride=1,
                              padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=128, out_features=12)
    def forward(self, x):
        print(x.shape)
        # [8 3 32 32] -> [8 16 14 14]
        x = self.cnn1(x)
        x = self.relu(x)
        # [8 16 14 14] -> [8 32 14 14]
        x = self.cnn2(x)
        x = self.relu(x)
        # [8 32 14 14] -> [8 32 7 7]
        x = self.pool(x)
        # [8 32 7 7] -> [8 128 1 1]
        x = self.cnn3(x)
        x = self.relu(x)
        # [8 128 1 1] -> [8 128]
        x = x.flatten(start_dim=1)
        return self.fc(x)


model = Model().to(device)
print(model)


def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fun = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(5):
        for step, (x, y) in enumerate(loader):
            out = model(x.to(device))
            loss = loss_fun(out.to("cpu"), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 200 == 0:
                acc = (out.to("cpu").argmax(dim=1) == y).sum().item() / len(y)
                print(f"epoch：{epoch}轮，序号：{step:<5}，损失：{loss.item():<20}，准确率：{acc}")
    torch.save(model, "./垃圾分类.model")

@torch.no_grad()
def test():
    model = torch.load("./垃圾分类.model").to("cpu")
    model.eval()
    correct = 0
    total = 0
    for i in range(100):
        x, y = next(iter(loader))
        out = model(x).argmax(dim=1)
        correct += (out == y).sum().item()
        total += len(y)
    print("准确率：" + str(correct / total))

train()
test()

x, y = next(iter(loader))
model = torch.load("./垃圾分类.model").to("cpu")
out = model(x).argmax(dim=1)
print(x.shape)

plt.rcParams['font.sans-serif'] = ['SimHei']
titles = [f"预测{labelNameDict[pre]}，真实：{labelNameDict[label]}" for pre, label in zip(out.tolist(), y.tolist())]
image = [img for img in x.permute(0, 2, 3, 1).numpy()]
plt.figure(figsize=(15, 5))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(image[i])
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()