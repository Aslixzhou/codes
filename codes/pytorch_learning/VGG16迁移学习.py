import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torchvision import transforms
from torchvision.models import get_model
from torchvision.transforms import v2
from torchvision import models
import PIL.Image
import matplotlib.pyplot as plt
# 临时改变环境变量 用于保存预训练模型的位置
os.environ['TORCH_HOME'] = './pre_model'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataPath = "./dataset"

train_data = torchvision.datasets.MNIST(
    root=dataPath,
    train=True,
    transform=v2.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # 单通道->三通道 一个通道复制三次
    ]),
    download=False
)

test_data = torchvision.datasets.MNIST(
    root=dataPath,
    train=False,
    transform=v2.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ]),
    download=False
)

# 数据集加载器
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=32,
    shuffle=True,
    drop_last=True
)

test_loader = Data.DataLoader(
    dataset=test_data,
    batch_size=32,
    shuffle=True,
    drop_last=True
)

# 显示图片
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
print(b_x.shape)
print(b_y.shape)
plt.imshow(b_x[0].permute(1, 2, 0).numpy())
plt.show()

# 导入vgg16
vgg16 = models.vgg16(pretrained=True)
vgg16 = get_model("vgg16", weights="DEFAULT")
print(vgg16)

'''
首先，通过 vgg = vgg16.features 将 VGG16 模型的特征提取部分提取出来，并赋值给变量 vgg。在这里，特征提取部分包括了神经网络的卷积层、池化层等用于提取图像特征的部分，而不包括后面的分类器部分。
接下来，通过 for param in vgg.parameters(): 遍历特征提取部分 vgg 中的所有参数。
对于每个参数，通过 param.requires_grad_(False) 将其 requires_grad 属性设置为 False，表示这些参数不需要梯度更新。这样，将特征提取部分的参数设置为不需要梯度更新后，这些参数将在模型的训练过程中保持固定不变。
通过将特征提取部分的参数设置为不需要梯度更新，我们可以保持这部分参数的固定状态，避免在训练过程中对这部分
'''

vgg = vgg16.features
for param in vgg.parameters():
    param.requires_grad_(False) # 参数冻结

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.vgg = vgg
        self.classifier = nn.Sequential(
            nn.Linear(25088, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.vgg(x)
        x = x.flatten(start_dim=1)
        output = self.classifier(x)
        return output

model = Model().to(device)
print(model)

def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fun = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(2):
        for index, (x, y) in enumerate(train_loader):
            out = model(x.to(device))
            loss = loss_fun(out.to("cpu"), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if index % 10 == 0:
                acc = (out.to("cpu").argmax(dim=1) == y).sum().item() / len(y)
                print(f"{epoch + 1} 序号：{index:<5} 损失：{loss.item():<20} 准确率：{acc}")
    torch.save(model, "./VGG16迁移学习.model")


@torch.no_grad()
def test():
    model = torch.load("./VGG16迁移学习.model").to("cpu")
    model.eval()
    x, y = next(iter(test_loader))
    output = model(x)
    pre_lab = torch.argmax(output, 1)
    correct = torch.sum(pre_lab == y.data)
    total = len(y)
    print(pre_lab)
    print(y.data)
    print("准确率：" + str(correct / total))


train()
test()

x, y = next(iter(test_loader))
model = torch.load("./10迁移学习数字计算器.model").to("cpu")
out = model(x).argmax(1)

plt.rcParams['font.sans-serif'] = ['SimHei']
title = [f"预测：{pre}, 真实：{label}" for pre, label in zip(out.tolist(), y.tolist())]
images = [img for img in x.permute(0, 2, 3, 1).numpy()]
plt.figure(figsize=(15, 35))
for i in range(64):
    plt.subplot(16, 4, i + 1)
    plt.imshow(images[i])
    plt.title(title[i])
    plt.xticks([]), plt.yticks([])
plt.show()


