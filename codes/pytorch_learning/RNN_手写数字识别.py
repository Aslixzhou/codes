import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataPath = "./dataset"
train_data = torchvision.datasets.MNIST(
    root=dataPath,
    train=True,
    transform=transforms.ToTensor(),
    download=False
)
test_data = torchvision.datasets.MNIST(
    root=dataPath,
    train=False,
    transform=transforms.ToTensor(),
    download=False
)

# 数据集加载器
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True,
    drop_last=True
)
test_loader = Data.DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=True,
    drop_last=True
)

for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
print(b_x.shape)
print(b_y.shape)


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        '''
        input_dim：输入的特征维度。
        hidden_dim：RNN 隐藏层的维度。隐藏层的神经元数
        layer_dim：RNN 的层数。
        output_dim：输出的维度。
        '''
        '''
        在循环神经网络（RNN）中，hidden_dim 表示 RNN 隐藏层的维度，即隐藏状态的大小或隐藏单元的数量。
        在 RNN 模型中，每一个时间步的输入数据通过 RNN 单元的计算会得到一个隐藏状态作为输出，
        该隐藏状态的维度即为 hidden_dim。
        '''
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim,
                          hidden_dim,
                          layer_dim,
                          batch_first=True,
                          nonlinearity='relu',
                          bidirectional=False)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x): # (batch_size, seq_length, input_dim)
        out, h_n = self.rnn(x, None)
        '''  (batch_size, seq_length, hidden_dim)
        out 是每个时间步的输出结果。
        h_n 是最终的隐藏状态。
        '''
        out = self.fc(out[:, -1, :])
        '''
        out[:, -1, :] 表示取 out 张量中的最后一个时间步的输出。
        具体来说，假设 out 的维度为 (batch_size, seq_length, hidden_dim)，其中：
        batch_size 表示输入数据的批次大小；
        seq_length 表示序列数据的长度，也就是 RNN 模型时间步的数量；
        hidden_dim 表示隐藏状态的维度，即隐藏层的大小。
        对于 out[:, -1, :]：
            : 表示取所有批次的数据；
            -1 表示取最后一个时间步的索引，即取序列数据的最后一个输出；
            : 表示取所有隐藏状态的维度，即全部保留。
        因此，out[:, -1, :] 将得到一个大小为 (batch_size, hidden_dim) 的张量，其中包含了每个批次在最后一个时间步的输出。
        '''
        return out

# 模型初始化
input_dim = 28
hidden_dim = 128
layer_dim = 2
output_dim = 10
model = Model(input_dim, hidden_dim, layer_dim, output_dim).to(device)
print(model)

def train():
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)
    loss_fun = nn.CrossEntropyLoss()
    train_loss_all = list()
    train_acc_all = list()
    model.train()
    for epoch in range(20):
        print("Epoch：" + str(epoch + 1))
        corrects = 0
        train_num = 0
        all_loss = 0
        for step, (x, y) in enumerate(train_loader):
            xData = x.view(-1, 28, 28)
            output = model(xData.to(device))
            pre_lab = torch.argmax(output.to("cpu"), 1)
            loss = loss_fun(output.to("cpu"), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            all_loss += loss.item() * x.size(0)
            corrects += torch.sum(pre_lab == y.data)
            train_num += x.size(0)
        train_loss_all.append(all_loss / train_num)
        train_acc_all.append(corrects.double().item() / train_num)
        print(f"{epoch + 1} loss：{train_loss_all[-1]:.4f} Acc：{train_acc_all[-1]:.4f}")
    torch.save(model, "./RNN_手写数字分类.model")
    return train_loss_all, train_acc_all

train_loss_all, train_acc_all = train()

@torch.no_grad()
def test():
    model = torch.load("./RNN_手写数字分类.model").to("cpu")
    model.eval()
    x, y = next(iter(test_loader))
    xData = x.view(-1, 28, 28)
    output = model(xData)
    pre_lab = torch.argmax(output, 1)
    correct = torch.sum(pre_lab == y.data)
    total = len(y)
    print(pre_lab)
    print(y.data)
    print("准确率：" + str(correct / total))

test()

x, y = next(iter(test_loader))
model = torch.load("./RNN_手写数字分类.model").to("cpu")
xData = x.view(-1, 28, 28)
output = model(xData).argmax(1)
plt.rcParams['font.sans-serif'] = ['SimHei']
title = [f"预测：{pre}, 真实：{label}" for pre, label in zip(output.tolist(), y.tolist())]
images = [img for img in x.permute(0, 2, 3, 1).numpy()]
plt.figure(figsize=(15, 35))
for i in range(64):
    plt.subplot(16, 4, i + 1)
    plt.imshow(images[i])
    plt.title(title[i])
    plt.xticks([]), plt.yticks([])
plt.show()

train_loss_all_np = list()
for i in range(20):
    train_loss_all_np.append(train_loss_all[i])
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_all_np, "ro-", label="Train loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.subplot(1, 2, 2)
plt.plot(train_acc_all, "ro-", label="Train Acc")
plt.xlabel("epoch")
plt.ylabel("Acc")
plt.legend()
plt.show()