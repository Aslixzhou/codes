import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
def load_data():
    import os
    import torchaudio
    xs = []
    ys = []
    #遍历文件夹下的所有文件
    for filename in os.listdir('./dataset/superb'):
        #只要图片,过滤一些无关的文件
        if not filename.endswith('.wav'):
            continue
        #读取声音信息
        x = torchaudio.load('./dataset/superb/%s' % filename)[0]
        x = x.reshape(-1, 1)
        #y来自文件名的第一个字符
        y = int(filename[0])
        xs.append(x)
        ys.append(y)
    return xs, ys

xs, ys = load_data()
print(len(xs), len(ys), xs[0].shape, ys[0])


import torch
#定义数据集
class Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(xs)
    def __getitem__(self, i):
        return xs[i], ys[i]

dataset = Dataset()
x, y = dataset[0]
print(len(dataset), x.shape, y)

#数据集加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=8,
                                     shuffle=True,
                                     drop_last=True)
x, y = next(iter(loader))
print(len(loader), x.shape, y)



#RNN神经网络
class Model(torch.nn.Module):
    #模型初始化部分
    def __init__(self):
        super().__init__()
        #循环层
        self.rnn1 = torch.nn.RNN(input_size=1,
                                 hidden_size=16,
                                 batch_first=True)
        self.rnn2 = torch.nn.RNN(input_size=16,
                                 hidden_size=32,
                                 batch_first=True)
        self.rnn3 = torch.nn.RNN(input_size=32,
                                 hidden_size=64,
                                 batch_first=True)
        self.rnn4 = torch.nn.RNN(input_size=64,
                                 hidden_size=128,
                                 batch_first=True)
        #激活函数
        self.relu = torch.nn.ReLU()
        #池化层
        self.pool = torch.nn.AvgPool1d(kernel_size=7, stride=5)
        #线性输出
        self.fc = torch.nn.Linear(in_features=640, out_features=10)

    #定义神经网络计算过程
    def forward(self, x):
        #循环神经网络计算,抽取特征
        #[8, 4000, 1] -> [8, 4000, 16]
        x, _ = self.rnn1(x)
        x = self.relu(x)
        #池化,缩小数据规模,合并特征
        #[8, 4000, 16] -> [8, 799, 16]
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        #重复上面的计算
        #[8, 799, 16] -> [8, 159, 32]
        x, _ = self.rnn2(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        #[8, 159, 32] -> [8, 31, 64]
        x, _ = self.rnn3(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        #[8, 31, 64] -> [8, 5, 128]
        x, _ = self.rnn4(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        #展平,准备线性计算
        #[8, 5, 128] -> [8, 640]
        x = x.flatten(start_dim=1)
        #线性计算输出
        #[8, 640] -> [8, 10]
        return self.fc(x)

model = Model()
print(model)
print(model(torch.randn(8, 4000, 1)).shape)


#训练
def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fun = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(5):
        for i, (x, y) in enumerate(loader):
            out = model(x)
            loss = loss_fun(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 1000 == 0:
                acc = (out.argmax(dim=1) == y).sum().item() / len(y)
                print(epoch, i, loss.item(), acc)
    torch.save(model, 'RNN_声音分类.model')
train()

#测试
@torch.no_grad()
def test():
    model = torch.load('RNN_声音分类.model')
    model.eval()
    correct = 0
    total = 0
    for i in range(100):
        x, y = next(iter(loader))
        out = model(x).argmax(dim=1)
        correct += (out == y).sum().item()
        total += len(y)
    print(correct / total)
test()