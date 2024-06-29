import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import pandas as pd

#加载处理好的数据集,每句话是15个词,y是2分类,字典在data/sst2/vocab.txt
data = pd.read_csv('./dataset/sst2/data.csv')
print(data,data.shape)

import torch

# 定义数据集
class Dataset(torch.utils.data.Dataset):

    def __len__(self):
        return len(data)
    def __getitem__(self, i):
        # 取数据
        x, y = data.iloc[i]
        # 以逗号分割x数据,转换为向量
        x = [int(i) for i in x.split(',')]
        x = torch.LongTensor(x)
        # y不需要太特别的处理
        y = int(y)
        return x, y

dataset = Dataset()
print(len(dataset), dataset[0])


#数据集加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=8,
                                     shuffle=True,
                                     drop_last=True)

#全连接神经网络
class Model(torch.nn.Module):
    #模型初始化部分
    def __init__(self):
        super().__init__()
        #词编码层,30522是词的数量,每个词会被编码为100维的向量
        self.embed = torch.nn.Embedding(num_embeddings=30522,
                                        embedding_dim=100)
        #RNN单元
        self.cell = torch.nn.GRUCell(input_size=100, hidden_size=512)
        #线性输出
        self.fc = torch.nn.Linear(in_features=512, out_features=2)
    #定义神经网络计算过程
    def forward(self, x):
        #每个词编码为100维的向量
        #[8, 15] -> [8, 15, 100]  8个数据 15个词 每个词是100维向量
        x = self.embed(x)
        # print("x.shape: ",x.shape) # torch.Size([8, 15, 100])
        #初始记忆为空
        h = None
        #从前向后读句子中的每一个词
        for i in range(x.shape[1]):
            #[8, 100],[8, 512] -> [8, 512]
            # print(x[:,i].shape) # torch.Size([8, 100])
            h = self.cell(x[:, i], h)
            '''
            假设 x 是一个形状为 [batch_size, sequence_length, feature_size] 的三维张量，其中：
                batch_size 是批次中样本的数量。
                sequence_length 是序列的长度（例如，在文本处理中，可以是句子中单词的数量）。
                feature_size 是每个元素的特征数量（例如，在词嵌入中，可以是每个词向量的维度）。
            使用 x[:, i] 进行索引时：
                第一个 : 表示选择所有批次中的元素。
                i 表示选择序列中的第 i 个时间步。
                因此，x[:, i] 的结果是一个形状为 [batch_size, feature_size] 的张量，它包含了批次中每个样本在第 i 个时间步的特征向量。
                例如，如果 x 表示一个批次中所有句子的词嵌入，x[:, i] 将给出这个批次中所有句子在第 i 个词的嵌入向量。
            '''
        #根据最后一个词的记忆,分类整句话
        #[8, 512] -> [8, 2]
        return self.fc(h)

model = Model()
print(model(torch.ones(8, 15).long()).shape) # torch.Size([8,2])

#训练
def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fun = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(2):
        for i, (x, y) in enumerate(loader):
            out = model(x)
            loss = loss_fun(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 2000 == 0:
                acc = (out.argmax(dim=1) == y).sum().item() / len(y)
                print(epoch, i, loss.item(), acc)
    torch.save(model, 'rnn_text.model')

train()

#测试
@torch.no_grad()
def test():
    model = torch.load('rnn_text.model')
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