'''
    自然语言处理（NLP）：RNN 可以用于文本生成、情感分析、命名实体识别、机器翻译、语言建模等任务。
    语音识别：RNN 可以用于语音识别和语音生成，例如语音转文本和文本转语音等任务。
    时间序列预测：RNN 可以用于股票价格预测、天气预测、交通流量预测等时间序列数据预测任务。
    图像描述生成：RNN 可以结合卷积神经网络（CNN）用于生成图像描述，实现图像和文本的对应关系学习。
    推荐系统：RNN 可以用于个性化推荐系统，根据用户的历史行为预测用户的未来偏好。
    时间序列分类：RNN 可以用于动作识别、心电图分类等时间序列数据的分类任务。
    手写识别：RNN 可以用于手写字符识别，例如手写数字或手写字母的识别。
    序列生成：RNN 可以生成类似于序列数据的输出，例如音乐生成、代码生成等。
    关系建模：RNN 可以用于建模序列数据之间的时序关系和依赖关系。
'''

'''
本案例使用Tushare获取某一只股票的历史数据。
把某一只股票的历史数据且分为训练数据集和测试数据集（训练集和测试集都是已知的、已经发生的历史数据）
在训练集上训练LSTM模型
利用训练好的模型，在测试集上进行预测 （测试集也是历史数据）
把在测试集上测得的结果与实际历史走势进行比较，看他们的一致程度
如果一致程度较好，就可以用来进行未来股价的预测（连续5天）

'''
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
from torchvision import transforms
import numpy as np
import math, random
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# import tushare as ts  # Tushare是一个免费、开源的python财经数据接口包
# ts.set_token('37d6c441a3c18daae39f578d92c1831d6ba8beffb2afa2840de1e62c') # 18332139833 zhou27166.
# # 获取某只股票的历史数据
# pro = ts.pro_api()
# df = pro.daily(start_date = '20180101',end_date = '20240606',ts_code = '600519.SH',)
# df_sorted = df.sort_values(by='trade_date')
# print(df_sorted,df_sorted.shape)
# df_sorted.to_csv('rnn_file.csv', index=False)


# 读取 CSV 文件
df = pd.read_csv('rnn_file.csv')
df=df[["open","close","high","low","vol"]]
print(df,df.shape)
# 获取股票波动的范围
close_max = df["close"].max()
close_min = df['close'].min()
print("最高价=", close_max)
print("最低价=", close_min)
print("波动值=", close_max - close_min)
print("上涨率=", (close_max - close_min) / close_min)
print("下跌率=", (close_max - close_min) / close_max)

#对输入数据进行归一化
df = df.apply(lambda x:(x-min(x))/(max(x)-min(x)))
print(df,df.shape)

# 检查归一化后的数据范围：在[0,1] 之间
close_max_n = df["close"].max()
close_min_n = df['close'].min()
print("最高价=", close_max_n)
print("最低价=", close_min_n)

# 思路：
# 根据前n天的数据，预测当天的收盘价(close), 例如，根据1月1-10日的数据(包含5个特征) 预测 1月11日的收盘价(一个值)
# 前n天的所有维度的数据为样本数据，而n+1的收盘价为标签数据。
# sequence的长度，表明了“块”相关数据的长度，即“块长”
# 本案例，并没有把“块”与块在外部连接起来，如果连接了，则相关性就扩展到整个数据集，而不是seq长度。

# 这个例子中：
# sequence length = 10： 序列长度
# input_size=5        ： 数据数据的维度

total_len = df.shape[0]
print("df shape =", df.shape)
print("df len  =", total_len)

print("")
print("按照序列的长度，重新结构化数据集")
sequence = 10
X = []
Y = []

# 一个连续sequence长度的数据为一个序列（输入序列），一个序列对应一个样本标签（预测值）
for i in range(df.shape[0] - sequence):
    X.append(np.array(df.iloc[i:(i + sequence), ].values, dtype=np.float32))
    Y.append(np.array(df.iloc[(i + sequence), 1], dtype=np.float32))

print("train data  of item  0: \n", X[0])
print("train label of item  0: \n", Y[0])

# 序列化后，样本数据的总长少了sequence length
print("\n序列化后的数据形状：")
X = np.array(X)
Y = np.array(Y)
Y = np.expand_dims(Y, 1)
print("X.shape =", X.shape)
print("Y.shape =", Y.shape)

# 通过切片的方式把数据集且分为训练集+验证集
# X[start: end; step]
# 数据集最前面的70%的数据作为训练集
train_x = X[:int(0.8 * total_len)]
train_y = Y[:int(0.8 * total_len)]
# 数据集前80%后的数据（20%）作为验证集
valid_x = X[int(0.8 * total_len):]
valid_y = Y[int(0.8 * total_len):]
print(train_x.shape)
print(train_y.shape)
print(valid_x.shape)
print(valid_y.shape)


# 把读取到的股票的数据，认为的分为训练集合测试集
class Mydataset(Dataset):

    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        return x1, y1

    def __len__(self):
        return len(self.x)


# 构建适合dataload的数据集
dataset_train = Mydataset(train_x, train_y)
dataset_valid = Mydataset(valid_x, valid_y)
# 启动dataloader
batch_size = 8
# 关闭shuffle，这样确保数据的时间顺序与走势与实际一致
train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=False)
print(train_loader)
print(test_loader)


# 闭环模型
class LSTM(nn.Module):
    # input_size:  输入层样本特征向量的长度
    # hidden_size：隐藏层输出特征向量的长度
    # num_layers：隐藏层的层数
    # output_size：整个网络的输出特征的长度
    def __init__(self, input_size=5, hidden_size=32, num_layers=1, output_size=1, batch_first=True,
                 batch_size=batch_size, is_close_loop=False):
        super(LSTM, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_first = batch_first
        self.is_close_loop = is_close_loop
        self.hidden0 = torch.zeros(num_layers, batch_size, hidden_size)
        self.cell0 = torch.zeros(num_layers, batch_size, hidden_size)

        # 定义LSTM网络
        # input_size:  输入层样本特征向量的长度
        # hidden_size：隐藏层输出特征向量的长度
        # num_layers：隐藏层的层数
        # batch_first=true： 数据格式为{batch，sequence，input_size}
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=batch_first)

        # 定义网络的输出层：
        # hidden_size：输出层的输入，隐藏层的特征输出
        # output_size：输出层的输出
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.output_size, bias=True)

    # 定义前向运算，把各层串联起来
    def forward(self, x):
        # 输入层直接送到lstm网络中
        # 输入层数据格式：x.shape = [batch, seq_len, hidden_size]
        # 隐藏层输出数据格式：hn.shape = [num_layes * direction_numbers, batch, hidden_size]
        # 隐藏层输出数据格式：cn.shape = [num_layes * direction_numbers, batch, hidden_size]
        '''
        out 是 LSTM 网络的输出，它是一个三维张量。如果 batch_first=True （通常是这样设置的），其形状为 [batch_size, seq_len, hidden_size] 。out 包含了整个输入序列在每个时间步的隐藏状态的输出。
        hidden 是 LSTM 网络最后一个时间步的隐藏状态，其形状为 [num_layers * direction_numbers, batch_size, hidden_size] 。对于常见的 LSTM ，direction_numbers 通常为 1 ，所以形状常简化为 [num_layers, batch_size, hidden_size] 。它代表了在处理完整个输入序列后，LSTM 单元的最终隐藏状态。
        cell 是 LSTM 网络最后一个时间步的细胞状态，其形状与 hidden 相同，即 [num_layers * direction_numbers, batch_size, hidden_size] 。
        out 通常用于序列到序列的任务，比如机器翻译或文本摘要，其中每个时间步的输出都是有用的。
        hidden 和 cell 通常用于需要序列最后状态的任务，比如分类或回归，其中序列的最终状态包含了整个序列的信息。

        out：这是 LSTM 网络的输出，它是一个三维张量，其形状取决于 batch_first 参数的设置。如果 batch_first=True（如你的代码所示），
        则输出的形状为 [batch, seq_len, hidden_size]，其中：batch 是批次大小，seq_len 是序列长度，hidden_size 是隐藏层的维度。
        hidden：这是 LSTM 网络最后一个时间步的隐藏状态，它是一个二维张量，形状为 [num_layers * direction_numbers, batch, hidden_size]。对于普通的 LSTM 网络，direction_numbers 通常是 1，因此形状简化为 [num_layers, batch, hidden_size]。
        cell：这是 LSTM 网络最后一个时间步的单元状态（也称为细胞状态），它的形状与 hidden 相同，也是 [num_layers * direction_numbers, batch, hidden_size]。
        '''
        out, (hidden, cell) = self.lstm(x, (self.hidden0, self.cell0))

        # 闭环
        if (self.is_close_loop == True):
            self.hidden0 = hidden
            self.cell0 = cell

        # 隐藏层的形状
        a, b, c = hidden.shape
        '''
        hidden 是 LSTM 网络的隐藏层输出，它是一个三维张量，其形状由 num_layers * direction_numbers（层数乘以方向数，对于普通的 LSTM，方向数通常是 1）、batch_size（批次大小）和 hidden_size（隐藏层的维度）组成。
        a 代表 num_layers * direction_numbers，即 LSTM 网络层数和方向数的乘积。
        b 代表 batch_size，即一次输入到网络中的样本数量。
        c 代表 hidden_size，即 LSTM 隐藏层的输出特征向量的维度。
        '''
        # 隐藏层的输出，就是全连接层的输入
        # 把隐藏层的输出hidden，向量化后：hidden.reshape(a*b,c)，送到输出层
        out = self.linear(hidden.reshape(a * b, c))  #torch.Size([8, 32])
        # 返回输出特征
        return out, (hidden, cell)


# 实例化LSTM网络
seq_length = 10
input_size = 5
hidden_size = 32
n_layers = 1
output_size = 1

lstm_model = LSTM(input_size=input_size,
                  hidden_size=hidden_size,
                  num_layers=1,
                  output_size=1,
                  batch_first=True,
                  batch_size=batch_size,
                  is_close_loop=False)
'''
LSTM(
  (lstm): LSTM(5, 32, batch_first=True)
  (linear): Linear(in_features=32, out_features=1, bias=True)
)
'''
print(lstm_model)

# 定义loss
criterion = nn.MSELoss()

# 定义优化器
Learning_rate = 0.001
optimizer = optim.Adam(lstm_model.parameters(), lr=Learning_rate)  # 使用 Adam 优化器 比课上使用的 SGD 优化器更加稳定

# 训练前的准备
n_epochs = 100
lstm_losses = []

# 开始训练
for epoch in range(n_epochs):
    for iter_, (x, label) in enumerate(train_loader):
        if (x.shape[0] != batch_size):
            continue
        pred, (h1, c1) = lstm_model(x)
        # 梯度复位
        optimizer.zero_grad()
        # 定义损失函数
        loss = criterion(pred, label)
        # 反向求导
        loss.backward(retain_graph=True)
        # 梯度迭代
        optimizer.step()
        # 记录loss
        lstm_losses.append(loss.item())

# 使用验证集进行预测
# 说明：
# 由于dataloader并不是按顺序读取的，而是随机读取
# 因此，每一次执行的结果都不一样
# 这种方式，实际上模拟了“不确定性”股票序列
data_loader = test_loader
# 存放测试序列的预测结果
predicts = []
# 存放测试序列的实际发生的结果
labels = []

for idx, (x, label) in enumerate(data_loader):
    if (x.shape[0] != batch_size):
        continue
    # 对测试集样本进行批量预测，把结果保存到predict Tensor中
    # 开环预测：即每一次序列预测与前后的序列无关。
    predict, (h, c) = lstm_model(x)
    # 把保存在tensor中的批量预测结果转换成list
    predicts.extend(predict.data.squeeze(1).tolist())
    # 把保存在tensor中的批量标签转换成list
    labels.extend(label.data.squeeze(1).tolist())

predicts = np.array(predicts)
labels = np.array(labels)
print(predicts.shape)
print(labels.shape)

# 把验证集的测试结果还原（去归一化）
predicts_unnormalized = close_min + (close_max - close_min) * predicts
labels_unnormalized = close_min + (close_max - close_min) * labels
print("shape:", predicts_unnormalized.shape)
print("正则化后的预测数据：\n", predicts)
print("")
print("正则化前的预测数据：\n", predicts_unnormalized)

# 显示预测结果与实际股价的关系
plt.plot(predicts_unnormalized,"r",label="pred") # 预测红
plt.plot(labels_unnormalized,  "b",label="real") # 真实蓝
plt.show()