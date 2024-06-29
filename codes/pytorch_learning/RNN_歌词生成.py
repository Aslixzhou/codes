import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import random
import zipfile

import sys

import torch
import random
import zipfile

'''
https://blog.csdn.net/qq_42589613/article/details/128530353?ops_request_misc=&request_id=&biz_id=102&utm_term=%E4%BD%BF%E7%94%A8PyTorch%E5%BC%80%E5%8F%91%E6%AD%8C%E8%AF%8D%E7%94%9F%E6%88%90%E5%99%A8%E6%A8%A1%E5%9E%8B&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-4-128530353.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187
'''

'''
corpus_indices【数据集字符对应的索引】
char_to_idx【词典不同字符对应索引】
idx_to_char【词典索引对应不同字符】
vocab_size【数据集不同字符数量】
'''

with zipfile.ZipFile('./dataset/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
print(corpus_chars[:100])
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:10000]#取前10000个字符进行后续模型训练
print(corpus_chars)

# idx_to_char = list(set(corpus_chars))
# char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
# vocab_size = len(char_to_idx)
# # 将训练数据集中每个字符转化为索引
# corpus_indices = [char_to_idx[char] for char in corpus_chars]
# print(corpus_indices) # 数据对应索引 [342, 399, 272, 611, 726,...
# print(char_to_idx) # 词典+索引 {'啃': 0, '断': 1, '著': 2, '牵': 3,...
# print(idx_to_char) # 词典+索引 ['啃', '断', '著', '牵', '脚', '视',...
# print(vocab_size) # 词典大小 1027
# # 打印前20个字符及其对应的索引
# print('chars: ', ''.join([idx_to_char[idx] for idx in corpus_indices[:20]]))
# print('indices: ', corpus_indices[:20])
#
# '''
# 样本序列：“想”“要”“有”“直”“升” ----> 标签序列：“要”“有”“直”“升”“机”
# '''
#
# # one-hot
#
# def one_hot(x, n_class, dtype=torch.float32):
#     # X shape: (batch), output shape: (batch, n_class)
#     x = x.long()
#     res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
#     res.scatter_(1, x.view(-1, 1), 1)
#     return res
#
# def to_onehot(X, n_class):
#     # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
#     return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]
#
#
# # 相邻采样
# def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
#     data_len = len(corpus_indices)
#     batch_len = data_len // batch_size
#     indices = corpus_indices[0: batch_size*batch_len].view(batch_size, batch_len)
#     epoch_size = (batch_len - 1) // num_steps
#     for i in range(epoch_size):
#         i = i * num_steps
#         X = indices[:, i: i + num_steps]
#         Y = indices[:, i + 1: i + num_steps + 1]
#         yield X, Y
#
# for X, Y in data_iter_consecutive(corpus_indices, batch_size=5, num_steps=6):
#     print('X: ', X, '\nY:', Y, '\n')
#     inputs = to_onehot(X, vocab_size)
#     outputs = to_onehot(Y,vocab_size)
#     print(len(inputs), inputs[0].shape) # 6 torch.Size([5, 1027]) num_steps batch_size vocab_size
#     print(len(outputs), outputs[0].shape) # 6 torch.Size([5, 1027]) 时间t 输入num_t_step


'''
X:  tensor([[ 228.,  304.,  674.,  455.,  128., 1026.],
        [ 292.,  263.,  880.,  405.,  518.,  790.],
        [ 848.,  495.,  411.,  404.,  438.,  880.],
        [ 411.,  925.,  627.,  880.,  486.,  674.],
        [ 302.,  842.,  974.,  515.,  866.,  969.]]) 
Y: tensor([[ 304.,  674.,  455.,  128., 1026.,  880.],
        [ 263.,  880.,  405.,  518.,  790.,   47.],
        [ 495.,  411.,  404.,  438.,  880.,   19.],
        [ 925.,  627.,  880.,  486.,  674.,  747.],
        [ 842.,  974.,  515.,  866.,  969.,  880.]]) 
'''


def create_dataset(lyrics, seq_length):
    dataX = []
    dataY = []
    chars = list(set(lyrics))
    input_size = len(set(lyrics))
    output_size = len(set(lyrics))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}

    for i in range(0, len(lyrics) - seq_length):
        seq_in = lyrics[i:i + seq_length]
        seq_out = lyrics[i + seq_length]
        dataX.append([char_to_idx[ch] for ch in seq_in])
        dataY.append(char_to_idx[seq_out])

    return np.array(dataX), np.array(dataY), char_to_idx, input_size, output_size

dataX,dataY,char_to_idx,input_size,output_size = create_dataset(corpus_chars,5)
print(dataX) # [798 266 114  19 986] 想要有直升
print(dataY) # [798 266 114  19 986] --> 994 机
# dataY = dataY[:6000]
# dataX = dataX[:6000]
print(char_to_idx) # {'名': 0, '映': 1, '管': 2, '捏': 3, '禁': 4,...
dataX = torch.from_numpy(dataX).long()
dataY = torch.from_numpy(dataY).long()

seq_length = 10
hidden_size = 128
num_epochs = 2
batch_size = 1

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output[:, -1, :])  # 只取最后一个时间步的输出
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


dataset = torch.utils.data.TensorDataset(dataX, dataY)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = RNNModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 将模型、损失函数和优化器移动到合适的设备上，例如 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = criterion.to(device)

# 训练模型
for epoch in range(num_epochs):
    # 初始化隐藏状态
    model.train()
    hidden = model.init_hidden(batch_size)
    hidden = hidden.to(device)  # 将隐藏状态移到 GPU
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        loss = 0
        # inputs, targets = inputs.to(device), targets.to(torch.long).to(device)  # 确保 targets 是 long 类型
        outputs,hidden = model(inputs,hidden)
        hidden = hidden.data
        # print("outputs", outputs)
        # print(outputs.argmax(dim=1))
        # print("targets.view(-1): ", targets.view(-1))
        loss += criterion(outputs, targets.view(-1))  # 计算损失
        # 每100个批次打印一次损失值
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(data_loader)}, Loss: {loss.item()}')
        optimizer.zero_grad()
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数
    print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

print('Training complete')


model.eval()
hidden = model.init_hidden(1)
start_char = '我'
generated_lyrics = [start_char]

with torch.no_grad():
    input_char = torch.tensor([[char_to_idx[start_char]]], dtype=torch.long)
    while len(generated_lyrics) < 100:
        output, hidden = model(input_char, hidden)
        _, predicted = torch.max(output, 1)
        next_char = list(char_to_idx.keys())[list(char_to_idx.values()).index(predicted.item())]
        generated_lyrics.append(next_char)
        input_char = torch.tensor([[predicted.item()]], dtype=torch.long)

generated_lyrics = ''.join(generated_lyrics)
print("Generated Lyrics:")
print(generated_lyrics)




# import d2lzh_pytorch as d2l
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# (corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics('./RNN-JayZhou/jaychou_lyrics.txt.zip')
#
# # 定义模型
# num_hiddens = 256
# # rnn_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
# rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
#
#
# class RNNModel(nn.Module):
#     def __init__(self, rnn_layer, vocab_size):
#         super(RNNModel, self).__init__()
#         self.rnn = rnn_layer
#         # rnn_layer.bidirectional，如果是双向循环网络则为2，单向则为1
#         self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
#         self.vocab_size = vocab_size
#         self.dense = nn.Linear(self.hidden_size, vocab_size)
#         self.state = None
#
#     def forward(self, inputs, state):  # inputs: (batch, seq_len)
#         # 获取one-hot向量表示
#         X = d2l.to_onehot(inputs, self.vocab_size)  # X是个list
#         Y, self.state = self.rnn(torch.stack(X), state)
#         # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
#         # 形状为(num_steps * batch_size, vocab_size)
#         output = self.dense(Y.view(-1, Y.shape[-1]))
#         return output, self.state
#
#
# # 定义预测函数
# def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char,
#                         char_to_idx):
#     state = None
#     output = [char_to_idx[prefix[0]]]  # output会记录prefix加上输出
#     for t in range(num_chars + len(prefix) - 1):
#         X = torch.tensor([output[-1]], device=device).view(1, 1)
#         if state is not None:
#             if isinstance(state, tuple):  # LSTM, state:(h, c)
#                 state = (state[0].to(device), state[1].to(device))
#             else:
#                 state = state.to(device)
#
#         (Y, state) = model(X, state)
#         if t < len(prefix) - 1:
#             output.append(char_to_idx[prefix[t + 1]])
#         else:
#             output.append(int(Y.argmax(dim=1).item()))
#     return ''.join([idx_to_char[i] for i in output])
#
#
# # 定义训练函数
# def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
#                                   corpus_indices, idx_to_char, char_to_idx,
#                                   num_epochs, num_steps, lr, clipping_theta,
#                                   batch_size, pred_period, pred_len, prefixes):
#     loss = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     model.to(device)
#     state = None
#     for epoch in range(num_epochs):
#         l_sum, n, start = 0.0, 0, time.time()
#         data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size, num_steps, device)  # 相邻采样
#         for X, Y in data_iter:
#             if state is not None:
#                 # 使用detach函数从计算图分离隐藏状态, 这是为了
#                 # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
#                 if isinstance(state, tuple):  # LSTM, state:(h, c)
#                     state = (state[0].detach(), state[1].detach())
#                 else:
#                     state = state.detach()
#
#             (output, state) = model(X, state)  # output: 形状为(num_steps * batch_size, vocab_size)
#
#             # Y的形状是(batch_size, num_steps)，转置后再变成长度为
#             # batch_size * num_steps 的向量，这样跟输出的行一一对应
#             y = torch.transpose(Y, 0, 1).contiguous().view(-1)
#             # y.long()表示向下取整
#             l = loss(output, y.long())
#             '''
#             y = torch.transpose(Y, 0, 1).contiguous().view(-1)作用
#             Y = tensor([[ 7.,  8.,  9., 10., 11., 12.],
#         		[22., 23., 24., 25., 26., 27.]])
#         	则y = torch.transpose(Y, 0, 1).contiguous().view(-1)结果为：
#         	tensor([ 7., 22.,  8., 23.,  9., 24., 10., 25., 11., 26., 12., 27.])
#             '''
#             optimizer.zero_grad()
#             l.backward()
#             # 梯度裁剪
#             d2l.grad_clipping(model.parameters(), clipping_theta, device)
#             optimizer.step()
#             l_sum += l.item() * y.shape[0]
#             n += y.shape[0]
#
#         try:
#             perplexity = math.exp(l_sum / n)
#         except OverflowError:
#             perplexity = float('inf')
#         if (epoch + 1) % pred_period == 0:
#             print('epoch %d, perplexity %f, time %.2f sec' % (
#                 epoch + 1, perplexity, time.time() - start))
#             for prefix in prefixes:
#                 print(' -', predict_rnn_pytorch(
#                     prefix, pred_len, model, vocab_size, device, idx_to_char,
#                     char_to_idx))
#
#
# # 训练与预测
# num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2  # 注意这里的学习率设置
# pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
# model = RNNModel(rnn_layer, vocab_size).to(device)
# train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
#                               corpus_indices, idx_to_char, char_to_idx,
#                               num_epochs, num_steps, lr, clipping_theta,
#                               batch_size, pred_period, pred_len, prefixes)
#
#
