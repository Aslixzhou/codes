import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optimizer
import torch.utils.data as Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.FloatTensor

sentences = [
    "i am a student ",
    "i am a boy ",
    "studying is not a easy work ",
    "japanese are bad guys ",
    "we need peace ",
    "computer version is increasingly popular ",
    "the word will get better and better "
]
sentence_list = "".join(sentences).split()  # 语料库---有重复单词
vocab = list(set(sentence_list))  # 词汇表---没有重复单词
word2idx = {w: i for i, w in enumerate(vocab)}  # 词汇表生成的字典，包含了单词和索引的键值对
vocab_size = len(vocab)
print(vocab)
print(word2idx)

w_size = 2  # 上下文单词窗口大小
batch_size = 8
word_dim = 2  # 词向量维度
skip_grams = []
for word_idx in range(w_size, len(sentence_list) - w_size):  # word_idx---是原语料库中的词索引
    center_word_vocab_idx = word2idx[sentence_list[word_idx]]  # 中心词在词汇表里的索引
    context_word_idx = list(range(word_idx - w_size, word_idx)) + list(
        range(word_idx + 1, word_idx + w_size + 1))  # 上下文词在语料库里的索引
    context_word_vocab_idx = [word2idx[sentence_list[i]] for i in context_word_idx]  # 上下文词在词汇表里的索引

    for idx in context_word_vocab_idx:
        skip_grams.append([center_word_vocab_idx, idx])  # 加入进来的都是索引值

print(skip_grams)

def make_data(skip_grams):
    input_data = []
    output_data = []
    for center, context in skip_grams:
        input_data.append(np.eye(vocab_size)[center])
        output_data.append(context)
    return input_data, output_data


input_data, output_data = make_data(skip_grams)
input_data, output_data = torch.Tensor(input_data), torch.LongTensor(output_data)
print(input_data)
print(output_data)
dataset = Data.TensorDataset(input_data, output_data)
loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        self.W = nn.Parameter(torch.randn(vocab_size, w_size).type(dtype))
        self.V = nn.Parameter(torch.randn(w_size, vocab_size).type(dtype))

    def forward(self, X):
        hidden = torch.mm(X, self.W)
        output = torch.mm(hidden, self.V)
        return output


model = Word2Vec().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optim = optimizer.Adam(model.parameters(), lr=1e-3)

for epoch in range(2):
    for i, (batch_x, batch_y) in enumerate(loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
        if (epoch + 1) % 1000 == 0:
            print(epoch + 1, i, loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()

for i, label in enumerate(vocab):
    W, WT = model.parameters()
    x, y = float(W[i][0]), float(W[i][1])
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()

'''
https://blog.csdn.net/tengyi45/article/details/133715412?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-133715412-blog-121733726.235^v43^pc_blog_bottom_relevance_base2&spm=1001.2101.3001.4242.2&utm_relevant_index=4
https://www.bilibili.com/video/BV14z4y19777/?spm_id_from=333.337.search-card.all.click&vd_source=65c8c97304c0e780ab98be4ad24d1fde
'''

words = list(word2idx.keys())  # 获取所有单词
for word in words:
    word_index = word2idx[word]
    word_vector = model.W[word_index]  # 获取对应单词的词向量
    print(f"单词: {word}, 词向量: {word_vector}")