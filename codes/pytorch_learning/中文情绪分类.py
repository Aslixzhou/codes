import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from sklearn.model_selection import train_test_split
import re
import jieba
from collections import Counter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据路径
dataPath = "./中文情绪分类/chineseComment/"
# 路径拼接
goodFile = dataPath + "good.txt"
badFile = dataPath + "bad.txt"

# 过滤标点符号 函数
def filter_punc(sentence):
    sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\'“”《》?“]+|[+——！，。？、~@#￥%……&*（）：]+", "", sentence)
    return (sentence)

sentence = "你好，哈哈哈哈哈~"
print(filter_punc(sentence)) # 你好哈哈哈哈哈


def prepareData(good_file, bad_file, is_filter=True):
    all_words = list()
    pos_sentences = list()
    neg_sentences = list()
    with open(good_file, 'r', encoding='utf-8') as f_goog:
        for index, line in enumerate(f_goog):
            if is_filter:
                line = filter_punc(line)
            words = jieba.lcut(line)
            if len(words) > 0:
                all_words += words
                pos_sentences.append(words)
    print(f"{good_file}包含{index + 1}条数据, {len(all_words)}个词语。")
    count = len(all_words)
    with open(bad_file, 'r', encoding='utf-8') as f_bad:
        for index, line in enumerate(f_bad):
            if is_filter:
                line = filter_punc(line)
            words = jieba.lcut(line)
            if len(words) > 0:
                all_words += words
                neg_sentences.append(words)
    print(f"{bad_file}包含{index + 1}条数据, {len(all_words) - count}个词语。")
    count = len(all_words)

    diction = dict()
    cnt = Counter(all_words)
    for word, freq in cnt.items():
        diction[word] = [len(diction), freq]
    print("字典大小：", len(diction))
    return pos_sentences, neg_sentences, diction

pos_sentences, neg_sentences, diction = prepareData(goodFile, badFile, True)
# print("pos_sentences: ",pos_sentences)
# print("neg_sentences: ",neg_sentences)
# print("diction: ",diction)
st = sorted([(v[1], w) for w, v in diction.items()])
print(st)

# word2index
def word2index(word, diction):
    if word in diction:
        value = diction[word][0] # 索引
    else:
        value = -1
    return value

# index2word
def index2word(index, diction):
    for w, v in diction.items():
        if v[0] == index:
            return w
    return None


def sentence2vec(sentence, diction):
    vector = np.zeros(len(diction))
    for l in sentence:
        vector[l] += 1
    return (1.0 * vector / len(sentence))



dataset = list()
labels = list()
sentences = list()
for sentence in pos_sentences:
    new_sentence = list()
    for pos_word in sentence:
        if pos_word in diction:
            new_sentence.append(word2index(pos_word, diction))
    dataset.append(sentence2vec(new_sentence, diction))
    labels.append(0)
    sentences.append(sentence)
for sentence in neg_sentences:
    new_sentence = list()
    for neg_word in sentence:
        if neg_word in diction:
            new_sentence.append(word2index(neg_word, diction))
    dataset.append(sentence2vec(new_sentence, diction))
    labels.append(1)
    sentences.append(sentence)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.1, random_state=56)
# 转换为张量
train_xt = torch.Tensor(X_train).to(dtype=torch.float32)
train_yt = torch.Tensor(y_train).to(dtype=torch.long)
test_xt = torch.Tensor(X_test).to(dtype=torch.float32)
test_yt = torch.Tensor(y_test).to(dtype=torch.long)
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
            nn.Linear(len(diction), 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.fc(x)

model = Model().to(device)
print(model)


def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fun = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1):
        for step, (x, y) in enumerate(train_data):
            out = model(torch.unsqueeze(x,dim=0).to(device))
            loss = loss_fun(torch.squeeze(out.to("cpu")), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if epoch % 1 == 0:
            print(f"epoch:{epoch}, 损失：{loss.item()}")
    torch.save(model, "./中文情绪分类.model")
train()

@torch.no_grad()
def test(test_xt):
    model = torch.load("./中文情绪分类.model").to("cpu")
    model.eval()
    out = model(test_xt)
    return out

predit = test(test_xt)
countTrue = 0
for i in range(len(test_yt)):
    if torch.argmax(predit[i]) == test_yt[i]:
        countTrue += 1
print("准确率：" + str(countTrue / len(test_yt)))