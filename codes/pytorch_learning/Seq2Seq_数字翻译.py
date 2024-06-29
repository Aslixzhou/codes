import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.utils.data as Data
import random
import pandas as pd
from torchtext import data
from torchtext.vocab import build_vocab_from_iterator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

numStrDict = {'0': '零', '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八',
              '9': '九'}

def numToStr(number):
    tempList = [i for i in str(number)]
    strList = [numStrDict[i] for i in tempList]
    return tempList, strList

numList = list()
strList = list()
for i in range(51000):
    tempNum = random.randint(10000000, 999999999999)
    tempNum, tempStr = numToStr(tempNum)
    numList.append(" ".join(tempNum))
    strList.append(" ".join(tempStr))

df = pd.DataFrame({
    'number': numList,
    'string': strList
})
print(df,df.shape)


def yield_tokens(data_iter): # 分词器生成
    tokenizer = data.get_tokenizer("basic_english")
    for test in data_iter:
        yield tokenizer(test)


numVocab = build_vocab_from_iterator(yield_tokens(df.number), min_freq=1, specials=['<PAD>', '<SOS>', '<EOS>'])
numVocab.set_default_index(numVocab['<PAD>'])
strVocab = build_vocab_from_iterator(yield_tokens(df.string), min_freq=1, specials=['<PAD>', '<SOS>', '<EOS>'])
strVocab.set_default_index(strVocab['<PAD>'])

numVocab_size = numVocab.__len__() # 13
strVocab_size = strVocab.__len__() # 13
print(numVocab.get_stoi()) # {'<EOS>': 2, '<PAD>': 0, '9': 7, '1': 5, '<SOS>': 1, '8': 3, '6': 4, '5': 6, '7': 8, '4': 9, '2': 10, '3': 11, '0': 12}
print(strVocab.get_stoi()) # {'六': 4, '<EOS>': 2, '<PAD>': 0, '八': 3, '<SOS>': 1, '一': 5, '五': 6, '九': 7, '七': 8, '四': 9, '二': 10, '三': 11, '零': 12}

def word2index(vocab, word):
    return vocab[word]

def index2word(vocab, index):
    return vocab.lookup_token(index)

def wordList2indexList(vocab, wordList):
    return vocab.lookup_indices(wordList)

def indexList2wordList(vocab, indexList):
    return vocab.lookup_tokens(indexList)

numMaxLength = max(len(i.split(" ")) for i in df["number"])
strMaxLength = max(len(i.split(" ")) for i in df["string"])
maxLength = max(numMaxLength, strMaxLength) + 2 # <SOS><EOS>

def sentenceDeal(vocab, wordList, maxLength):
    wordList.insert(0, "<SOS>") # head
    wordList.append("<EOS>") # tail
    for i in range(maxLength - len(wordList)):
        wordList.append("<PAD>")
    return wordList2indexList(vocab, wordList)

tokenizer = data.get_tokenizer("basic_english")
allNumberList = list()
for index in range(len(df)):
    allNumberList.append(sentenceDeal(numVocab, tokenizer(df.number[index]), maxLength))
allStringList = list()
for index in range(len(df)):
    allStringList.append(sentenceDeal(strVocab, tokenizer(df.string[index]), maxLength))

df["numberCode"] = allNumberList
df["stringCode"] = allStringList
# print(df,df.shape)
# print(df["numberCode"])
# print(df["stringCode"])


# 训练集
class trainDataset(Data.Dataset):
    def __len__(self):
        return len(df[:50000])
    def __getitem__(self, i):
        x = df[:50000].numberCode.tolist()[i]
        y = df[:50000].stringCode.tolist()[i]
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        return x, y
# 测试集
class testDataset(Data.Dataset):
    def __len__(self):
        return len(df[50000:])
    def __getitem__(self, i):
        x = df[50000:].numberCode.tolist()[i]
        y = df[50000:].stringCode.tolist()[i]
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        return x, y

# 训练集
traindataset = trainDataset()
print(traindataset.__getitem__(0),len(traindataset))
testdataset = testDataset()
print(testdataset.__getitem__(0),len(testdataset))


trainLoader = Data.DataLoader(dataset=traindataset,
                              batch_size=64,
                              shuffle=True,
                              drop_last=True)
testLoader = Data.DataLoader(dataset=testdataset,
                              batch_size=64,
                              shuffle=True,
                              drop_last=True)
print(len(trainLoader), next(iter(trainLoader)))



class Encoder(nn.Module):
    def __init__(self, encoder_vocab_size, encoder_embedding_dim, encoder_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=encoder_vocab_size, embedding_dim=encoder_embedding_dim)
        self.gru = nn.GRU(input_size=encoder_embedding_dim, hidden_size=encoder_hidden_size, batch_first=True)
    def forward(self, x):
        x = self.embedding(x)
        _, encoder_hidden = self.gru(x)
        return encoder_hidden

class Decoder(nn.Module):
    def __init__(self, decoder_vocab_size, decoder_embedding_dim, decoder_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=decoder_vocab_size, embedding_dim=decoder_embedding_dim)
        self.gru = nn.GRU(input_size=decoder_embedding_dim, hidden_size=decoder_hidden_size, batch_first=True)
    def forward(self, x, hidden):
        x = self.embedding(x)
        decoder_output, decoder_hidden = self.gru(x, hidden)
        return decoder_output, decoder_hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder_vocab_size, encoder_embedding_dim, encoder_hidden_size,
                 decoder_vocab_size, decoder_embedding_dim, decoder_hidden_size):
        super().__init__()
        self.encoder = Encoder(encoder_vocab_size, encoder_embedding_dim, encoder_hidden_size)
        self.decoder = Decoder(decoder_vocab_size, decoder_embedding_dim, decoder_hidden_size)
        self.fc = nn.Linear(decoder_hidden_size, decoder_vocab_size)
    def forward(self, encoder_x, decoder_x):
        decoder_input = decoder_x[:, :-1] # 去掉最后一个
        encoder_hidden = self.encoder(encoder_x)
        decoder_output, _ = self.decoder(decoder_input, encoder_hidden)
        out = self.fc(decoder_output)
        return out

encoder_embedding_dim = 32
encoder_hidden_size = 512
decoder_embedding_dim = 32
decoder_hidden_size = 512
model = Seq2Seq(numVocab_size, encoder_embedding_dim, encoder_hidden_size,
                strVocab_size, decoder_embedding_dim, decoder_hidden_size).to(device)
print(model)


def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fun = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1):
        for step, (x, y) in enumerate(trainLoader):
            out = model(x.to(device), y.to(device))
            y = y[:, 1:] # 用生成<SOS>之后的计算损失
            loss = loss_fun(out.reshape(-1, out.shape[-1]).to("cpu"), y.reshape(-1).to("cpu"))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 200 == 0:
                acc = (out.to("cpu").argmax(dim=2) == y).sum().item() / (len(y) * (maxLength - 1))
                print(f"epoch：{epoch + 1}，step：{step + 1}， 损失：{loss.item()}，准确率：{acc}")
    torch.save(model, "./seq2seq_数字翻译.model")
train()

@torch.no_grad()
def test(dataDf, index, strVocab):
    model = torch.load("./seq2seq_数字翻译.model").to("cpu")
    model.eval()
    result = list()
    x = torch.Tensor(dataDf.iloc[index]["numberCode"]).to(dtype=torch.long)
    x = torch.unsqueeze(x, dim=0)
    decoder_input = torch.unsqueeze(torch.Tensor([word2index(strVocab, "<SOS>")]), dim=0).to(dtype=torch.long)
    encoder_hidden = model.encoder(x)
    decoder_hidden = encoder_hidden
    while True:
        decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
        out = model.fc(decoder_output)
        word = index2word(strVocab, torch.argmax(out[0][0]))
        result.append(word)
        if word == '<EOS>' or len(result) >= maxLength - 2:
            break
        decoder_input = torch.unsqueeze(torch.Tensor([torch.argmax(out[0][0])]), dim=0).to(dtype=torch.long)
    return "".join(result)

testIndex = 50096
preOut = test(df, testIndex, strVocab).replace("<SOS>", "").replace("<EOS>", "")
print(f"Number：{df.number[testIndex]}\nString：{df.string[testIndex]}\n预测：{preOut}")

