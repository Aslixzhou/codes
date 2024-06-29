import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import pickle

def get_datas(file = "dataset\\translate.csv",nums = None):
    all_datas = pd.read_csv(file)
    en_datas = list(all_datas["english"])
    ch_datas = list(all_datas["chinese"])
    if nums == None:
        return en_datas,ch_datas
    else:
        return en_datas[:nums],ch_datas[:nums]

class MyDataset(Dataset):
    def __init__(self,en_data,ch_data,en_word_2_index,ch_word_2_index):
        self.en_data = en_data
        self.ch_data = ch_data
        self.en_word_2_index = en_word_2_index
        self.ch_word_2_index = ch_word_2_index

    def __getitem__(self,index):
        en = self.en_data[index]
        ch = self.ch_data[index]
        en_index = [self.en_word_2_index[i] for i in en]
        ch_index = [self.ch_word_2_index[i] for i in ch]
        return en_index,ch_index

    def __len__(self):
        assert len(self.en_data) == len(self.ch_data)
        return len(self.ch_data)

    def batch_data_process(self,batch_datas): # 数据处理 填充
        global device
        en_index , ch_index = [],[]
        en_len , ch_len = [],[]
        for en,ch in batch_datas:
            en_index.append(en)
            ch_index.append(ch)
            en_len.append(len(en))
            ch_len.append(len(ch))
        max_en_len = max(en_len)
        max_ch_len = max(ch_len)
        en_index = [ i + [self.en_word_2_index["<PAD>"]] * (max_en_len - len(i))   for i in en_index]
        ch_index = [[self.ch_word_2_index["<BOS>"]] + i + [self.ch_word_2_index["<EOS>"]] + [self.ch_word_2_index["<PAD>"]] * (max_ch_len - len(i))   for i in ch_index]
        en_index = torch.tensor(en_index,device = device)
        ch_index = torch.tensor(ch_index,device = device)
        return en_index,ch_index

class Encoder(nn.Module):
    def __init__(self,encoder_embedding_num,encoder_hidden_num,en_corpus_len): # en_corpus_len词库长度 encoder_embedding_num每个embedding大小
        super().__init__()
        self.embedding = nn.Embedding(en_corpus_len,encoder_embedding_num)
        self.lstm = nn.LSTM(encoder_embedding_num,encoder_hidden_num,batch_first=True)

    def forward(self,en_index):
        en_embedding = self.embedding(en_index)
        # print("en_embedding.shape: ",en_embedding.shape) # torch.Size([3,4,50])
        encoder_output,encoder_hidden =self.lstm(en_embedding)
        # print("encoder_hidden: ",len(encoder_hidden),encoder_hidden[0].shape)
        return encoder_hidden # 最后一个隐层信息

class Decoder(nn.Module):
    def __init__(self,decoder_embedding_num,decoder_hidden_num,ch_corpus_len):
        super().__init__()
        self.embedding = nn.Embedding(ch_corpus_len,decoder_embedding_num)
        self.lstm = nn.LSTM(decoder_embedding_num,decoder_hidden_num,batch_first=True)

    def forward(self,decoder_input,hidden):
        embedding = self.embedding(decoder_input)
        decoder_output,decoder_hidden = self.lstm(embedding,hidden)
        return decoder_output,decoder_hidden


class Seq2Seq(nn.Module):
    def __init__(self,encoder_embedding_num,encoder_hidden_num,en_corpus_len,decoder_embedding_num,decoder_hidden_num,ch_corpus_len):
        super().__init__()
        self.encoder = Encoder(encoder_embedding_num,encoder_hidden_num,en_corpus_len)
        self.decoder = Decoder(decoder_embedding_num,decoder_hidden_num,ch_corpus_len)
        self.classifier = nn.Linear(decoder_hidden_num,ch_corpus_len)
        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self,en_index,ch_index):
        decoder_input = ch_index[:,:-1]
        label = ch_index[:,1:]
        encoder_hidden = self.encoder(en_index)
        decoder_output,_ = self.decoder(decoder_input,encoder_hidden)
        # print("decoder_output: ",len(decoder_output),decoder_output[0].shape)
        pre = self.classifier(decoder_output)
        # print("pre.shape: ",pre.shape) # torch.Size([3,6,3592])
        loss = self.cross_loss(pre.reshape(-1,pre.shape[-1]),label.reshape(-1))
        return loss

def translate(sentence):
    global en_word_2_index,model,device,ch_word_2_index,ch_index_2_word
    en_index = torch.tensor([[en_word_2_index[i] for i in sentence]],device=device) # 加上batch_size维度
    print("sentence: ",en_index)
    result = []
    encoder_hidden = model.encoder(en_index)
    decoder_input = torch.tensor([[ch_word_2_index["<BOS>"]]],device=device)
    decoder_hidden = encoder_hidden
    while True:
        # print("translate_decoder_input.shape: ",decoder_input.shape)
        decoder_output,decoder_hidden = model.decoder(decoder_input,decoder_hidden)
        # print("translate_decoder_output: ", len(decoder_output), decoder_output[0].shape)
        pre = model.classifier(decoder_output)
        w_index = int(torch.argmax(pre,dim=-1))
        word = ch_index_2_word[w_index]
        if word == "<EOS>" or len(result) > 50:
            break
        result.append(word)
        decoder_input = torch.tensor([[w_index]],device=device) # decoder_input
    print("译文: ","".join(result))


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with open("dataset\\ch.vec","rb") as f1:
        _, ch_word_2_index,ch_index_2_word = pickle.load(f1)

    with open("dataset\\en.vec","rb") as f2:
        _, en_word_2_index, en_index_2_word = pickle.load(f2)

    print("ch_word_2_index: ",ch_word_2_index)
    print("ch_index_2_word: ",ch_index_2_word)
    print("en_word_2_index: ",en_word_2_index)
    print("en_index_2_word: ",en_index_2_word)
    ch_corpus_len = len(ch_word_2_index)
    en_corpus_len = len(en_word_2_index)
    print(ch_corpus_len)
    print(en_corpus_len)

    en_datas,ch_datas = get_datas(nums=100)
    print("use ch_datas: ",ch_datas)
    print("use en_datas: ",en_datas)

    ch_word_2_index.update({"<PAD>": ch_corpus_len, "<BOS>": ch_corpus_len + 1, "<EOS>": ch_corpus_len + 2})
    en_word_2_index.update({"<PAD>": en_corpus_len})
    ch_index_2_word += ["<PAD>", "<BOS>", "<EOS>"]
    en_index_2_word += ["<PAD>"]
    ch_corpus_len += len(ch_word_2_index)
    en_corpus_len = len(en_word_2_index)

    batch_size = 3
    dataset = MyDataset(en_datas,ch_datas,en_word_2_index,ch_word_2_index)
    dataloader = DataLoader(dataset,batch_size,shuffle=False,collate_fn = dataset.batch_data_process)
    # for en_index,ch_index  in dataloader:
    #     print(en_index)
    #     print(ch_index)
    #     '''
    #     tensor([[29,  6, 12, 77],
    #             [29,  6, 12, 77],
    #             [57, 13,  7, 12]])
    #     tensor([[3590, 2085,    0, 3591, 3589, 3589, 3589],
    #             [3590,    4,   33,    0, 3591, 3589, 3589],
    #             [3590,    4,   91,  415,    2,    0, 3591]])
    #     '''
    #     break

    encoder_embedding_num = 50
    encoder_hidden_num = 128
    decoder_embedding_num = 107
    decoder_hidden_num = 128
    epoch = 5
    lr = 0.001
    model = Seq2Seq(encoder_embedding_num,encoder_hidden_num,en_corpus_len,decoder_embedding_num,decoder_hidden_num,ch_corpus_len).to(device)
    opt = torch.optim.Adam(model.parameters(),lr = lr)

    for e in range(epoch):
        for en_index,ch_index  in dataloader:
            loss = model(en_index,ch_index)
            loss.backward()
            opt.step()
            opt.zero_grad()

        print(f"loss:{loss:.3f}")


    while True:
        s = input("请输入英文: ")
        translate(s)


