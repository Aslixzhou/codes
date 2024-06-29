import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings
warnings.filterwarnings('ignore')

import jieba
data_path = "./文本情感分类/weibo_senti_100k.csv"
data_stop_path = "./文本情感分类/hit_stopwords.txt"
with open(data_path, 'r', encoding='utf-8') as file:
    data_list = file.readlines()[1:]
with open(data_path, 'r', encoding='utf-8') as file:
    stops_word = file.readlines()
stops_word = [line.strip() for line in stops_word]
stops_word.append(" ")
stops_word.append(",")
stops_word.append("\n")

min_seq = 1
top_n = 1000
UNK = "<UNK>"
PAD = "<PAD>"
voc_dict = {}
data = []
max_len_seq = 0

for item in data_list[:1000]: # for item in data_list:
    label = item[0]
    content = item[2:].strip()
    seg_list = jieba.cut(content,cut_all=False)
    seg_res = []
    for seg_item in seg_list:
        if seg_item in stops_word:
            continue
        seg_res.append(seg_item)
        if seg_item in voc_dict.keys():
            voc_dict[seg_item] = voc_dict[seg_item] + 1
        else:
            voc_dict[seg_item] = 1
    if len(seg_res) > max_len_seq:
        max_len_seq = len(seg_res)
    # print(content)
    # print(seg_res)
    data.append([label,seg_res])

voc_list = sorted([_ for _ in voc_dict.items() if _[1] > min_seq],
                  key=lambda x : x[1],
                  reverse=True)[:top_n]
voc_dict = {word_count[0]:idx for idx,word_count in enumerate(voc_list)}
voc_dict.update({UNK:len(voc_dict),PAD:len(voc_dict)+1})
print(voc_dict)

ff = open("dict.txt","w")
for item in voc_dict.keys():
    ff.writelines("{},{}\n".format(item,voc_dict[item]))
ff.close()


from torch.utils.data import Dataset,DataLoader
import numpy as np

def read_dict(voc_dict_path):
    voc_dict = {}
    dict_list = open(voc_dict_path).readlines()
    for item in dict_list:
        item = item.split(",")
        # print("[0]:",item[0],"[1]: ",item[1])
        voc_dict[item[0]] = int(item[1].strip())
    return voc_dict

class text_Cls(Dataset):
    def __init__(self,voc_dict_path,data,max_len_seq):
        self.voc_dict = read_dict(voc_dict_path)
        self.data = data
        self.max_len_seq = max_len_seq
        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        data = self.data[item]
        label = int(data[0])
        word_list = data[1]
        input_idx = []
        for word in word_list:
            if word in self.voc_dict.keys():
                input_idx.append(self.voc_dict[word])
            else:
                input_idx.append(self.voc_dict["<UNK>"])
        if len(input_idx) < self.max_len_seq:
            input_idx += [self.voc_dict["<PAD>"] for _ in range(self.max_len_seq - len(input_idx))]
        data = np.array(input_idx)
        return label,data


def data_loader(data,max_len_seq):
    voc_dict_path = "dict.txt"
    dataset = text_Cls(voc_dict_path=voc_dict_path,data=data,max_len_seq=max_len_seq)
    return DataLoader(dataset,shuffle=True,batch_size=10)


train_loader = data_loader(data,max_len_seq)
for i,batch in enumerate(train_loader):
    print(batch)