import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings
warnings.filterwarnings('ignore')

'''
https://blog.csdn.net/m0_49963403/article/details/135309575?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171919962116800184159944%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=171919962116800184159944&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-24-135309575-null-null.142^v100^pc_search_result_base1&utm_term=rnn%E8%82%A1%E7%A5%A8%E9%A2%84%E6%B5%8B%E9%A1%B9%E7%9B%AE%E5%AE%9E%E6%88%98pyTorch&spm=1018.2226.3001.4187
https://www.bilibili.com/video/BV1Tu411R7k8/?spm_id_from=333.337.search-card.all.click&vd_source=1396e30dc4fcabf50a79ee190b4031af
'''

from tqdm import tqdm
import matplotlib.pyplot as plt
import unicodedata
import torch
import torch.nn as nn
import string
from io import open

all_letters = string.ascii_letters + '!'
n_letters = len(all_letters) + 1

name_path = './中文情绪分类/names/Japanese.txt'

names = open(name_path, encoding='utf-8').read().strip().split('\n')
names_with_endmark = []
for name in names:
    names_with_endmark.append(name + '!')
print(names_with_endmark)

Ascii_names = []  # 把names格式转为Ascii
for name in names_with_endmark:
    Ascii_names.append(''.join(letter for letter in unicodedata.normalize('NFD', name) if
                               unicodedata.category(letter) != 'Mn' and letter in all_letters))

# 上面这行代码解读：
#         1. unicodedata.normalize('NFD', name)：对输入的字符串name进行NFD（Normalization Form D）标准化。NFD将每个字符分解为其基本形式和所有可分解的组合标记。
#         2. letter for letter in ...：这是一个生成器表达式，它会遍历经过NFD标准化后的字符串name中的每一个字符letter。
#         3. if unicodedata.category(letter) != 'Mn'：检查每个字符c的Unicode类别是否不等于'Mn'。'Mn'代表"Mark, Non-Spacing"，即非-spacing组合标记，这些标记不占据自己的空间位置，而是附加在其他字符上改变其样式或语意。
#         4. ''.join(...)：将所有满足条件（非'Mn'类别）的字符连接成一个新的字符串。由于连接符是空字符串''，所以结果是一个没有分隔符的连续字符串。
# print(Ascii_names)   #这里和上面pring(names_with_endmark)输出结果看不出差别，因为只是编码方式不同

'''
N vs N RNN
'''
def input_onehot(name):
    onehot_tensor = torch.zeros(len(name), 1, n_letters)
    for i in range(len(name)):
        onehot_tensor[i][0][all_letters.find(name[i])] = 1
    return onehot_tensor


def target_onehot(name):
    onehot = []
    for i in range(1, len(name)):
        onehot.append(all_letters.find(name[i]))
    onehot.append(all_letters.find('!'))
    onehot_tensor = torch.tensor(onehot)
    return onehot_tensor

print("input: ",input_onehot('Arai'))
print("target: ",target_onehot('Arai'))

'''
用‘Abe’这个名字举例来说，input_onehot就是对应[Abe]的onehot向量（训练输入），而target_onehot就是对应[be!]的onehot向量（训练输出目标）
'''
# onehot解码的函数，用于把训练后的onehot向量转回字母：
def onehot_letter(onehot):  #onehot编码转letter
    _,letter_index = torch.topk(onehot,k=1)
    return all_letters[letter_index]


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.h0 = torch.zeros(1, self.hidden_size)
        # i2h input → hidden，hidden理解为语义
        # i2o input → output
        # o2o output→ output
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)  # 抑制过拟合
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden


rnn = RNN(n_letters, 128, n_letters)
criterion = nn.NLLLoss()  # Negative Log Likelihood Loss，即负对数似然损失。
opt = torch.optim.SGD(params=rnn.parameters(), lr=5e-4)  # 随机梯度下降优化方法
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt, T_max=100, last_epoch=-1)  # 增加余弦退火自调整学习率
epoch = 1000


def train(input_name_tensor, target_name_tensor):
    target_name_tensor.unsqueeze_(-1)
    hidden = rnn.h0  # 对h0进行初始化

    opt.zero_grad()

    name_loss = 0
    for i in range(input_name_tensor.size(0)):
        output, hidden = rnn(input_name_tensor[i], hidden)
        loss = criterion(output, target_name_tensor[i])
        name_loss += loss
    name_loss.backward()  # 对整个名字的loss进行backward

    opt.step()
    return name_loss


for e in tqdm(range(epoch)):
    total_loss = 0
    for name in Ascii_names:
        total_loss = total_loss + train(input_onehot(name), target_onehot(name))
    print(total_loss)
    plt_loss = total_loss.detach()
    plt.scatter(e, plt_loss, s=2, c='r')
    scheduler.step()
torch.save(rnn.state_dict(), 'weight/epoch=1000--initial_lr=5e-4.pth')  # 保存训练好的权重

plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


rnn_predict = RNN(n_letters, 128, n_letters)
rnn_predict.load_state_dict(state_dict=torch.load('weight/epoch=1000--initial_lr=5e-4.pth'))
rnn_predict.eval()
current_letter_onehot = input_onehot('A').squeeze(0)
current_letter = onehot_letter(current_letter_onehot)
hpre = rnn_predict.h0
full_name = ''
while current_letter != '!':  # 判断是不是该结束了
    full_name = full_name + current_letter
    predict_onehot, hcur = rnn_predict(current_letter_onehot, hpre)
    hpre = hcur
    current_letter_onehot = predict_onehot
    current_letter = onehot_letter(current_letter_onehot)
print(full_name)
