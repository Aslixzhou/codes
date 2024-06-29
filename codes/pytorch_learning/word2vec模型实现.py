import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn

'''CBOW'''

text = """People who truly loved once are far more likely to love again.
Difficult circumstances serve as a textbook of life for people.
The best preparation for tomorrow is doing your best today.
The reason why a great man is great is that he resolves to be a great man.
The shortest way to do many things is to only one thing at a time.
Only they who fulfill their duties in everyday matters will fulfill them on great occasions. 
I go all out to deal with the ordinary life. 
I can stand up once again on my own.
Never underestimate your power to change yourself.""".split()

word = set(text)
word_size = len(word)

word_to_ix = {word: ix for ix, word in enumerate(word)}
ix_to_word = {ix: word for ix, word in enumerate(word)}
print(word_to_ix)
print(ix_to_word)

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

'''
word = set(text): 这里假设 text 是一个文本数据，通过 set 函数将文本中的单词去重，得到一个包含文本中所有不重复单词的集合 word。
word_size = len(word): 统计去重后的单词集合的大小，即文本中不重复单词的数量，得到 word_size。
word_to_ix = {word: ix for ix, word in enumerate(word)}: 创建一个字典 word_to_ix，将每个单词映射到一个唯一的索引。通过 enumerate(word) 遍历单词集合，为每个单词分配一个索引，构建单词到索引的映射关系。
ix_to_word = {ix: word for ix, word in enumerate(word)}: 创建一个字典 ix_to_word，将索引映射回对应的单词，构建索引到单词的映射关系。
print(word_to_ix): 打印单词到索引的映射关系，展示每个单词对应的索引。
print(ix_to_word): 打印索引到单词的映射关系，展示每个索引对应的单词。
def make_context_vector(context, word_to_ix): 定义一个函数 make_context_vector，用于根据上下文生成上下文向量。
context 是一个包含上下文单词的列表。
word_to_ix 是单词到索引的映射字典。
idxs = [word_to_ix[w] for w in context]: 将上下文中的每个单词根据 word_to_ix 映射为对应的索引，得到索引列表 idxs。
return torch.tensor(idxs, dtype=torch.long): 将索引列表转换为 PyTorch 的 Tensor 类型，并指定数据类型为 torch.long，返回上下文向量。
'''

EMDEDDING_DIM = 100  # 词向量维度

data = []
for i in range(2, len(text) - 2):
    context = [text[i - 2], text[i - 1],
               text[i + 1], text[i + 2]]
    target = text[i]
    data.append((context, target))

print(data)

class CBOW(torch.nn.Module):
    def __init__(self, word_size, embedding_dim):
        super(CBOW, self).__init__()

        self.embeddings = nn.Embedding(word_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.activation_function1 = nn.ReLU()

        self.linear2 = nn.Linear(128, word_size)
        self.activation_function2 = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1, -1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out

    def get_word_embedding(self, word):
        word = torch.tensor([word_to_ix[word]])
        return self.embeddings(word).view(1, -1)


model = CBOW(word_size, EMDEDDING_DIM)

loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 开始训练
for epoch in range(1):
    total_loss = 0
    for context, target in data:
        context_vector = make_context_vector(context, word_to_ix)
        log_probs = model(context_vector)
        total_loss += loss_function(log_probs, torch.tensor([word_to_ix[target]]))
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

# 预测
context1 = ['preparation', 'for', 'is', 'doing']
context_vector1 = make_context_vector(context1, word_to_ix)
a = model(context_vector1)
print("a: ",a)

context2 = ['People', 'who', 'loved', 'once']
context_vector2 = make_context_vector(context2, word_to_ix)
b = model(context_vector2)
print("b: ",b)

print(f'文本数据: {" ".join(text)}\n')
print(f'预测1: {context1}\n')
print(f'预测结果: {ix_to_word[torch.argmax(a[0]).item()]}')
print('\n')
print(f'预测2: {context2}\n')
print(f'预测结果: {ix_to_word[torch.argmax(b[0]).item()]}')

'''
https://blog.csdn.net/weixin_50706330/article/details/127335284?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171927733816800222824359%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171927733816800222824359&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-127335284-null-null.142^v100^pc_search_result_base1&utm_term=CBOW%E6%A8%A1%E5%9E%8B%E5%AE%9E%E7%8E%B0&spm=1018.2226.3001.4187
https://www.bilibili.com/video/BV1XH4y1D7cJ/?spm_id_from=333.788&vd_source=1396e30dc4fcabf50a79ee190b4031af
'''

# 创建一个空字典，用于存储单词和对应的词向量
word_embeddings = {}

# 遍历词表中的每个单词
for word in word_to_ix.keys():
    # 获取单词的索引
    word_index = word_to_ix[word]
    # 获取单词的词向量
    word_embedding = model.get_word_embedding(word).detach().numpy()
    # 将单词和词向量存储到字典中
    word_embeddings[word] = word_embedding
    print(word," : ",word_embeddings[word])



'''Skip-gram'''
