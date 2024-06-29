from transformers import AutoTokenizer

#加载字典和分词工具,这里的checkpoint一般和预训练模型保持一致,两者是成对使用的
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

#试编码一句话,所有一句编码后都是等长的
out = tokenizer(text='测试语句',
                padding='max_length',
                max_length=8,
                return_tensors='pt',
                truncation=True)

#查看编码结果
for k, v in out.items():
    print(k, v, v.shape, v.dtype)

#反编码
print(tokenizer.decode(out['input_ids'][0]))
from datasets import load_dataset, load_from_disk
#加载离线数据集
dataset = load_from_disk('./dataset/ChnSentiCorp')
print(dataset, dataset['train'][0])


#添加取数据时的回调函数,可以在这里完成数据的处理工作
def f(data):
    #编码句子
    result = tokenizer(text=data['text'],
                       padding='max_length',
                       max_length=200,
                       return_tensors='pt',
                       truncation=True)

    #label字段不能落下,添加到返回结果里
    result['label'] = data['label']
    return result
dataset = dataset.with_transform(f, output_all_columns=False)
print(dataset['train'][0])


import torch
loader = torch.utils.data.DataLoader(dataset=dataset['train'],
                                     batch_size=8,
                                     shuffle=True,
                                     drop_last=True)
for k, v in next(iter(loader)).items():
    print(k, v.shape, v.dtype)
print(len(loader))


from transformers import AutoModelForSequenceClassification

#手动定义模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #加载预训练模型,自动添加分类头,可以直接使用该模型训练
        #离线加载
        pretrained = AutoModelForSequenceClassification.from_pretrained(
            './pre_model/huggingface_nlp.model', num_labels=2)
        #锁定参数,不训练
        for param in pretrained.parameters():
            param.requires_grad_(False)
        #只需要特征抽取部分
        self.pretrained = pretrained.bert
        self.pretrained.eval()
        #线性输出层,这部分是要重新训练的
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, data):
        #调用预训练模型抽取参数,因为预训练模型是不训练的,所以这里不需要计算梯度
        with torch.no_grad():
            #[8, 200, 768]
            out = self.pretrained(
                input_ids=data['input_ids'],
                token_type_ids=data['token_type_ids'],
                attention_mask=data['attention_mask']).last_hidden_state
        #只要取第0个字符的特征计算分类结果即可,这和BERT模型本身的训练方式有关系
        #[8, 200, 768] -> [8, 768]
        out = out[:, 0]

        #计算线性输出
        #[8, 768] -> [8, 2]
        return self.fc(out)

model = Model()
#试算
print(model({
    'input_ids': torch.ones(8, 200).long(),
    'token_type_ids': torch.ones(8, 200).long(),
    'attention_mask': torch.ones(8, 200).long(),
    'label': torch.ones(8).long(),
}).shape)


#训练
def train():
    #注意这里的参数列表,只包括要训练的参数即可
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    loss_fun = torch.nn.CrossEntropyLoss()
    model.fc.train()
    #定义计算设备,优先使用gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print('device=', device)
    for i, data in enumerate(loader):
        #如果使用gpu,数据要搬运到显存里
        for k in data.keys():
            data[k] = data[k].to(device)
        out = model(data)
        loss = loss_fun(out, data['label'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 200 == 0:
            acc = (out.argmax(dim=1) == data['label']).sum().item() / len(
                data['label'])
            print(i, loss.item(), acc)
    #保存模型,只保存训练的部分即可
    torch.save(model.fc.to('cpu'), 'model/9.model')
train()

#测试
@torch.no_grad()
def test():
    #加载保存的模型
    model.fc = torch.load('model/9.model')
    model.fc.eval()
    #加载测试数据集,共10000条数据
    loader_test = torch.utils.data.DataLoader(dataset=dataset['test'],
                                              batch_size=8,
                                              shuffle=True,
                                              drop_last=True)
    correct = 0
    total = 0
    for i in range(100):
        data = next(iter(loader_test))
        #这里因为数据量不大,使用cpu计算就可以了
        out = model(data).argmax(dim=1)
        correct += (out == data['label']).sum().item()
        total += len(data['label'])
    print(correct / total)
test()