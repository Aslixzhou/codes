from sklearn.preprocessing import OneHotEncoder
import numpy as np

# one-hot编码
# 假设我们有一个包含分类数据的特征
# 这里使用一个简单的列表来表示
data = [['cat'], ['dog'], ['bird'], ['cat'], ['bird']]
# 创建一个OneHotEncoder对象
encoder = OneHotEncoder(sparse=False)
# 将数据转换为numpy数组
data_array = np.array(data)
# 对数据进行One-Hot编码
onehot_encoded = encoder.fit_transform(data_array)
# 打印One-Hot编码后的结果
print(onehot_encoded)



import torch
from pyhanlp import *
from sklearn.preprocessing import OneHotEncoder
import numpy as np

content = "虽然原始的食材便具有食物原始的风情，云初还是认为，" \
          "最美味的食物还是需要经过分割，烹调，处置，最后端上桌的食物才是最符合大唐人肠胃的食物。"
words = HanLP.segment(content)

key = []
for i in words:
    key.append(i.word)
print(key)
print(np.array(key).reshape(-1, 1))

enc = OneHotEncoder()
enc.fit(np.array(key).reshape(-1, 1))
print("虽然：", enc.transform([['虽然']]).toarray())
print("enc.categories: ", enc.categories_)
# print(enc.inverse_transform([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]))


from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

def cut_word(text):
    # ⽤结巴对中⽂字符串进⾏分词
    text = " ".join(list(jieba.cut(text)))
    return text
def text_chinese_tfidf_demo():
    # 对中⽂进⾏特征抽取
    data = ["⼀种还是⼀种今天很残酷，明天更残酷，后天很美好，但绝对⼤部分是死在明天晚上，所以每个⼈不要放弃今天。",
    "我们看到的从很远星系来的光是在⼏百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
    "如果只⽤⼀种⽅式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    # 将原始数据转换成分好词的形式
    text_list = []
    for sent in data:
        text_list.append(cut_word(sent))
    print(text_list)
    transfer = TfidfVectorizer(stop_words=['⼀种', '不会', '不要'])
    data = transfer.fit_transform(text_list)
    print("⽂本特征抽取的结果：\n", data.toarray(),data.shape)
    print("返回特征名字：\n", transfer.get_feature_names_out())
    return None


data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}]
transfer = DictVectorizer(sparse=False)
# 调⽤fit_transform
data = transfer.fit_transform(data)
print("返回的结果:\n", data)
# 打印特征名字
print("特征名字：\n", transfer.get_feature_names_out())

print("--------------文本特征提取-----------------")
data = ["life is short,i like like python", "life is too long,i dislike python"]
transfer = CountVectorizer(stop_words=[],max_features=5)
data = transfer.fit_transform(data)
print("⽂本特征抽取的结果：\n", data.toarray())
print("返回特征名字：\n", transfer.get_feature_names_out())

print("------------中文文本特征提取---------------")
text_chinese_tfidf_demo()



# 定义类别列表
labels = ['猫', '狗', '兔子', '猫', '兔子']
# 获取不同的类别
unique_labels = list(set(labels))
# 确定类别数量
num_classes = len(unique_labels)
# 创建一个空列表来存储 one-hot 编码结果
one_hot_encoded = []
# 遍历类别列表进行 one-hot 编码
for label in labels:
    one_hot = [0] * num_classes
    one_hot[unique_labels.index(label)] = 1
    one_hot_encoded.append(one_hot)
print(one_hot_encoded,'\n')

from sklearn.feature_extraction.text import CountVectorizer

data = ["Get busy living, Or get busy dying.",
        "No pains, no gains."]
transfer = CountVectorizer()
data = transfer.fit_transform(data)
print("特征名字：\n", transfer.get_feature_names_out())
print("独热编码：\n", data.toarray())


import jieba
from sklearn.feature_extraction.text import CountVectorizer

data = ["南京市长江大桥",
        "一分耕耘一分收获。"]
text_list = []
for sent in data:
        words = " ".join(list(jieba.cut(sent)))
        text_list.append(words)
print("分词：\n", text_list)
transfer = CountVectorizer()
data = transfer.fit_transform(text_list)
print("特征名字：\n", transfer.get_feature_names_out())
print("独热编码：\n", data.toarray())


from sklearn.feature_extraction.text import TfidfVectorizer
data = ["I enjoy coding.",
        "I like python.",
        "I dislike python."]
transfer = TfidfVectorizer(stop_words=['I'])
data = transfer.fit_transform(data)
print()
print("特征名字：\n", transfer.get_feature_names_out())
print("文本特征抽取结果：\n", data.toarray())



print("N-GRAM: ")
