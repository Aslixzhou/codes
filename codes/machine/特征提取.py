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
    print("⽂本特征抽取的结果：\n", data.toarray())
    print("返回特征名字：\n", transfer.get_feature_names_out())
    return None

if __name__ == '__main__':
    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}]
    transfer = DictVectorizer(sparse=False)
    # 调⽤fit_transform
    data = transfer.fit_transform(data)
    print("返回的结果:\n", data)
    # 打印特征名字
    print("特征名字：\n", transfer.get_feature_names_out())

    print("--------------文本特征提取-----------------")
    data = ["life is short,i like like python", "life is too long,i dislike python"]
    transfer = CountVectorizer(stop_words=[])
    data = transfer.fit_transform(data)
    print("⽂本特征抽取的结果：\n", data.toarray())
    print("返回特征名字：\n", transfer.get_feature_names_out())

    print("------------中文文本特征提取---------------")
    text_chinese_tfidf_demo()

