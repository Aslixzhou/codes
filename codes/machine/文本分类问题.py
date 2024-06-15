from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF

if __name__ == '__main__':
    print("--------------文本特征提取-----------------")
    data = ["life is short,i like like python", "life is too long,i dislike python"]
    transfer = CountVectorizer(stop_words=[])
    data = transfer.fit_transform(data)
    print("⽂本特征抽取的结果：\n", data.toarray())
    print("返回特征名字：\n", transfer.get_feature_names_out())

    print("--------------传统单词向量空间文本分类问题-----------------")
    # 加载垃圾邮件数据集，这里使用的是20个新闻组数据集中的垃圾邮件子集
    data = fetch_20newsgroups(subset='all',
                              categories=['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med'],
                              shuffle=True, random_state=42)

    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    # 使用TF-IDF向量化器将文本转换为特征向量
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train) # 将数据集转换成特征向量
    X_test = vectorizer.transform(X_test)
    print("Number of features:", len(vectorizer.vocabulary_))
    print("X_train: \n", X_train)
    '''
    在这个表示中，(0, 7124) 表示一个稀疏矩阵中的一个元素，它的含义是：
        第一个数字 0 表示这个样本在 X_train 中的索引，即第 0 个样本。
        第二个数字 7124 表示特征的索引，即在 TF-IDF 特征向量中，这个元素对应的特征在词汇表中的索引为 7124。
        而 0.03913747061190574 则表示这个样本中对应于索引 7124 的特征的 TF-IDF 值，即该词在当前文档中的重要性程度。
    综合起来，(0, 7124)	0.03913747061190574 表示第一个样本中，词汇表中索引为 7124 的词的 TF-IDF 值为 0.03913747061190574
    '''
    print("y_train: \n", y_train)
    print("X_test: \n", X_test)
    print("y_test: \n", y_test)
    # 初始化感知机模型
    perceptron = Perceptron()
    # 拟合模型
    perceptron.fit(X_train, y_train)
    # 进行预测
    y_pred = perceptron.predict(X_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    # 输出分类报告
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    print("perceptron.coef_.shape[0]: ", perceptron.coef_[0].shape[0])
    print("perceptron.coef_:\n", perceptron.coef_[0])
    print("perceptron_intercept_", perceptron.intercept_[0])

    print("--------------话题向量空间文本分类问题（奇异值分解）-----------------")
    # 5个文档
    docs = ["Love is patient, love is kind. It does not envy, it does not boast, it is not proud.",
            "It does not dishonor others, it is not self-seeking, it is not easily angered, it keeps no record of wrongs.",
            "Love does not delight in evil but rejoices with the truth.",
            "It always protects, always trusts, always hopes, always perseveres.",
            "Love never fails. But where there are prophecies, they will cease; where there are tongues, \
            they will be stilled; where there is knowledge, it will pass away. (1 Corinthians 13:4-8 NIV)"]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)  # 转成权重矩阵
    print("-----------------转成权重------------------")
    print(X,X.shape)
    print("--------获取特征（单词）---------")
    words = vectorizer.get_feature_names_out()
    print(words)
    print(len(words), "个特征（单词）")  # 52个单词
    topics = 4
    lsa = TruncatedSVD(n_components=topics)  # 潜在语义分析，设置4个话题
    X1 = lsa.fit_transform(X)  # 训练并进行转化
    print("----------------lsa奇异值-----------------")
    print(np.diag(lsa.singular_values_))
    print("--------5个文本，在4个话题向量空间下的表示---------")
    print("文本-话题 5*4：\n", X1)  # 5个文本，在4个话题向量空间下的表示
    print("--------lsa.components_---------")
    print("话题-单词 5*4：\n", lsa.components_, lsa.components_.shape)  # 4话题*52单词,话题向量空间

    print()
    pick_docs = 2  # 每个话题挑出2个最具代表性的文档
    topic_docid = [X1[:, t].argsort()[:-(pick_docs + 1):-1] for t in range(topics)]
    # argsort,返回排序后的序号
    print("--------每个话题挑出2个最具代表性的文档---------")
    print(topic_docid)
    pick_keywords = 3  # 每个话题挑出3个关键词
    topic_keywdid = [lsa.components_[t].argsort()[:-(pick_keywords + 1):-1] for t in range(topics)]
    print("--------每个话题挑出3个关键词---------")
    print(topic_keywdid)
    print()
    print("--------打印LSA分析结果---------")
    for t in range(topics):
        print("话题 {}".format(t))
        print("\t 关键词：{}".format(", ".join(words[topic_keywdid[t][j]] for j in range(pick_keywords))))
        for i in range(pick_docs):
            print("\t\t 文档{}".format(i))
            print("\t\t", docs[topic_docid[t][i]])


    print()
    print()
    print("--------------话题向量空间文本分类问题（非负矩阵分解）-----------------")
    docs = ["Love is patient, love is kind. It does not envy, it does not boast, it is not proud.",
            "It does not dishonor others, it is not self-seeking, it is not easily angered, it keeps no record of wrongs.",
            "Love does not delight in evil but rejoices with the truth.",
            "It always protects, always trusts, always hopes, always perseveres.",
            "Love never fails. But where there are prophecies, they will cease; where there are tongues, \
            they will be stilled; where there is knowledge, it will pass away. (1 Corinthians 13:4-8 NIV)"]
    # 转成 TF-IDF 权重矩阵
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)
    # 构建 NMF 模型，设置 4 个主题
    topics = 4
    nmf = NMF(n_components=topics)
    # 训练并转换
    X1 = nmf.fit_transform(X)
    # 获取特征（单词）
    words = vectorizer.get_feature_names_out()
    # 获取奇异值
    print("----------------NMF奇异值------------------")
    print(np.diag(nmf.components_))
    print("--------5个文本，在4个话题向量空间下的表示---------")
    print("文本-话题 5*4：\n", X1)  # 5个文本，在4个话题向量空间下的表示
    print("--------NMF.components_---------")
    print("话题-单词 4*52：\n", nmf.components_, nmf.components_.shape)  # 4 个话题 * 52 单词，话题向量空间
    print()

    # 每个话题挑出 2 个最具代表性的文档
    pick_docs = 2
    topic_docid = [X1[:, t].argsort()[:-(pick_docs + 1):-1] for t in range(topics)]
    print("--------每个话题挑出 2 个最具代表性的文档---------")
    print(topic_docid)
    # 每个话题挑出 3 个关键词
    pick_keywords = 3
    topic_keywdid = [nmf.components_[t].argsort()[:-(pick_keywords + 1):-1] for t in range(topics)]
    print("--------每个话题挑出 3 个关键词---------")
    print(topic_keywdid)
    print()

    # 打印 NMF 分析结果
    print("--------打印 NMF 分析结果---------")
    for t in range(topics):
        print("话题 {}".format(t))
        print("\t 关键词：{}".format(", ".join(words[topic_keywdid[t][j]] for j in range(pick_keywords))))
        for i in range(pick_docs):
            print("\t\t 文档{}".format(i))
            print("\t\t", docs[topic_docid[t][i]])
