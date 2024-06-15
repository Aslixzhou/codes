import gensim
from gensim import corpora
from pprint import pprint
if __name__ == '__main__':
    '''
    实现功能：
        使用Python的gensim库实现LDA主题模型
    '''
    # 假设我们有一些文档数据
    documents = [
        "这是第一个文档。",
        "这是第二个文档，与第一个文档相似。",
        "第三个文档与前两个文档不同，讨论的是另一个主题。",
        "This is the first document. It contains important information.",
        "This document is the second document. It also has important content.",
        "And this is the third one. It may contain some relevant details.",
        "Is this the first document? Yes, it is.",
    ]

    # 创建文本语料库
    texts = [[text for text in doc.split()] for doc in documents]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 获取词袋内容和对应的索引
    word2id = dictionary.token2id

    # 打印词袋内容和对应的索引
    print("词袋内容和对应的索引:")
    for word, index in word2id.items():
        print(f"词袋索引 {index}: 内容 {word}")

    # 使用LDA模型
    lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=4, random_state=100,
                                       update_every=1, chunksize=100, passes=1000, alpha='auto', per_word_topics=True)

    # 打印主题
    pprint(lda_model.print_topics())

    # 获取文档的主题分布
    doc_topics = lda_model[corpus]
    for i, doc_topic in enumerate(doc_topics):
        print(f"文档 {i} 的主题分布: {doc_topic}")

    '''
    主题分布：
        主题1: 0.9666007
        主题2: 0.012248422
        主题3: 0.012331777
        这表明模型认为文档3在这三个主题中的分布情况，主题1的概率最高，为0.9666007。
    [(3, [1]), (4, [1]), (5, [1]), (6, [1]), (7, [1]), (8, [1]), (9, [1]), (10, [1]), (11, [1])]:
        这部分表示文档3与主题1相关的词语在词袋中的索引。每个数字表示一个词语在词袋中的索引，从3到11都与主题1相关。
    [(3, [(1, 0.9999851)]), (4, [(1, 0.99999636)]), (5, [(1, 0.99999684)]), (6, [(1, 0.99999636)]), (7, [(1, 0.99998194)]), (8, [(1, 0.99999636)]), (9, [(1, 0.99999684)]), (10, [(1, 0.99998516)]), (11, [(1, 0.99997026)])]:
        这部分表示在文档3中，与主题1相关的词语的贡献情况。例如，(4, [(1, 0.99999636)]) 表示在文档3中，词袋索引为4的词语对主题1的贡献概率为0.99999636。
    '''


