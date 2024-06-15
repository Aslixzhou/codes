from gensim import corpora, models
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation

# 预处理文本
def preprocess(text):
    # 使用Gensim的文本预处理工具进行处理，包括去除标点符号
    custom_filters = [strip_punctuation]
    processed_text = preprocess_string(text, custom_filters)
    return processed_text

if __name__ == '__main__':
    from gensim import corpora, models, similarities

    print("------------------主题分析及文本相似性分析-------------------")

    # 创建一个简单的文本数据集作为示例
    documents = [
        # "This is the first document.",
        # "This document is the second document.",
        # "And this is the third one.",
        # "Is this the first document?",
        "Love is patient, love is kind. It does not envy, it does not boast, it is not proud.",
        "It does not dishonor others, it is not self-seeking, it is not easily angered, it keeps no record of wrongs.",
        "Love does not delight in evil but rejoices with the truth.",
        "It always protects, always trusts, always hopes, always perseveres.",
        "Love never fails. But where there are prophecies, they will cease; where there are tongues, \
        they will be stilled; where there is knowledge, it will pass away. (1 Corinthians 13:4-8 NIV)"
    ]
    # 预处理文本数据：
    # 切分文档为单词
    text = [document.split() for document in documents]

    # 创建一个词典，将每个单词映射到一个唯一的整数ID
    dictionary = corpora.Dictionary(text)

    # 使用词典将文本转化为文档-词袋（document-term）表示
    corpus = [dictionary.doc2bow(doc) for doc in text]

    # 训练LDA模型并执行主题建模：
    # 训练LDA模型
    lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)

    # 输出主题及其词汇
    for topic in lda_model.print_topics():
        print(topic)

    # 文本相似性分析：
    from gensim import similarities

    # 创建一个索引
    index = similarities.MatrixSimilarity(lda_model[corpus])

    # 定义一个查询文本
    query = "This is a new document."

    # 预处理查询文本
    query_bow = dictionary.doc2bow(query.split())

    # 获取查询文本与所有文档的相似性得分
    sims = index[lda_model[query_bow]]

    # 按相似性得分降序排列文档
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    # 输出相似文档及其得分
    for document_id, similarity in sims:
        print(f"Document {document_id}: Similarity = {similarity}")

    print("------------------关键词提取-------------------")

    # 示例文档
    documents = [
        "This is the first document. It contains important information.",
        "This document is the second document. It also has important content.",
        "And this is the third one. It may contain some relevant details.",
        "Is this the first document? Yes, it is."
    ]

    # 预处理文档并创建词袋表示
    text = [preprocess(document) for document in documents]
    dictionary = corpora.Dictionary(text)
    corpus = [dictionary.doc2bow(doc) for doc in text]

    # 计算TF-IDF模型
    tfidf_model = models.TfidfModel(corpus)
    lda_model = models.LdaModel(corpus)

    # 获取TF-IDF加权
    for i, doc in enumerate(corpus):
        tfidf_weights = tfidf_model[doc]
        print(f"TF-IDF Weights for Document {i}: {tfidf_weights}")

    print(len(lda_model.print_topics())) # (94, '0.040*"18" + 0.040*"20" + 0.040*"15" + 0.040*"16" + 0.040*"17" + 0.040*"13" + 0.040*"19" + 0.040*"23" + 0.040*"22" + 0.040*"12"') 表示主题 94，其中每个单词后面的数字表示该单词在主题中的权重。这意味着在主题 94 中，单词 "18"、"20"、"15"、"16"、"17" 等都具有相同的权重，且权重均为 0.040。

    # 获取文档的主题分布
    for i, doc in enumerate(corpus):
        document_topics = lda_model.get_document_topics(doc)
        print(f"Topic Distribution for Document {i}: {document_topics}")

    '''
    TF-IDF Weights for Document: 这部分显示了每个文档中单词的 TF-IDF 权重。TF-IDF（Term Frequency-Inverse Document Frequency）是一种用于评估单词在文档集合中重要性的方法。每个元组表示一个单词的索引和其对应的 TF-IDF 权重。例如，"(0, 0.12322375060267284)" 表示单词索引为 0 的单词在该文档中的 TF-IDF 权重为 0.12322375060267284。
    Topic Distribution for Document: 这部分显示了每个文档的主题分布。在主题模型中，文档通常被表示为在不同主题上的概率分布。每个元组表示一个主题的索引和文档中该主题的概率。例如，"(69, 0.90099984)" 表示主题索引为 69 的主题在该文档中的概率为 0.90099984。
    '''
