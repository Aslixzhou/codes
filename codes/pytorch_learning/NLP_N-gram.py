import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import nltk
nltk.download('punkt')

text = """
To be or not to be, that is the question;
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub,
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life;
The oppressor's wrong, the proud man's contumely,
The pangs of despis'd love, the law's delay,
The insolence of office, and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would these fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death—
The undiscover'd country, from whose bourn
No traveller returns—puzzles the will,
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all;
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry,
And lose the name of action.
"""

# 去掉换行符，并将文本转换为小写
text = text.replace('\n', ' ').lower()

def create_ngram_model(text, n):
    words = text.split()  # 将文本分割成单词
    ngrams = []  # 用于存储n-grams的列表

    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i + n])  # 创建一个n-gram
        ngrams.append(ngram)

    return ngrams

n = 5  # 选择n-gram模型
ngram_model = create_ngram_model(text, n)
# 打印前10个n-grams
print(ngram_model[:10])


'''
使用n-gram生成新文本：
    生成文本的方法是随机选择一个n-gram作为起始点，然后根据模型中的n-gram频率来选择接下来的n-gram，依此类推，直到生成所需长度的文本。
'''
import random

def generate_text(ngram_model, n, length=50):
    generated_text = random.choice(ngram_model)  # 随机选择一个n-gram作为起始点
    words = generated_text.split()

    while len(words) < length:
        possible_next_ngrams = [ngram for ngram in ngram_model if ' '.join(words[-n + 1:]) in ngram]
        if not possible_next_ngrams:
            break
        next_ngram = random.choice(possible_next_ngrams)
        words.extend(next_ngram.split())

    generated_text = ' '.join(words)
    return generated_text

generated_text = generate_text(ngram_model, n, length=100)

print(generated_text)


'''改进温度'''
def generate_text_with_temperature(ngram_model, n, length=50, temperature=1.0):
    generated_text = random.choice(ngram_model)
    words = generated_text.split()

    while len(words) < length:
        possible_next_ngrams = [ngram for ngram in ngram_model if ' '.join(words[-n + 1:]) in ngram]
        if not possible_next_ngrams:
            break
        # 根据温度参数调整选择下一个n-gram的随机性
        next_ngram = random.choices(possible_next_ngrams, weights=[1.0 / temperature] * len(possible_next_ngrams))[0]
        words.extend(next_ngram.split())

    generated_text = ' '.join(words)
    return generated_text

# 使用温度参数为0.5生成文本
generated_text = generate_text_with_temperature(ngram_model, n, length=100, temperature=0.5)

print(generated_text)


'''使用TextBlob'''
from textblob import TextBlob
blob = TextBlob(text)
blob.ngrams(n=2)



'''
https://www.bilibili.com/video/BV1hb4y1W7GU/?spm_id_from=333.337.search-card.all.click&vd_source=1396e30dc4fcabf50a79ee190b4031af
https://blog.csdn.net/wuShiJingZuo/article/details/135765152
'''