import jieba
import matplotlib.pyplot as plt
import csv
from wordcloud import WordCloud
import csv
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# 确保 NLTK 数据可用
# 设置本地的 NLTK 数据路径
nltk_data_path = "models/nltk_data"  # 假设数据存放在当前目录下的 "nltk_data" 文件夹中

# 将本地路径添加到 NLTK 数据路径中
nltk.data.path.append(nltk_data_path)
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords

# 测试是否加载成功
print(nltk.data.find('tokenizers/punkt'))
print(nltk.data.find('taggers/averaged_perceptron_tagger'))
print(nltk.data.find('corpora/stopwords'))

#三个方法
class WordCloud2Frame:
    def __init__(self):
        pass

    def wordfrequencygitbase(self, filename):
        print(f"Processing file: {filename}")

        # 读取 CSV 文件中的每行句子
        data = []
        with open(filename, encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            for row in csv_reader:  # 将csv 文件中的数据保存到data中
                if len(row) > 1:
                    data.append(row[1])  # 选择某一列加入到data数组中
        print(f"Original data: {data[:5]}")  # 打印前五行数据

        # 清理空格和无效符号
        data_cleaned = []
        for i in data:
            data_cleaned.append(i.replace(", ", ",").replace(" ", ""))
        print(f"Cleaned data: {data_cleaned[:5]}")  # 打印清理后的数据

        # 将所有句子拼接成一个字符串
        all_text = " ".join(data_cleaned)
        print(f"Combined text: {all_text[:100]}")  # 打印部分文本内容

        # 提取名词
        words = word_tokenize(all_text)  # 分词
        words_tagged = pos_tag(words)  # 词性标注
        nouns = [word for word, pos in words_tagged if pos.startswith('NN')]  # 选择名词
        print(f"Nouns: {nouns[:20]}")  # 打印部分名词

        # 统计词频
        tf = {}
        for noun in nouns:
            if noun in tf:
                tf[noun] += 1
            else:
                tf[noun] = 1

        # 去除低频词和停用词
        stop_words = set(stopwords.words('english'))
        filtered_tf = {word: count for word, count in tf.items() if count > 1 and word.lower() not in stop_words}

        # 按词频排序
        sorted_tf = sorted(filtered_tf.items(), key=lambda x: x[1], reverse=True)
        print(f"Sorted word frequencies: {sorted_tf[:10]}")  # 打印前十名词频

        return sorted_tf
    
    def wordfrequency(self,filename):
        print(filename)
        data = []
        with open(filename) as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            # header = next(csv_reader)        # 读取第一行每一列的标题
            for row in csv_reader:  # 将csv 文件中的数据保存到data中
                data.append(row[1])  # 选择某一列加入到data数组中
            print(data)

        data1=[]
        for i in data:
            data1.append(i.replace(", ",",").replace(" ",""))
        print(data1)

        allstr=""
        for d in data1:
            allstr=allstr+' '+d
        print(allstr)
        seg_list = jieba.cut(allstr, cut_all=False)
        print(seg_list)
        tf = {}
        for seg in seg_list:
            if seg in tf:
                tf[seg] += 1
            else:
                tf[seg] = 1
        ci = list(tf.keys())

        for seg in ci:
            if tf[seg] < 1 or len(seg) < 0 or "一" in seg or "," in seg or ";" in seg or " " in seg:  #or seg in stopword
                tf.pop(seg)

        print(tf)

        ci, num, data = list(tf.keys()), list(tf.values()), []
        for i in range(len(tf)):
            data.append((num[i], ci[i]))  # 逐个将键值对存入data中
        data.sort()  # 升序排列
        data.reverse()  # 逆序，得到所需的降序排列

        tf_sorted = {}
        print(len(data), data[0], data[0][0], data[0][1])

        for i in range(len(data)):
            tf_sorted[data[i][1]] = data[i][0]
        print(tf_sorted)
        return tf_sorted

    def wordfrequencyStr(self,datastr):
        datastr = datastr.replace("\n", "")
        seg_list = jieba.cut(datastr, cut_all=False)
        tf = {}
        for seg in seg_list:
            if seg in tf:
                tf[seg] += 1
            else:
                tf[seg] = 1
        ci = list(tf.keys())

        for seg in ci:
            if tf[seg] < 1 or len(seg) < 0 or "一" in seg  or " " in seg  or ";" in seg or "-" in seg:  #or seg in stopword
                tf.pop(seg)

        print(tf)

        ci, num, data = list(tf.keys()), list(tf.values()), []
        for i in range(len(tf)):
            data.append((num[i], ci[i]))  # 逐个将键值对存入data中
        data.sort()     # 升序排列
        data.reverse()  # 逆序，得到所需的降序排列

        tf_sorted = {}
        for i in range(len(data)):
            tf_sorted[data[i][1]] = data[i][0]
        print(tf_sorted)

        return tf_sorted


    def plotwordcloud(self,tf_sorted,save_path,save_type):
        font=r'C:\Windows\Fonts\simfang.ttf'
        print(tf_sorted)
        wc=WordCloud(font_path=font, width=800, height=600).generate_from_frequencies(tf_sorted)
        plt.clf()
        plt.imshow(wc)
        plt.axis('off')
        plt.savefig(save_path+save_type+".png", facecolor='white')


