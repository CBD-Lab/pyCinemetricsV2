import os
import numpy as np
import cv2
import csv
import torch
import torch.nn
import shutil
import torchvision.models as models
import torch.cuda
import nltk
import matplotlib.pyplot as plt
import jieba.posseg as pseg
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from wordcloud import WordCloud
from transformers import GitProcessor, GitForCausalLM
from transformers import MarianMTModel, MarianTokenizer
from scipy.signal import argrelextrema
from torchvision import transforms
from algorithms.wordCloud2Frame import WordCloud2Frame
from ui.progressBar import *
from collections import Counter
from PIL import Image

# 已经下载在./models/nltk_data中
# nltk.download('punkt_tab')
# # 下载 Punkt 分词器
# nltk.download('punkt')
# # 下载 Averaged Perceptron 词性标注器
# nltk.download('averaged_perceptron_tagger')
# # 下载停用词表
# nltk.download('stopwords')

# 设置本地的 NLTK 数据路径
nltk_data_path = "./models/nltk_data"  # 假设数据存放在当前目录下的 "nltk_data" 文件夹中

# 将本地路径添加到 NLTK 数据路径的最前面
nltk.data.path.insert(0, nltk_data_path)

# 测试是否加载成功
# print(nltk.data.find('tokenizers/punkt'))
# print(nltk.data.find('taggers/averaged_perceptron_tagger'))
# print(nltk.data.find('corpora/stopwords'))


class ObjectDetection(QThread):
    signal = Signal(int, int, int, str)  # 进度更新信号
    finished = Signal(bool)        # 任务完成信号
    is_stop = 0                    # 是否中断标志

    def __init__(self, video, image_path, option):
        super(ObjectDetection, self).__init__()
        self.is_stop = 0
        self.image_path = image_path
        self.frame_path = image_path + "/frame" 
        # 视频路径和保存路径
        self.video_path = video
        self.txt_path = os.path.join(self.image_path, 'video.txt')  # 帧号范围文件
        self.output_dir = os.path.join(self.image_path, 'ImagetoText')
        self.output_csv_path = os.path.join(self.image_path, 'image2Text.csv')
        self.option = option
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

    def smooth(self, x, window_len=13, window='hanning'):
        """平滑函数"""
        s = np.r_[2 * x[0] - x[window_len:1:-1],
                x, 2 * x[-1] - x[-1:-window_len:-1]]
        if window == 'flat':  # 移动平均
            w = np.ones(window_len, 'd')
        else:
            w = getattr(np, window)(window_len)
        y = np.convolve(w / w.sum(), s, mode='same')
        return y[window_len - 1:-window_len + 1]


    def process_video_by_segments(self, video_path, txt_path, output_dir, target_width=640, target_height=360, len_window=100, threshold=0.05, min_distance=100):
        """根据帧号范围处理视频并保存每段关键帧"""
        if os.path.exists(output_dir):
            # 删除目录及其所有内容
            shutil.rmtree(output_dir)
        # 创建新的空目录
        os.makedirs(output_dir)
        
        filename = os.path.splitext(os.path.basename(video_path))[0]

        # 加载视频
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"视频总帧数: {frame_count}, 原始分辨率: {original_width}x{original_height}, 目标分辨率: {target_width}x{target_height}")

        # 读取txt文件中的帧号范围
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                segments = [list(map(int, line.strip().split())) for line in f if line.strip()]
        except Exception as e:
            print(f"读取文件失败: {e}")
            return

        # 进度条设置
        file_list = segments
        total_number = len(file_list)  # 总任务数
        task_id = 0  # 子任务序号

        for seg_idx, (start_frame, end_frame) in enumerate(segments):
            if start_frame >= frame_count or end_frame >= frame_count or start_frame > end_frame:
                print(f"段 {seg_idx} 的帧号范围无效: {start_frame}-{end_frame}")
                continue
            
            print(f"处理第 {seg_idx} 段: 帧号范围 {start_frame}-{end_frame}")

            frame_diffs = []
            prev_frame = None

            # 提取段内帧差
            for i in range(start_frame, end_frame + 1):
                if self.is_stop:
                    return
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    print(f"读取帧 {i} 失败，跳过")
                    continue

                # 调整分辨率
                resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                curr_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

                if prev_frame is not None:
                    diff = cv2.absdiff(curr_frame, prev_frame)
                    frame_diffs.append(np.sum(diff) / (target_width * target_height))
                prev_frame = curr_frame


            # 平滑差异并检测局部极值
            frame_diffs = np.array(frame_diffs)
            if len(frame_diffs) > 0:
                smoothed_diffs = self.smooth(frame_diffs, len_window)
                keyframes = argrelextrema(smoothed_diffs, np.greater)[0]

                # 阈值过滤
                keyframes = [k for k in keyframes if smoothed_diffs[k] > threshold]

                # 最小间隔过滤
                filtered_keyframes = []
                for k in keyframes:
                    if len(filtered_keyframes) == 0 or k - filtered_keyframes[-1] >= min_distance:
                        filtered_keyframes.append(k)
                keyframes = filtered_keyframes

                # 保存段内关键帧
                for k_idx in keyframes:
                    frame_idx = start_frame + k_idx
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                        output_path = os.path.join(output_dir, f"{filename}_segment_{seg_idx}_frame_{frame_idx}.jpg")
                        cv2.imwrite(output_path, resized_frame)
            else:
                print(f"段 {seg_idx} 没有足够的帧计算帧差")
            task_id += 1
            percent = round(float(task_id / total_number) * 100)
            self.signal.emit(percent, task_id, total_number, "image2Text")  # 发送实时任务进度和总任务进度
        self.signal.emit(99, task_id, total_number, "image2Text")  # 发送实时任务进度和总任务进度
        cap.release()
        print(f"所有段处理完成，结果保存在: {output_dir}")

    def run(self):

        if self.option == 0:
            process_dir = self.frame_path
        elif self.option == 1:
            process_dir = self.output_dir
            self.process_video_by_segments(self.video_path, self.txt_path, self.output_dir)
            
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            # 可能引发错误的代码块

            subtitle_model_path = r'./models/git-base' 
            processor = GitProcessor.from_pretrained(subtitle_model_path)
            subtitle_model = GitForCausalLM.from_pretrained(subtitle_model_path)
            subtitle_model.to(device)
        except Exception as e:
            self.signal.emit(101, 101, 101, "image2Text")  # 完成后发送信号
            self.finished.emit(True)
            print(f"加载GitBase模型时出错: {e}")
            return
        
        # 进度条设置
        file_list = os.listdir(process_dir)
        total_number = len(file_list)  # 总任务数
        task_id = 0  # 子任务序号

        # 加载模型实现英文转中文
        # 加载预训练模型和分词器
        translate_model_path = './models/opus-mt-en-zh'  # 英文到中文的翻译模型
        translate_model = MarianMTModel.from_pretrained(translate_model_path)
        translate_tokenizer = MarianTokenizer.from_pretrained(translate_model_path)
        translate_model.to(device)
        try:
            # 打开 CSV 文件以写入模式
            with open(self.output_csv_path, mode="w+", newline="", encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                
                # 写入 CSV 表头
                csv_writer.writerow(["Filename", "Caption_en", "Caption_zh-cn"])
                
                # 遍历目录中的所有图片文件
                for filename in os.listdir(process_dir):
                    if self.is_stop:
                        break
                    # 检查文件是否为图片（可以根据需要扩展支持的格式）
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        # 构建图片的完整路径
                        image_path = os.path.join(process_dir, filename)
                        try:
                            # 加载图片
                            image = Image.open(image_path).convert("RGB")  # 确保图片为 RGB 格式
                            
                            # 预处理图片
                            inputs = processor(images=image, return_tensors="pt")
                            
                            # 生成图像描述
                            output_ids = subtitle_model.generate(**inputs, max_length=100, num_beams=4)
                            caption = processor.decode(output_ids[0], skip_special_tokens=True)

                            # 对文本进行分词
                            translated = translate_tokenizer(caption, return_tensors="pt", padding=True)

                            # 使用模型进行翻译
                            translated_output = translate_model.generate(**translated)

                            # 解码翻译后的文本
                            translated_text = translate_tokenizer.decode(translated_output[0], skip_special_tokens=True)
                            print(translated_text)

                            # 写入 CSV 文件
                            csv_writer.writerow([filename, caption, translated_text])
                            # print(f"图片: {filename} 的描述已保存到 CSV 文件")
                        
                        except Exception as e:
                            # 如果图片加载或处理失败，打印错误信息
                            print(f"处理图片 {filename} 时出错: {e}")
                            self.finished.emit(True)
                            break
                    percent = round(float(task_id / total_number) * 100)
                    self.signal.emit(percent, task_id, total_number, "image2Text")  # 发送实时任务进度和总任务进度
                    task_id += 1

            if self.is_stop:
                self.finished.emit(True)
                pass
            else:
                input_csv = os.path.join(self.image_path, 'image2Text.csv')  # 替换为你的 CSV 文件路径
                output_dir = self.image_path  # 替换为你的输出目录

                # 检查文件是否存在
                if not os.path.exists(input_csv):
                    print(f"Error: File {input_csv} does not exist!")
                else:
                    self.generate_word_cloud_from_csv_en(input_csv, 1, os.path.join(output_dir, "wordcloud_en.png"))    # 生成英文词云
                    self.generate_word_cloud_from_csv_ch(input_csv, 2, os.path.join(output_dir, "wordcloud_ch.png"))    # 生成中文词云
                self.signal.emit(101, 101, 101, "image2Text")  # 完成后发送信号
                self.finished.emit(True)
                pass
        except PermissionError as e:
            print(f"文件写入失败: {e}")
            self.signal.emit(101, 101, 101, "image2Text")  # 完成后发送信号
            self.finished.emit(True)
    

    def extract_nouns_by_sentence(self,all_text):
        """
        对文本按句子分割，提取每个句子的名词。
        """
        try:
            sentences = sent_tokenize(all_text)  # 按句子分割
            all_nouns = {}  # 用于存储每个句子的名词

            for sentence in sentences:
                words = word_tokenize(sentence)  # 分词
                words_tagged = pos_tag(words)  # 词性标注
                nouns = [word for word, pos in words_tagged if pos.startswith('NN')]  # 提取名词
                # 将名词存入字典
                if nouns:
                    all_nouns[sentence] = nouns
        except Exception as e:
            print(f"Exception occurred: {e}")
            return None
        return all_nouns

    def generate_word_cloud_from_csv_ch(self, filename, col, output_path):

        # 读取 CSV 文件中的每行句子
        data = []
        try:
            with open(filename, 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                # 跳过第一行（列名）
                next(csv_reader)
                for row in csv_reader:
                    if len(row) > col:
                        # 保留空格
                        data.append(row[col].strip())  # 假设目标列col
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return
        
        if not data:
            print("Error: No data found in the CSV file.")
            return
        
        # 将所有句子拼接成一个字符串
        all_text = " ".join(data)  # 保留原始空格

        if not all_text.strip():
            print("Error: No valid text found.")
            return

        # 自定义停用词（可以添加更多停用词）
        stop_words = set([
            '的', '了', '在', '是', '我', '他', '她', '它', '我们', '你', '他们', '这', '那',
            '与', '和', '对', '为', '上', '下', '中', '大', '小', '要', '也', '就', '不', '能'
        ])

        # 使用 jieba 提取名词
        words = pseg.cut(all_text)  # 使用 pseg.cut() 进行词性标注
        nouns = [word for word, flag in words if 'n' in flag]  # 提取名词（词性标记中包含 'n' 表示名词）

        # 去除停用词
        filtered_nouns = [word for word in nouns if word not in stop_words]

        # 统计词频
        word_counts = Counter(filtered_nouns)

        # 去除低频词（例如，出现次数小于 2 的词）
        min_freq = 2
        filtered_word_counts = {word: count for word, count in word_counts.items() if count >= min_freq}

        font_path = "./fonts/msyh.ttf"
        wordcloud = WordCloud(width=800, height=600, background_color='white', font_path=font_path).generate_from_frequencies(filtered_word_counts)
        print("Wordcloud image size:", wordcloud.to_array().shape)
        # 显示并保存词云图
        plt.figure(figsize=(8, 6))  # 设置显示的尺寸
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(output_path)

    def generate_word_cloud_from_csv_en(self, filename, col, output_path):
        """
        从 CSV 文件生成基于名词的词云图，并逐句提取名词。
        """

        # 读取 CSV 文件中的每行句子
        data = []
        try:
            with open(filename, 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                # 跳过第一行（列名）
                next(csv_reader)
                for row in csv_reader:
                    if len(row) > col:
                        # 保留空格
                        data.append(row[col].strip())  # 假设目标列col
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return
        
        if not data:
            print("Error: No data found in the CSV file.")
            return

        # 将所有句子拼接成一个字符串
        all_text = " ".join(data)  # 保留原始空格

        if not all_text.strip():
            print("Error: No valid text found.")
            return
        # 提取名词
        sentence_nouns = self.extract_nouns_by_sentence(all_text)

        if not sentence_nouns:
            print("Error: No nouns found in the text.")
            return

        # 统计所有句子的名词频率，用于生成词云
        tf = {}
        for nouns in sentence_nouns.values():
            for noun in nouns:
                tf[noun] = tf.get(noun, 0) + 1

        # 去除停用词和低频词
        stop_words = set(stopwords.words('english'))
        filtered_tf = {word: count for word, count in tf.items() if count > 1 and word.lower() not in stop_words}
        if not filtered_tf:
            print("Error: No valid words for the word cloud.")
            return

        # 生成词云图
        wordcloud = WordCloud(width=800, height=600, background_color='white').generate_from_frequencies(filtered_tf)
        print("Wordcloud image size:", wordcloud.to_array().shape)
        # 显示并保存词云图
        plt.figure(figsize=(8, 6))  # 设置显示的尺寸
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(output_path)

    def stop(self):
        self.is_stop = 1