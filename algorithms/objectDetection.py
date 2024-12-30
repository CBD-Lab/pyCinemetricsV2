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
nltk.download('punkt_tab')
# 下载 Punkt 分词器
nltk.download('punkt')

# 下载 Averaged Perceptron 词性标注器
nltk.download('averaged_perceptron_tagger')

# 下载停用词表
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords

# 设置本地的 NLTK 数据路径
# nltk_data_path = "models/nltk_data"  # 假设数据存放在当前目录下的 "nltk_data" 文件夹中

# 将本地路径添加到 NLTK 数据路径中
# nltk.data.path.append(nltk_data_path)
# from nltk import word_tokenize, pos_tag
# from nltk.corpus import stopwords

# # 测试是否加载成功
# print(nltk.data.find('tokenizers/punkt'))
# print(nltk.data.find('taggers/averaged_perceptron_tagger'))
# print(nltk.data.find('corpora/stopwords'))

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from transformers import GitProcessor, GitForCausalLM
from scipy.signal import argrelextrema
from torchvision import transforms
from algorithms.wordcloud2frame import WordCloud2Frame
from ui.progressbar import *
from collections import Counter
from PIL import Image

class ObjectDetection(QThread):
    signal = Signal(int, int, int, str)  # 进度更新信号
    finished = Signal(bool)        # 任务完成信号
    is_stop = 0                    # 是否中断标志

    def __init__(self, video, image_path):
        super(ObjectDetection, self).__init__()
        self.is_stop = 0
        self.image_path = image_path
        # 视频路径和保存路径
        self.video_path = video
        self.txt_path = os.path.join(self.image_path, 'video.txt')  # 帧号范围文件
        self.output_dir = os.path.join(self.image_path, 'ImagetoText')
        self.output_csv_path = os.path.join(self.image_path, 'image_descriptions.csv')
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
            with open(txt_path, 'r') as f:
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
            self.signal.emit(percent, task_id, total_number, "objectDetect")  # 发送实时任务进度和总任务进度
        self.signal.emit(99, task_id, total_number, "objectDetect")  # 发送实时任务进度和总任务进度
        cap.release()
        print(f"所有段处理完成，结果保存在: {output_dir}")

    def run(self):
        
        self.process_video_by_segments(self.video_path, self.txt_path, self.output_dir)

        try:
            # 可能引发错误的代码块
            processor = GitProcessor.from_pretrained("microsoft/git-base")
            model = GitForCausalLM.from_pretrained("microsoft/git-base")
        except Exception as e:
            self.signal.emit(101, 101, 101, "objectDetect")  # 完成后发送信号
            self.finished.emit(True)
            print(f"加载GitBase模型时出错: {e}")
            return

        
        # 进度条设置
        file_list = os.listdir(self.output_dir)
        total_number = len(file_list)  # 总任务数
        task_id = 0  # 子任务序号

        try:
            # 打开 CSV 文件以写入模式
            with open(self.output_csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                
                # 写入 CSV 表头
                csv_writer.writerow(["Filename", "Caption"])
                
                # 遍历目录中的所有图片文件
                for filename in os.listdir(self.output_dir):
                    if self.is_stop:
                        break
                    # 检查文件是否为图片（可以根据需要扩展支持的格式）
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        # 构建图片的完整路径
                        image_path = os.path.join(self.output_dir, filename)
                        try:
                            # 加载图片
                            image = Image.open(image_path).convert("RGB")  # 确保图片为 RGB 格式
                            
                            # 预处理图片
                            inputs = processor(images=image, return_tensors="pt")
                            
                            # 生成图像描述
                            output_ids = model.generate(**inputs, max_length=100, num_beams=4)
                            caption = processor.decode(output_ids[0], skip_special_tokens=True)
                            
                            # 写入 CSV 文件
                            csv_writer.writerow([filename, caption])
                            print(f"图片: {filename} 的描述已保存到 CSV 文件")
                        
                        except Exception as e:
                            # 如果图片加载或处理失败，打印错误信息
                            print(f"处理图片 {filename} 时出错: {e}")
                            self.finished.emit(True)
                            break
                    percent = round(float(task_id / total_number) * 100)
                    self.signal.emit(percent, task_id, total_number, "objectDetect")  # 发送实时任务进度和总任务进度
                    task_id += 1

            if self.is_stop:
                self.finished.emit(True)
                pass
            else:
                input_csv = os.path.join(self.image_path, 'image_descriptions.csv')  # 替换为你的 CSV 文件路径
                output_dir = self.image_path  # 替换为你的输出目录

                # 检查文件是否存在
                if not os.path.exists(input_csv):
                    print(f"Error: File {input_csv} does not exist!")
                else:
                    self.generate_word_cloud_from_csv(input_csv, output_dir)
                self.signal.emit(101, 101, 101, "objectDetect")  # 完成后发送信号
                self.finished.emit(True)
                pass#self.object_detection_csv(framelist, self.image_path)
        except PermissionError as e:
            print(f"文件写入失败: {e}")
            self.signal.emit(101, 101, 101, "objectDetect")  # 完成后发送信号
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
        except:
            return None
        return all_nouns

    def generate_word_cloud_from_csv(self,filename, output_dir):
        """
        从 CSV 文件生成基于名词的词云图，并逐句提取名词。
        """

        # 读取 CSV 文件中的每行句子
        data = []
        try:
            with open(filename, encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                for row in csv_reader:
                    if len(row) > 1:
                        # 保留空格
                        data.append(row[1].strip())  # 假设目标列是第二列
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

        # # 打印每个句子的名词
        # for idx, (sentence, nouns) in enumerate(sentence_nouns.items(), start=1):
        #     print(f"Sentence {idx}: {sentence}")
        #     print(f"Nouns: {nouns}")

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

        # 显示并保存词云图
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        output_image = os.path.join(output_dir, "wordcloud.png")
        plt.savefig(output_image, dpi=300, format='png', bbox_inches='tight')
        plt.close()  # 关闭当前图形

    def stop(self):
        self.is_stop = 1