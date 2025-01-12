import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from PySide6.QtCore import QThread, Signal

class Similarity(QThread):
    #  通过类成员对象定义信号对象
    signal = Signal(int, int, int, str)
    #线程中断
    is_stop = 0
    # 线程结束信号
    finished = Signal(bool)

    def __init__(self, filename, st, ed, option):
        super().__init__()
        self.filename = filename
        self.st = st
        self.ed = ed
        self.option = option


    def run(self):
        # 读取视频并计算帧间相似度
        print(f"视频地址{self.filename}")
        frame_similarities = self.process_video(self.filename, self.st, self.ed)
        
        if self.option == 0:
            # 绘制图像并保存
            self.plot_and_save_single(frame_similarities)
        else:
            self.plot_and_save_double(frame_similarities)
            
        self.signal.emit(101, 101, 101, "similarity")  # 完事了再发一次

        self.finished.emit(True)

    def stop(self):
        self.flag = 1

    def moving_average(self, data, window_size):
        # 计算移动平均
        # 使用 np.pad() 对数据进行边缘填充
        return np.convolve(data, np.ones(window_size) / window_size, mode='same')

    def calculate_ssim(self, frame1, frame2, resize_dim=(48, 27)):
        # 将图像转换为灰度图像并计算SSIM
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 对图像进行resize
        frame1_resized = cv2.resize(frame1_gray, resize_dim)
        frame2_resized = cv2.resize(frame2_gray, resize_dim)

        return ssim(frame1_resized, frame2_resized)

    def process_video(self, video_path, start, end, resize_dim=(48, 27)):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 跳到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        # 第一帧无法比较，第i帧与第i-1比较
        frame_similarities = []
        frame_similarities.append(1)    
        ret, prev_frame = cap.read()

        if not ret:
            cap.release()
            return frame_similarities

        while frame_count < end - start:
            ret, curr_frame = cap.read()
            if not ret:
                break

            if self.is_stop:
                self.finished.emit(True)
                break
            
            frame_count += 1
            # 计算相邻帧的相似度
            similarity = self.calculate_ssim(prev_frame, curr_frame, resize_dim)
            frame_similarities.append(similarity)

            # 更新进度信号
            percent = round((frame_count / (end - start)) * 100)
            self.signal.emit(percent, frame_count, end - start, "similarity")

            prev_frame = curr_frame

        cap.release()
        return np.array(frame_similarities)

    def get_fps(self, video_path):
        # 获取视频的fps
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    def plot_and_save_double(self, frame_similarities):
        # 计算 1 秒的 MA (窗口大小基于 fps)
        fps = self.get_fps(self.filename)
        
        window_size_1s = int(fps)  # 1秒对应的帧数
        ma_1s = self.moving_average(1 - frame_similarities, window_size_1s)

        # 计算 5 秒的 MA
        window_size_5s = int(5 * fps)  # 5秒对应的帧数
        ma_5s = self.moving_average(1 - frame_similarities, window_size_5s)

        # 绘制每帧的预测概率图和移动平均图
        plt.figure(figsize=(12, 6), facecolor='white')
        # 设置 y 轴范围
        plt.ylim(0, 1)
        # 设置 x 轴范围
        plt.xlim(self.st, self.ed)
        plt.plot(range(self.st, self.ed + 1), 1 - frame_similarities, label="Probability per frame", color='blue')
        plt.plot(range(self.st, self.ed + 1), ma_1s, label="1-second MA", color='green', linestyle='--')
        plt.plot(range(self.st, self.ed + 1), ma_5s, label="5-second MA", color='red', linestyle='--')

        plt.xlabel("Frame Index")
        plt.ylabel("Prediction Probability")
        plt.title("Prediction Probability with Moving Averages for Each Frame")
        plt.legend()
        plt.grid()

        # 找出满足条件的帧索引
        highlight_indices = np.where(1 - frame_similarities > 0.7)[0]


        # 获取每个帧的对应图像并在图上标记
        for idx in highlight_indices:
            actual_frame_index = self.st + idx  # 计算实际的帧号
            previous_frame_index = actual_frame_index - 1  # 前一帧的帧号
            
            # 使用 OpenCV 从视频中读取当前帧和前一帧
            cap = cv2.VideoCapture(self.filename)  # 打开视频文件
            
            # 读取当前帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_index)  # 定位到当前帧
            ret, current_frame = cap.read()
            if ret:
                # 转换 BGR (OpenCV 默认格式) 为 RGB
                current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                
                # 将当前帧转换为图像对象
                current_img = Image.fromarray(current_frame_rgb)
                
                # 缩放图片以适应图表
                current_img = current_img.resize((30, 30))  # 可以调整大小

            # 读取前一帧
            if previous_frame_index >= self.st:
                cap.set(cv2.CAP_PROP_POS_FRAMES, previous_frame_index)  # 定位到前一帧
                ret, previous_frame = cap.read()
                if ret:
                    # 转换 BGR (OpenCV 默认格式) 为 RGB
                    previous_frame_rgb = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2RGB)
                    
                    # 将前一帧转换为图像对象
                    previous_img = Image.fromarray(previous_frame_rgb)
                    
                    # 缩放图片以适应图表
                    previous_img = previous_img.resize((30, 30))  # 可以调整大小

                    # 拼接当前帧和前一帧
                    concatenated_img = Image.new('RGB', (current_img.width + previous_img.width, current_img.height))
                    concatenated_img.paste(current_img, (0, 0))  # 将当前帧粘贴到左边
                    concatenated_img.paste(previous_img, (current_img.width + 1, 0))  # 将前一帧粘贴到右边

                    # 创建图像的图像对象
                    imgbox = OffsetImage(concatenated_img, zoom=1)
                    
                    # 获取拼接后的图像的位置坐标
                    x_pos = actual_frame_index
                    y_pos = 1 - frame_similarities[idx] + 0.1  # 对应的概率位置
                    if y_pos > 1.0:
                        y_pos = 0.9
                        x_pos = actual_frame_index + 15
                    
                    # 在图上添加拼接后的图像
                    ab = AnnotationBbox(imgbox, (x_pos, y_pos), frameon=False)
                    plt.gca().add_artist(ab)

            cap.release()  # 释放视频捕获对象

        # 保存到文件
        image_save_path = "./img/" + str(os.path.basename(self.filename )[0:-4]) + "/mosaic.png"
        plt.savefig(image_save_path)
        plt.close()

        # 反转图像

        # 绘制每帧的预测概率图和移动平均图
        plt.figure(figsize=(12, 6), facecolor='white')
        # 设置 y 轴范围
        plt.ylim(0, 1)
        # 设置 x 轴范围
        plt.xlim(self.st, self.ed)
        plt.plot(range(self.st, self.ed + 1), frame_similarities, label="Probability per frame", color='blue')
        plt.plot(range(self.st, self.ed + 1), 1 - ma_1s, label="1-second MA", color='green', linestyle='--')
        plt.plot(range(self.st, self.ed + 1), 1 - ma_5s, label="5-second MA", color='red', linestyle='--')

        plt.xlabel("Frame Index")
        plt.ylabel("Prediction Probability")
        plt.title("Prediction Probability with Moving Averages for Each Frame")
        plt.legend()
        plt.grid()


        # 找出满足条件的帧索引
        highlight_indices = np.where(frame_similarities < 0.3)[0]

        # 获取每个帧的对应图像并在图上标记
        for idx in highlight_indices:
            actual_frame_index = self.st + idx  # 计算实际的帧号
            previous_frame_index = actual_frame_index - 1  # 前一帧的帧号
            
            # 使用 OpenCV 从视频中读取当前帧和前一帧
            cap = cv2.VideoCapture(self.filename)  # 打开视频文件
            
            # 读取当前帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_index)  # 定位到当前帧
            ret, current_frame = cap.read()
            if ret:
                # 转换 BGR (OpenCV 默认格式) 为 RGB
                current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                
                # 将当前帧转换为图像对象
                current_img = Image.fromarray(current_frame_rgb)
                
                # 缩放图片以适应图表
                current_img = current_img.resize((30, 30))  # 可以调整大小

            # 读取前一帧
            if previous_frame_index >= self.st:
                cap.set(cv2.CAP_PROP_POS_FRAMES, previous_frame_index)  # 定位到前一帧
                ret, previous_frame = cap.read()
                if ret:
                    # 转换 BGR (OpenCV 默认格式) 为 RGB
                    previous_frame_rgb = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2RGB)
                    
                    # 将前一帧转换为图像对象
                    previous_img = Image.fromarray(previous_frame_rgb)
                    
                    # 缩放图片以适应图表
                    previous_img = previous_img.resize((30, 30))  # 可以调整大小

                    # 拼接当前帧和前一帧
                    concatenated_img = Image.new('RGB', (current_img.width + previous_img.width, current_img.height))
                    concatenated_img.paste(current_img, (0, 0))  # 将当前帧粘贴到左边
                    concatenated_img.paste(previous_img, (current_img.width + 1, 0))  # 将前一帧粘贴到右边

                    # 创建图像的图像对象
                    imgbox = OffsetImage(concatenated_img, zoom=1)
                    
                    # 获取拼接后的图像的位置坐标
                    x_pos = actual_frame_index
                    y_pos = frame_similarities[idx] - 0.1  # 对应的概率位置
                    if y_pos < 0:
                        y_pos = 0.1
                        x_pos = actual_frame_index + 15
                    
                    # 在图上添加拼接后的图像
                    ab = AnnotationBbox(imgbox, (x_pos, y_pos), frameon=False)
                    plt.gca().add_artist(ab)

            cap.release()  # 释放视频捕获对象

        # 保存到文件
        image_save_path = "./img/" + str(os.path.basename(self.filename )[0:-4]) + "/similarity_reversed.png"
        plt.savefig(image_save_path)
        plt.close()

    def plot_and_save_single(self, frame_similarities):

        # 计算 1 秒的 MA (窗口大小基于 fps)
        fps = self.get_fps(self.filename)
        
        window_size_1s = int(fps)  # 1秒对应的帧数
        ma_1s = self.moving_average(1 - frame_similarities, window_size_1s)

        # 计算 5 秒的 MA
        window_size_5s = int(5 * fps)  # 5秒对应的帧数
        ma_5s = self.moving_average(1 - frame_similarities, window_size_5s)

        # 绘制每帧的预测概率图和移动平均图
        plt.figure(figsize=(12, 6))
        # 设置坐标轴背景颜色
        plt.gca().set_facecolor('white')  # 白色背景
        # 设置 y 轴范围
        plt.ylim(0, 1)
        # 设置 x 轴范围
        plt.xlim(self.st, self.ed)
        plt.plot(range(self.st, self.ed + 1), 1 - frame_similarities, label="Probability per frame", color='blue')
        plt.plot(range(self.st, self.ed + 1), ma_1s, label="1-second MA", color='green', linestyle='--')
        plt.plot(range(self.st, self.ed + 1), ma_5s, label="5-second MA", color='red', linestyle='--')

        plt.xlabel("Frame Index")
        plt.ylabel("Prediction Probability")
        plt.title("Prediction Probability with Moving Averages for Each Frame")
        plt.legend()
        plt.grid()

        # 找出满足条件的帧索引
        highlight_indices = np.where(1 - frame_similarities > 0.7)[0]


        # 获取每个帧的对应图像并在图上标记
        for idx in highlight_indices:
            actual_frame_index = self.st + idx  # 计算实际的帧号
            
            # 使用 OpenCV 从视频中读取对应帧
            cap = cv2.VideoCapture(self.filename)  # 打开视频文件
            cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_index)  # 定位到指定的帧
            
            ret, frame = cap.read()  # 读取该帧

            if ret:
                # 转换 BGR (OpenCV 默认格式) 为 RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 将帧转换为图像对象
                img = Image.fromarray(frame_rgb)
                
                # 缩放图片以适应图表
                img = img.resize((30, 30))  # 可以调整大小
                imgbox = OffsetImage(img, zoom=1)
                
                # 获取该位置的坐标
                x_pos = actual_frame_index
                y_pos = 1 - frame_similarities[idx] + 0.1  # 对应的概率位置
                if y_pos > 1.0:
                    y_pos = 0.9
                    x_pos = actual_frame_index + 15
                
                # 在图上添加图片
                ab = AnnotationBbox(imgbox, (x_pos, y_pos), frameon=False)
                plt.gca().add_artist(ab)

            cap.release()  # 释放视频捕获对象

        # 保存到文件
        image_save_path = "./img/" + str(os.path.basename(self.filename )[0:-4]) + "/similarity.png"
        plt.savefig(image_save_path)
        plt.close()

        # 反转图像

        # 绘制每帧的预测概率图和移动平均图
        plt.figure(figsize=(12, 6))
        # 设置坐标轴背景颜色
        plt.gca().set_facecolor('white')  # 白色背景
        # 设置 y 轴范围
        plt.ylim(0, 1)
        # 设置 x 轴范围
        plt.xlim(self.st, self.ed)
        plt.plot(range(self.st, self.ed + 1), frame_similarities, label="Probability per frame", color='blue')
        plt.plot(range(self.st, self.ed + 1), 1 - ma_1s, label="1-second MA", color='green', linestyle='--')
        plt.plot(range(self.st, self.ed + 1), 1 - ma_5s, label="5-second MA", color='red', linestyle='--')

        plt.xlabel("Frame Index")
        plt.ylabel("Prediction Probability")
        plt.title("Prediction Probability with Moving Averages for Each Frame")
        plt.legend()
        plt.grid()


        # 找出满足条件的帧索引
        highlight_indices = np.where(frame_similarities < 0.3)[0]

        # 获取每个帧的对应图像并在图上标记
        for idx in highlight_indices:
            actual_frame_index = self.st + idx  # 计算实际的帧号
            
            # 使用 OpenCV 从视频中读取对应帧
            cap = cv2.VideoCapture(self.filename)  # 打开视频文件
            cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_index)  # 定位到指定的帧
            
            ret, frame = cap.read()  # 读取该帧
            if ret:
                # 转换 BGR (OpenCV 默认格式) 为 RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 将帧转换为图像对象
                img = Image.fromarray(frame_rgb)
                
                # 缩放图片以适应图表
                img = img.resize((30, 30))  # 可以调整大小
                imgbox = OffsetImage(img, zoom=1)
                
                # 获取该位置的坐标
                x_pos = actual_frame_index
                y_pos = frame_similarities[idx] - 0.1  # 对应的概率位置
                if y_pos < 0:
                    y_pos = 0.1
                    x_pos = actual_frame_index + 15
                
                # 在图上添加图片
                ab = AnnotationBbox(imgbox, (x_pos, y_pos), frameon=False)
                plt.gca().add_artist(ab)

            cap.release()  # 释放视频捕获对象

        # 保存到文件
        image_save_path = "./img/" + str(os.path.basename(self.filename )[0:-4]) + "/similarity_reversed.png"
        plt.savefig(image_save_path)
        plt.close()


