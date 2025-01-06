import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from PySide2.QtCore import QThread, Signal

class Similarity(QThread):
    #  通过类成员对象定义信号对象
    signal = Signal(int, int, int, str)
    #线程中断
    is_stop = 0
    # 线程结束信号
    finished = Signal(bool)

    def __init__(self, filename, st, ed):
        super().__init__()
        self.filename = filename
        self.st = st
        self.ed = ed


    def run(self):
        # 读取视频并计算帧间相似度
        print(f"视频地址{self.filename}")
        frame_similarities = self.process_video(self.filename, self.st, self.ed)
        
        # 绘制图像并保存
        self.plot_and_save(frame_similarities)
        
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

        frame_similarities = []

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

    def plot_and_save(self, frame_similarities):

        # 计算 1 秒的 MA (窗口大小基于 fps)
        fps = self.get_fps(self.filename)
        
        window_size_1s = int(fps)  # 1秒对应的帧数
        ma_1s = self.moving_average(1 - frame_similarities, window_size_1s)

        # 计算 5 秒的 MA
        window_size_5s = int(5 * fps)  # 5秒对应的帧数
        ma_5s = self.moving_average(1 - frame_similarities, window_size_5s)

        # 绘制每帧的预测概率图和移动平均图
        plt.figure(figsize=(12, 6))
        plt.plot(range(self.st, self.ed), 1- frame_similarities, label="Probability per frame", color='blue')
        plt.plot(range(self.st, self.ed), ma_1s, label="1-second MA", color='green', linestyle='--')
        plt.plot(range(self.st, self.ed), ma_5s, label="5-second MA", color='red', linestyle='--')

        plt.xlabel("Frame Index")
        plt.ylabel("Prediction Probability")
        plt.title("Prediction Probability with Moving Averages for Each Frame")
        plt.legend()
        plt.grid()

        # 保存到文件
        image_save_path = "./img/" + str(os.path.basename(self.filename )[0:-4]) + "/similarity.png"
        plt.savefig(image_save_path)
        plt.close()

        # 反转图像

        # 绘制每帧的预测概率图和移动平均图
        plt.figure(figsize=(12, 6))
        plt.plot(range(self.st, self.ed), frame_similarities, label="Probability per frame", color='blue')
        plt.plot(range(self.st, self.ed), 1 - ma_1s, label="1-second MA", color='green', linestyle='--')
        plt.plot(range(self.st, self.ed), 1 - ma_5s, label="5-second MA", color='red', linestyle='--')

        plt.xlabel("Frame Index")
        plt.ylabel("Prediction Probability")
        plt.title("Prediction Probability with Moving Averages for Each Frame")
        plt.legend()
        plt.grid()

        # 保存到文件
        image_save_path = "./img/" + str(os.path.basename(self.filename )[0:-4]) + "/similarity_reversed.png"
        plt.savefig(image_save_path)
        plt.close()


