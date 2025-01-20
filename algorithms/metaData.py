import os
import re
import easyocr
import cv2
import csv
import numpy as np
from algorithms.wordCloud2Frame import WordCloud2Frame
from ui.progressBar import pyqtbar
from ui.progressBar import *

class CrewProcessor(QThread):
    #  通过类成员对象定义信号对象
    signal = Signal(int, int, int, str)
    #线程中断
    is_stop = 0
    #往主线程传递字幕
    Crewsignal = Signal(str)
    # 线程结束信号
    finished = Signal(bool)

    def __init__(self, v_path, save_path, CrewValue, parent,st,ed):
        super(CrewProcessor, self).__init__()
        self.reader = easyocr.Reader(['ch_sim', 'en'])
        self.v_path = v_path
        self.save_path = save_path
        self.CrewValue = 10
        self.st = st
        self.ed = ed
        self.parent = parent

    def run(self):
        # print(1)
        path=self.v_path
        cap = cv2.VideoCapture(path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.ed>frame_count:
            self.ed=frame_count
        CrewList = []
        CrewStr = ""
        List = []
        Str_z = {}
        i = self.st
        n = 0
        _, frame = cap.read(i)
        h,w=frame.shape[0:2]    #图片尺寸，截取下三分之一和中间五分之四作为字幕检测区域
        start_h = 0
        end_h = h
        start_w = 0
        end_w = w
        img1=frame[start_h:end_h,start_w:end_w,:]
        i=i+1
        th=0.2

        # 进度条设置
        total_number = (self.ed-self.st)  # 总任务数
        # 图片拼接
        stitched_frames = []
        num_frames=[]

        while i<self.ed:
            if self.is_stop:
                self.finished.emit(True)
                break

            if img1 is None:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            _, frame = cap.read(i)
            h, w = frame.shape[0:2]  # 图片尺寸，截取下三分之一和中间五分之四作为字幕检测区域
            start_h = (h // 5)*2
            end_h = h
            start_w = 0
            end_w = w
            img2 = frame[start_h:end_h, start_w:end_w]
            Crew_event= self.CrewDetect(img1, img2, th)
            if Crew_event:
                wordslist = self.reader.readtext(img2)
                if len(wordslist) > 10:
                    #记录图片和帧号
                    stitched_frames.append(frame)
                    num_frames.append(i)

                    Str = ""
                    x_Str = ""
                    old_w = []
                    for w in wordslist:
                        if old_w == []:
                            old_w = w
                        if w[1] is not None:
                            w = list(w)
                            pattern = r'[^\u4e00-\u9fa5a-zA-Z]'
                            w[1] = re.sub(pattern, ' ', w[1])
                            if (not List or w[1] != List[-1][1]):
                                List.append([i, w[1]])
                                if abs(w[0][0][1] - old_w[0][0][1]) > 10:
                                    y_Str = x_Str
                                    x_Str = re.sub(' ', '', x_Str)
                                    if x_Str not in Str_z.values():
                                        Str_z[n] = x_Str
                                        n = n + 1
                                        Str = Str + y_Str + '\n'
                                    x_Str = ""
                                    old_w = w
                                x_Str = x_Str + w[1] + ' '
                    if (Str != "" and Str != '\n'):
                        CrewList.append([i, Str])
                        CrewStr = CrewStr + '\n' + str(i) + '\n' + Str + '\n'
            else:
                img1=img2
            i = i + self.CrewValue
            percent = round(float((i + 1-self.st) / (self.ed-self.st)) * 100)
            # print(percent)
            self.signal.emit(percent, i + 1-self.st, self.ed-self.st, "MetaData")
        self.signal.emit(101, 101, 101, "matedata")  # 完事了再发一次

        if self.is_stop:
            self.finished.emit(True)
        else:
            self.CrewImage(stitched_frames,num_frames)
            # print("显示字幕结果", CrewStr)
            self.Crew2Srt(CrewList, self.save_path)
            self.Crewsignal.emit(CrewStr)
            cap.release()
            self.finished.emit(True)

    def CrewImage(self, stitched_frames, num_frames):
        if stitched_frames:
            # 假设我们按行拼接（每10张一行）
            images_per_row = 10
            # 计算需要填充的黑色图像
            rows = len(stitched_frames) // images_per_row + (1 if len(stitched_frames) % images_per_row != 0 else 0)
            # 创建一个全黑的图像作为填充，大小与其他图像相同
            h, w = stitched_frames[0].shape[:2]
            black_image = np.zeros((h, w, 3), dtype=np.uint8)  # 黑色图像
            # 拼接图像
            stitched_image = []

            for idx, frame in enumerate(stitched_frames):
                # 获取当前图像对应的编号
                frame_number = num_frames[idx]

                # 在左上角写上编号，字体为绿色
                cv2.putText(frame, str(frame_number), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            for i in range(rows):
                # 获取当前行的图像
                row_images = stitched_frames[i * images_per_row:(i + 1) * images_per_row]
                # 如果当前行不足 10 张，补充黑色图像
                if len(row_images) < images_per_row:
                    row_images += [black_image] * (images_per_row - len(row_images))
                # 横向拼接当前行
                stitched_image.append(np.hstack(row_images))

            # 将所有行垂直拼接
            stitched_image = np.vstack(stitched_image)
            # 保存拼接后的图像
            cv2.imwrite(self.save_path + 'matedata.png', stitched_image)
            print("Stitched image saved as 'matedata.png'")

    def Crew2Srt(self,CrewList, savePath):

        # path为输出路径和文件名，newline=''是为了不出现空行
        csv_path = os.path.join(savePath, "matedata.csv")
        csv_File = open(csv_path, "w+", newline = '')
        srt_File = os.path.join(savePath, "matedata.srt")
        # name为列名
        name = ['FrameId','Crew']

        try:
            writer = csv.writer(csv_File)
            writer.writerow(name)
            for i in range(len(CrewList)):
                datarow=[CrewList[i][0]]
                datarow.append(CrewList[i][1])
                writer.writerow(datarow)
        finally:
            csv_File.close()
        with open(srt_File, 'w', encoding='utf-8') as f:
            for i in range(len(CrewList)):
                f.write(str(CrewList[i][1]) + '\n')

    def cmpHash(self,hash1, hash2):
        n = 0
        # hash长度不同则返回-1代表传参出错
        if len(hash1) != len(hash2):
            return -1
        # 遍历判断
        for i in range(len(hash1)):
            # 不相等则n计数+1，n最终为相似度
            if hash1[i] != hash2[i]:
                n = n + 1
        n = n/len(hash1)
        return n

    def aHash(self, img):
        if img is None:
            print("none")
        imgsmall = cv2.resize(img, (16, 4))
        # 转换为灰度图
        gray = cv2.cvtColor(imgsmall, cv2.COLOR_BGR2GRAY)
        # s为像素和初值为0，hash_str为hash值初值为''
        s = 0
        hash_str = ''
        # 遍历累加求像素和
        for i in range(4):
            for j in range(16):
                s = s + gray[i, j]
        # 求平均灰度
        avg = s / 64
        # 灰度大于平均值为1相反为0生成图片的hash值
        for i in range(4):
            for j in range(16):
                if gray[i, j] > avg:
                    hash_str = hash_str + '1'
                else:
                    hash_str = hash_str + '0'
        return hash_str

    def CrewDetect(self,img1, img2, th):
        hash1 = self.aHash(img1)
        hash2 = self.aHash(img2)
        n = self.cmpHash(hash1, hash2)  # 不同加1，相同为0
        if n > th:
            Crew_event=True
        else:
            Crew_event=False
        return Crew_event

    def stop(self):
        self.is_stop = 1
