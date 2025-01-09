import os
import easyocr
import cv2
import csv
from algorithms.wordCloud2Frame import WordCloud2Frame
from ui.progressBar import *

class SubtitleProcessor(QThread):
    #  通过类成员对象定义信号对象
    signal = Signal(int, int, int, str)
    #线程中断
    is_stop = 0
    #往主线程传递字幕
    subtitlesignal = Signal(str)
    # 线程结束信号
    finished = Signal(bool)

    def __init__(self, v_path, save_path, subtitleValue, parent):
        super(SubtitleProcessor, self).__init__()
        self.reader = easyocr.Reader(['ch_sim', 'en'])
        self.v_path = v_path
        self.save_path = save_path
        self.subtitleValue = subtitleValue
        self.parent = parent

    def run(self):
        path=self.v_path
        cap = cv2.VideoCapture(path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        subtitleList = []
        subtitleStr = ""
        i = 0
        _, frame = cap.read(i)
        h,w=frame.shape[0:2]    #图片尺寸，截取下三分之一和中间五分之四作为字幕检测区域
        start_h = (h // 3)*2
        end_h = h
        start_w = w // 20
        end_w = (w // 20) * 19
        img1=frame[start_h:end_h,start_w:end_w,:]
        i=i+1
        th=0.2

        # 进度条设置
        total_number = frame_count  # 总任务数

        while i<frame_count:
            if self.is_stop:
                self.finished.emit(True)
                break

            if img1 is None:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            _, frame = cap.read(i)
            h, w = frame.shape[0:2]  # 图片尺寸，截取下三分之一和中间五分之四作为字幕检测区域
            start_h = (h // 3)*2
            end_h = h
            start_w = w // 10
            end_w = (w // 10) * 9
            img2 = frame[start_h:end_h, start_w:end_w]
            subtitle_event= self.subtitleDetect(img1, img2, th)
            if subtitle_event:
                wordslist = self.reader.readtext(img2)
                for w in wordslist:
                    if w[1] is not None:
                        if not subtitleList or w[1] + '\n' != subtitleList[-1][1]:
                            subtitleList.append([i, w[1]])
                            subtitleStr = subtitleStr + w[1] + '\n'
            else:
                img1=img2
            i = i + self.subtitleValue
            percent = round(float(i / frame_count) * 100)
            self.signal.emit(percent, i, frame_count, "subtitle")   # 刷新进度条 不严谨
        self.signal.emit(101, 101, 101, "subtitle")  # 完事了再发一次

        if self.is_stop:
            self.finished.emit(True)
        else:
            print("显示字幕结果", subtitleStr)
            self.subtitle2Srt(subtitleList, self.save_path)
            self.subtitlesignal.emit(subtitleStr)
            cap.release()
            wc2f = WordCloud2Frame()
            tf = wc2f.wordfrequency(os.path.join(self.save_path, "subtitle.csv"))
            wc2f.plotwordcloud(tf, self.save_path, "/subtitle")
            self.finished.emit(True)

    def subtitle2Srt(self,subtitleList, savePath):

        # path为输出路径和文件名，newline=''是为了不出现空行
        csv_path = os.path.join(savePath, "subtitle.csv")
        csv_File = open(csv_path, "w+", newline = '')
        srt_File = os.path.join(savePath, "subtitle.srt")
        # name为列名
        name = ['FrameId','Subtitles']

        try:
            writer = csv.writer(csv_File)
            writer.writerow(name)
            for i in range(len(subtitleList)):
                datarow=[subtitleList[i][0]]
                datarow.append(subtitleList[i][1])
                writer.writerow(datarow)
        finally:
            csv_File.close()
        with open(srt_File, 'w', encoding='utf-8') as f:
            for i in range(len(subtitleList)):
                f.write(str(subtitleList[i][1]) + '\n')

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

    def subtitleDetect(self,img1, img2, th):
        hash1 = self.aHash(img1)
        hash2 = self.aHash(img2)
        n = self.cmpHash(hash1, hash2)  # 不同加1，相同为0
        if n > th:
            subtitle_event=True
        else:
            subtitle_event=False
        return subtitle_event

    def stop(self):
        self.is_stop = 1