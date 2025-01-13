import os
import easyocr
import re
import cv2
import csv
from algorithms.wordCloud2Frame import WordCloud2Frame
from ui.progressBar import pyqtbar
from ui.progressBar import *

class InterTitle(QThread):
    #  通过类成员对象定义信号对象
    signal = Signal(int, int, int, str)
    #线程中断
    is_stop = 0
    #往主线程传递字幕
    intertitlesignal = Signal(str)
    # 线程结束信号
    finished = Signal(bool)

    def __init__(self, v_path, save_path, intertitleValue, parent):
        super(InterTitle, self).__init__()
        #self.reader = easyocr.Reader(['ch_sim', 'en'])
        self.reader = easyocr.Reader(['ch_tra', 'en'])
        self.v_path = v_path
        self.save_path = save_path
        self.intertitleValue = intertitleValue
        self.parent = parent

    def run(self):
        intertitleList = []
        intertitleStr = ""
        imglist = os.listdir(self.save_path + "/frame/")
        if imglist:  # 如果目录中存在图片
            for index, img in enumerate(imglist):  # 遍历每个图片文件

                if self.is_stop:
                    self.finished.emit(True)
                    return

                match = re.search(r'\d+', img)
                number = int(match.group())
                img_path = self.save_path + "/frame/" + img  # 构建图片的完整路径
                img_ = cv2.imread(img_path)
                wordslist = self.reader.readtext(img_)
                str=""
                for w in wordslist:
                    if w[1] is not None:
                        w = list(w)
                        pattern = r'[^\u4e00-\u9fa5a-zA-Z]'
                        w[1] = re.sub(pattern, ' ', w[1])
                        str=str+w[1]
                        intertitleStr = intertitleStr + w[1]
                # print(number)
                if str:
                    intertitleList.append([number, str])
                    intertitleStr = intertitleStr + '\n\n'

                percent = round(float((index + 1) / len(imglist)) * 100)
                self.signal.emit(percent, index + 1, len(imglist), "interTitle")  # 发送实时任务进度和总任务进度)

        self.intertitle2Srt(intertitleList, self.save_path)
        self.intertitlesignal.emit(intertitleStr)
        wc2f = WordCloud2Frame()
        tf = wc2f.wordfrequency(os.path.join(self.save_path, "intertitle.csv"))
        wc2f.plotwordcloud(tf, self.save_path, "/intertitle")
        self.signal.emit(101, 101, 101, "interTitle")  # 完事了再发一次
        self.finished.emit(True)

    def intertitle2Srt(self,subtitleList, savePath):

        # path为输出路径和文件名，newline=''是为了不出现空行
        csv_path = os.path.join(savePath, "intertitle.csv")
        csv_File = open(csv_path, "w+", newline = '')
        srt_File = os.path.join(savePath, "intertitle.srt")
        # name为列名
        name = ['FrameId','intertitles']

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

    def stop(self):
        self.is_stop = 1