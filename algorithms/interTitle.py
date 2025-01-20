import os
import time
from paddleocr import PaddleOCR
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

    def __init__(self, v_path, save_path, intertitleValue,parent):
        super(InterTitle, self).__init__()
        self.reader = PaddleOCR(use_angle_cls=True)
        self.v_path = v_path
        self.save_path = save_path
        self.intertitleValue = intertitleValue
        self.parent = parent

    def run(self):
        intertitleList = []
        intertitleStr = ""
        imglist = os.listdir(self.save_path + "/frame/")
        start_time= time.time()
        if imglist:  # 如果目录中存在图片
            for index, img in enumerate(imglist):  # 遍历每个图片文件

                if self.is_stop:
                    self.finished.emit(True)
                    return

                match = re.search(r'\d+', img)
                number = int(match.group())
                img_path = self.save_path + "/frame/" + img  # 构建图片的完整路径
                wordslist = self.reader.ocr(img_path, cls=True)
                str=""
                if wordslist[0]:
                    for w in wordslist[0]:
                        if w[1][0] is not None:
                            w_str=w[1][0]
                            pattern = r'[^\u4e00-\u9fa5a-zA-Z]'
                            w_str = re.sub(pattern, ' ', w_str)
                            str=str+w_str+' '
                            intertitleStr = intertitleStr + w_str+' '
                    # print(number)
                if str:
                    intertitleList.append([number, str])
                    intertitleStr = intertitleStr + '\n\n'

                percent = round(float((index + 1) / len(imglist)) * 100)
                self.signal.emit(percent, index + 1, len(imglist), "interTitle")  # 发送实时任务进度和总任务进度)

        self.intertitle2Srt(intertitleList, self.save_path,self.v_path)
        print(f"interTitle completed in {time.time()-start_time} seconds")
        self.intertitlesignal.emit(intertitleStr)
        wc2f = WordCloud2Frame()
        tf = wc2f.wordfrequency(os.path.join(self.save_path, "intertitle.csv"))
        wc2f.plotwordcloud(tf, self.save_path, "/intertitle")
        self.signal.emit(101, 101, 101, "interTitle")  # 完事了再发一次
        self.finished.emit(True)
    def intertitle2Srt(self, subtitleList, savePath, video_path):
        """
        Converts the list of subtitles to both a CSV and an SRT file with timestamps.
        :param subtitleList: List of subtitles with frame number and text.
        :param savePath: The directory where the files will be saved.
        :param video_path: Path to the video file for calculating total frames and FPS.
        """
        # Open the video file to get total frame count and FPS

        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = video.get(cv2.CAP_PROP_FPS) if fps is None else fps  # Use given FPS or extract from the video file
        video.release()

        csv_path = os.path.join(savePath, "intertitle.csv")
        csv_File = open(csv_path, "w+", newline='')
        srt_File = os.path.join(savePath, "intertitle.srt")

        name = ['FrameId', 'intertitles']

        try:
            # Write CSV
            writer = csv.writer(csv_File)
            writer.writerow(name)
            for i in range(len(subtitleList)):
                datarow = [subtitleList[i][0]]
                datarow.append(subtitleList[i][1])
                writer.writerow(datarow)
        finally:
            csv_File.close()

        # Write SRT with timestamps
        with open(srt_File, 'w', encoding='utf-8') as f:
            for i in range(len(subtitleList)):
                frame_id = subtitleList[i][0]
                subtitle_text = subtitleList[i][1]

                # Calculate the start and end time based on total frames and FPS
                start_time = frame_id / video_fps
                end_time = (frame_id + 1) / video_fps  # Assuming each subtitle lasts for 1 frame

                start_time_str = self.seconds_to_srt_format(start_time)
                end_time_str = self.seconds_to_srt_format(end_time)

                # Write the subtitle in SRT format
                f.write(f"{i + 1}\n")
                f.write(f"{start_time_str} --> {end_time_str}\n")
                f.write(f"{subtitle_text}\n\n")

        print("SRT file with timestamps has been created.")

    def seconds_to_srt_format(self, seconds):
        """
        Converts a time in seconds to the SRT timestamp format (HH:MM:SS,SSS).
        :param seconds: Time in seconds.
        :return: Time formatted as HH:MM:SS,SSS.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = round(seconds % 60, 3)
        minutes = str(minutes).zfill(2)
        seconds = str(seconds).zfill(3)

        return f"{hours:02}:{minutes}:{seconds[:2]},{seconds[2:]}"

    def stop(self):
        self.is_stop = 1