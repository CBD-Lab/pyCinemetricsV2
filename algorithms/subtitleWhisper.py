import os
import torch
from whisper import Whisper
from whisper.model import ModelDimensions
import torch
import cv2
import csv
from algorithms.wordCloud2Frame import WordCloud2Frame
from ui.progressBar import *
from moviepy import VideoFileClip


class SubtitleProcessorWhisper(QThread):
    signal = Signal(int, int, int, str)
    is_stop = 0
    subtitlesignal = Signal(str)
    finished = Signal(bool)

    def __init__(self, v_path, save_path, parent):
        super(SubtitleProcessorWhisper, self).__init__()
        self.v_path = v_path
        self.save_path = save_path
        self.parent = parent

    def run(self):
        # 从视频中提取音频文件
        self.signal.emit(50, 0, 1, "CrewDetect")
        audio_path = os.path.join(self.save_path, "subtitle.mp3")
        self.extract_audio(self.v_path, audio_path)
        try:
            # 使用 Whisper 库转录音频文件
            checkpoint = torch.load(r"./models/large-v3-turbo.pt", weights_only=True,
                                    map_location="cuda" if torch.cuda.is_available() else "cpu")
            dims = ModelDimensions(**checkpoint['dims'])
            model = Whisper(dims)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to("cuda" if torch.cuda.is_available() else "cpu")
            result = model.transcribe(audio_path)

            # 将转录结果保存到 CSV 和 SRT 文件中
            subtitleList = []
            subtitleStr = ""
            for segment in result["segments"]:
                subtitleList.append([round(segment["start"], 2), round(segment["end"], 2), segment["text"]])
                # subtitleList.append([segment["start"], segment["end"], segment["text"]])
                subtitleStr += segment["text"] + "\n"
            self.subtitle2Srt(subtitleList, self.save_path)
            self.subtitle2Csv(subtitleList, self.save_path)

            # 发送字幕给主线程
            self.subtitlesignal.emit(subtitleStr)
            # 完成处理
            self.signal.emit(101, 101, 101, "CrewDetect")
            self.finished.emit(True)
        except:
            # 完成处理
            self.signal.emit(101, 101, 101, "CrewDetect")
            self.finished.emit(True)

    def extract_audio(self, v_path, audio_path):
        video = VideoFileClip(v_path)
        audio = video.audio
        audio.write_audiofile(audio_path, codec='mp3')

    def subtitle2Srt(self, subtitleList, savePath):
        # Save subtitles to SRT format
        srt_File = os.path.join(savePath, "subtitle.srt")
        with open(srt_File, "w", encoding="utf-8") as f:
            for idx, item in enumerate(subtitleList):
                start_time = self.format_time(item[0])
                end_time = self.format_time(item[1])
                f.write(f"{idx + 1}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{item[2]}\n\n")

    def subtitle2Csv(self, subtitleList, savePath):
        # Save subtitles to CSV format
        csv_path = os.path.join(savePath, "subtitle.csv")
        with open(csv_path, "w+", newline="") as csv_File:
            writer = csv.writer(csv_File)
            writer.writerow(["start_seconds", "end_seconds", "Subtitles"])
            for item in subtitleList:
                writer.writerow([item[0], item[1], item[2]])

    def format_time(self, seconds):
        # Convert seconds to SRT time format (HH:MM:SS,SSS)
        millis = int((seconds - int(seconds)) * 1000)  # Get milliseconds
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

    def stop(self):
        self.is_stop = 1
