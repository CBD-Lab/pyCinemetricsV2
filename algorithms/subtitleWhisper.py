import os
import torch
from faster_whisper import WhisperModel
from moviepy import VideoFileClip
from pydub import AudioSegment
import time
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
        # 从视频中提取音频文件并分段
        self.signal.emit(0, 0, 0, "Extracting audio...")
        audio_path = os.path.join(self.save_path, "subtitle.mp3")
        self.extract_audio(self.v_path, audio_path)

        try:
            # 分段处理音频
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio)  # 总时长（毫秒）
            num_segments = 4  # 分成四段
            segment_duration = duration // num_segments

            subtitleList = []
            subtitleStr = ""
            current_offset = 0  # 当前时间偏移量

            # Use faster-whisper for transcription
            model = WhisperModel(r"models/faster-whisper-base")  # Load the small model of faster-whisper

            for i in range(num_segments):

                if self.is_stop:
                    self.finished.emit(True)
                    return
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, duration)
                audio_segment = audio[start_time:end_time]

                # 保存每段音频为临时文件
                segment_path = os.path.join(self.save_path, f"segment_{i + 1}.mp3")
                audio_segment.export(segment_path, format="mp3")

                try:
                    # Start transcription
                    start_time = time.time()  # Start the timer
                    segments, _ = model.transcribe(segment_path)  # Get transcriptions

                    for segment in segments:
                        adjusted_start = round(segment.start + current_offset / 1000, 2)
                        adjusted_end = round(segment.end + current_offset / 1000, 2)
                        subtitleList.append([adjusted_start, adjusted_end, segment.text])
                        subtitleStr += segment.text + "\n"

                    # End transcription
                    end_time = time.time()  # End the timer
                    print(f"Time taken for segment {i + 1}: {end_time - start_time} seconds")

                finally:
                    # 删除临时文件，确保无论发生什么情况都清理文件
                    if os.path.exists(segment_path):
                        os.remove(segment_path)

                # 更新偏移量
                current_offset += segment_duration

            # 将转录结果保存到 CSV 和 SRT 文件中
            self.subtitle2Srt(subtitleList, self.save_path)
            self.subtitle2Csv(subtitleList, self.save_path)

            # 发送字幕给主线程
            self.subtitlesignal.emit(subtitleStr)
            # 完成处理
            self.signal.emit(101, 101, 101, "subtitle")
            self.finished.emit(True)
        except Exception as e:
            print(f"Error during processing: {e}")
            # 完成处理
            self.signal.emit(101, 101, 101, "subtitle")
            self.finished.emit(True)

    def extract_audio(self, v_path, audio_path):
        video = VideoFileClip(v_path)
        try:
            audio = video.audio
            audio.write_audiofile(audio_path, codec='mp3')
        finally:
            video.close()  # 确保释放资源

    def subtitle2Srt(self, subtitleList, savePath):
        # Save subtitles to SRT format
        srt_File = os.path.join(savePath, "subtitle.srt")
        with open(srt_File, "w") as f:  # 强制使用 utf-8 编码
            for idx, item in enumerate(subtitleList):
                start_time = self.format_time(item[0])
                end_time = self.format_time(item[1])
                f.write(f"{idx + 1}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{item[2]}\n\n")

    def subtitle2Csv(self, subtitleList, savePath):
        # Save subtitles to CSV format
        csv_path = os.path.join(savePath, "subtitle.csv")
        with open(csv_path, "w+", newline="") as csv_File:  # 强制使用 utf-8 编码
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
