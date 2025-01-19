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
        self.signal.emit(10, 0, 1, "Extracting mp3")
        audio_path = os.path.join(self.save_path, "subtitle.mp3")
        self.extract_audio(self.v_path, audio_path)

        try:
            # 分段处理音频

            audio = AudioSegment.from_file(audio_path)
            duration = len(audio)  # 总时长（毫秒）
            segment_duration = 10 * 60 * 1000  # 每段时长：10分钟（600,000毫秒）
            num_segments = (duration + segment_duration - 1) // segment_duration  # 向上取整，计算需要的段数

            subtitleList = []
            subtitleStr = ""
            previous_subtitle = ""  # Keep track of the last subtitle
            current_offset = 0  # 当前时间偏移量

            model_load_start_time = time.time()
            # Use faster-whisper for transcription
            model = WhisperModel(r"models/faster-whisper-small", device="cpu",
                                 compute_type="int8")  # Load the small model of faster-whisper
            model_load_end_time = time.time()
            model_load_total_time = model_load_end_time - model_load_start_time
            print(f"Model loaded in {model_load_total_time} seconds")
            total_transcribe_time = 0

            for i in range(num_segments):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, duration)
                audio_segment = audio[start_time:end_time]

                # 保存每段音频为临时文件
                segment_path = os.path.join(self.save_path, f"segment_{i + 1}.mp3")
                audio_segment.export(segment_path, format="mp3")

                try:
                    # Start transcription
                    start_time = time.time()  # Start the timer
                    segments, info = model.transcribe(segment_path, log_progress=True,multilingual=True)  # Get transcriptions

                    self.signal.emit(10 + i * 90 // num_segments, 0, 1, f"Transcribing Segment {i + 1}")

                    for segment in segments:
                        adjusted_start = round(segment.start + current_offset / 1000, 2)
                        adjusted_end = round(segment.end + current_offset / 1000, 2)
                        subtitleList.append([adjusted_start, adjusted_end, segment.text])

                        # Check for duplicates before adding to subtitleStr
                        if segment.text != previous_subtitle:
                            subtitleStr += segment.text + "\n"
                            previous_subtitle = segment.text

                    # End transcription
                    end_time = time.time()  # End the timer
                    total_transcribe_time += (end_time - start_time)
                    print(f"Time taken for segment {i + 1}: {end_time - start_time} seconds")
                finally:
                    # 删除临时文件
                    if os.path.exists(segment_path):
                        os.remove(segment_path)

                # 更新偏移量
                current_offset += segment_duration
            print(f"Time taken for all {total_transcribe_time} seconds")
            del model

            # 发送字幕给主线程
            self.subtitlesignal.emit(subtitleStr)

            # 将转录结果保存到 CSV 和 SRT 文件中
            self.subtitle2Srt(subtitleList, self.save_path)
            self.subtitle2Csv(subtitleList, self.save_path)

            # 完成处理
            self.signal.emit(101, 101, 101, "Subtitle")
            self.finished.emit(True)
        except Exception as e:
            print(f"Error during processing: {e}")
            # 完成处理
            self.signal.emit(101, 101, 101, "Subtitle")
            self.finished.emit(True)

    def extract_audio(self, v_path, audio_path):
        video = VideoFileClip(v_path)
        try:
            audio = video.audio
            audio.write_audiofile(audio_path, codec='mp3')
        finally:
            video.close()  # 确保释放资源

    def subtitle2Srt(self, subtitleList, savePath):
        # 去除重复字幕
        merged_subtitles = []
        for item in subtitleList:
            if merged_subtitles and merged_subtitles[-1][2] == item[2]:
                # 合并时间段
                merged_subtitles[-1][1] = item[1]
            else:
                merged_subtitles.append(item)

        # Save subtitles to SRT format
        srt_File = os.path.join(savePath, "subtitle.srt")
        with open(srt_File, "w", encoding="utf-8") as f:  # 强制使用 utf-8 编码
            for idx, item in enumerate(merged_subtitles):
                start_time = self.format_time(item[0])
                end_time = self.format_time(item[1])
                f.write(f"{idx + 1}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{item[2]}\n\n")

    def subtitle2Csv(self, subtitleList, savePath):
        # 去除重复字幕
        merged_subtitles = []
        for item in subtitleList:
            if merged_subtitles and merged_subtitles[-1][2] == item[2]:
                # 合并时间段
                merged_subtitles[-1][1] = item[1]
            else:
                merged_subtitles.append(item)

        # Save subtitles to CSV format
        csv_path = os.path.join(savePath, "subtitle.csv")
        with open(csv_path, "w+", newline="", encoding="utf-8") as csv_File:  # 强制使用 utf-8 编码
            writer = csv.writer(csv_File)
            writer.writerow(["start_seconds", "end_seconds", "Subtitles"])
            for item in merged_subtitles:
                writer.writerow([item[0], item[1], item[2]])

    def format_time(self, seconds):
        # Convert seconds to SRT time format (HH:MM:SS,SSS)
        millis = int((seconds - int(seconds)) * 1000)  # Get milliseconds
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

    def stop(self):
        self.is_stop = 1
