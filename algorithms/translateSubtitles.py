# import torch
# import csv
# import re
# from transformers import T5ForConditionalGeneration, T5Tokenizer
# from ui.progressbar import pyqtbar
# from ui.progressbar import *
# from PySide2.QtCore import QThread, Signal
#
#
# class TranslateSrtProcessor(QThread):
#     signal = Signal(int, int, int, str)
#     subtitlesignal = Signal(str)
#     finished = Signal(bool)
#
#     def __init__(self, srt_path, save_path, parent, src_lang='en', target_lang='en'):
#         super(TranslateSrtProcessor, self).__init__()
#         self.srt_path = srt_path
#         self.save_path = save_path
#         self.output_srt_file = self.save_path + 'translated.srt'
#         self.output_csv_file = self.save_path + 'translated.csv'
#         self.parent = parent
#
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.model_name = r'./models/t5_translate_en_ru_zh_small_1024'
#         self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
#         self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
#         self.model.to(self.device)
#
#     def run(self):
#         # Read the input SRT file
#         with open(self.srt_path, "r", encoding="utf-8") as f:
#             lines = f.readlines()
#
#         translated_lines = []
#         csv_data = []
#         translated_text = ""
#         prefix = "translate to zh: "
#         time_pattern = r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})"
#
#         # Process each line
#         for i, line in enumerate(lines):
#             # Extract timestamps
#             time_match = re.match(time_pattern, line.strip())
#             if time_match:
#                 start_time = int(time_match.group(1)) * 3600 + int(time_match.group(2)) * 60 + int(
#                     time_match.group(3)) + int(time_match.group(4)) / 1000
#                 end_time = int(time_match.group(5)) * 3600 + int(time_match.group(6)) * 60 + int(
#                     time_match.group(7)) + int(time_match.group(8)) / 1000
#                 translated_lines.append(line)  # Keep the original timestamp
#
#             elif "-->" not in line and line.strip().isdigit() is False and line.strip():
#                 # Translate the subtitle text
#                 src_text = prefix + line.strip()
#                 if src_text.strip():
#                     input_ids = self.tokenizer(src_text, return_tensors="pt")
#                     generated_tokens = self.model.generate(**input_ids.to(self.device))
#                     translated_segment = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
#                     translated_lines.append(translated_segment + "\n")
#                     translated_text += translated_segment.strip() + "\n"
#
#                     # Save the translation with timestamps to CSV
#                     csv_data.append([round(start_time, 2), round(end_time, 2), translated_segment.strip()])
#                 else:
#                     translated_lines.append("\n")
#             else:
#                 # Keep timestamp or sequence numbers unchanged
#                 translated_lines.append(line)
#
#         # Write the translated SRT file
#         with open(self.output_srt_file, "w", encoding="utf-8") as f:
#             f.writelines(translated_lines)
#
#         # Write the translated subtitles to CSV
#         with open(self.output_csv_file, "w", newline="", encoding="utf-8") as f:
#             writer = csv.writer(f)
#             writer.writerow(['start_time', 'end_time', 'Subtitles'])  # Write the header
#             writer.writerows(csv_data)
#
#         self.subtitlesignal.emit(translated_text.strip())
#         # Emit signals indicating that translation is complete
#         self.signal.emit(101, 101, 101, "Translation Complete")
#         self.finished.emit(True)
#         print(f"Translation completed! SRT file saved at: {self.output_srt_file}")
#         print(f"CSV file saved at: {self.output_csv_file}")
#
#     def stop(self):
#         self.is_stop = 1
import torch
import csv
import re
from transformers import MarianMTModel, MarianTokenizer
from ui.progressBar import *
from PySide6.QtCore import QThread, Signal


class TranslateSrtProcessor(QThread):
    signal = Signal(int, int, int, str)
    subtitlesignal = Signal(str)
    finished = Signal(bool)

    def __init__(self, srt_path, save_path, parent, src_lang='en', target_lang='zh'):  # 修改目标语言为 zh
        super(TranslateSrtProcessor, self).__init__()
        self.srt_path = srt_path
        self.save_path = save_path
        self.output_srt_file = self.save_path + '/translated.srt'
        self.output_csv_file = self.save_path + '/translated.csv'
        self.parent = parent

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = r'./models/opus-mt-en-zh'  # 修改为 opus-mt-en-zh 模型
        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model.to(self.device)

    def run(self):
        # Read the input SRT file
        with open(self.srt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        translated_lines = []
        csv_data = []
        translated_text = ""
        time_pattern = r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})"

        # Process each line
        for i, line in enumerate(lines):
            # Extract timestamps
            time_match = re.match(time_pattern, line.strip())
            if time_match:
                start_time = int(time_match.group(1)) * 3600 + int(time_match.group(2)) * 60 + int(
                    time_match.group(3)) + int(time_match.group(4)) / 1000
                end_time = int(time_match.group(5)) * 3600 + int(time_match.group(6)) * 60 + int(
                    time_match.group(7)) + int(time_match.group(8)) / 1000
                translated_lines.append(line)  # Keep the original timestamp

            elif "-->" not in line and line.strip().isdigit() is False and line.strip():
                # Translate the subtitle text
                src_text = line.strip()
                if src_text.strip():
                    input_ids = self.tokenizer([src_text], return_tensors="pt", padding=True, truncation=True)
                    generated_tokens = self.model.generate(**input_ids.to(self.device))
                    translated_segment = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                    translated_lines.append(translated_segment + "\n")
                    translated_text += translated_segment.strip() + "\n"

                    # Save the translation with timestamps to CSV
                    csv_data.append([round(start_time, 2), round(end_time, 2), translated_segment.strip()])
                else:
                    translated_lines.append("\n")
            else:
                # Keep timestamp or sequence numbers unchanged
                translated_lines.append(line)

        # Write the translated SRT file
        with open(self.output_srt_file, "w", encoding="utf-8") as f:
            f.writelines(translated_lines)

        # Write the translated subtitles to CSV
        with open(self.output_csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(['start_time', 'end_time', 'Subtitles'])  # Write the header
            writer.writerows(csv_data)

        self.subtitlesignal.emit(translated_text.strip())
        # Emit signals indicating that translation is complete
        self.signal.emit(101, 101, 101, "Translation Complete")
        self.finished.emit(True)
        print(f"Translation completed! SRT file saved at: {self.output_srt_file}")
        print(f"CSV file saved at: {self.output_csv_file}")

    def stop(self):
        self.is_stop = 1
