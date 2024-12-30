import os
import re
import cv2
from pathlib import Path
from PySide2.QtWidgets import QDialog, QListWidget, QListWidgetItem, QHBoxLayout, QScrollArea, QPushButton, QVBoxLayout
from PySide2.QtGui import QPixmap, QIcon, QImage
from PySide2.QtCore import Qt, QSize
from algorithms.key_frame_extract import getEffectiveFrame

# 添加帧的弹出窗口
class KeyframeWindow(QDialog):
    def __init__(self, video_path, start_frame, end_frame, mode = "key", parent=None):
        super(KeyframeWindow, self).__init__(parent)

        self.parent = parent
        self.mode = mode
        self.video_path = video_path  # 传入的关键帧列表
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.setWindowTitle("Keyframe Preview")
        # self.frames_dict = self.extract_keyframes(start_frame, end_frame)
        if mode == "key":
            self.frames_dict = self.extract_keyframes(start_frame, end_frame)
        else:
            self.frames_dict = self.extract_frames(start_frame, end_frame)


        # 布局
        layout = QVBoxLayout()
        
        # 创建 QListWidget 用于显示帧图
        self.listWidget = QListWidget(self)
        self.listWidget.setSelectionMode(QListWidget.MultiSelection)  # 设置列表项为多选模式
        self.listWidget.setViewMode(QListWidget.IconMode)  # 设置为图标模式
        self.listWidget.setIconSize(QSize(100, 67))  # 设置每个图标的大小
        self.listWidget.setSpacing(10)  # 设置图标之间的间距
        self.listWidget.setFlow(QListWidget.LeftToRight)  # 水平排列

        # 添加关键帧到 QListWidget 中
        for item in self.frames_dict:
            frame = item['frame']  # 获取当前帧数据
            frame_num = item['frame_num']  # 获取帧编号
            pixmap = self.convert_frame_to_pixmap(frame)  # 将每帧转换为 QPixmap
            item = QListWidgetItem(QIcon(pixmap), frame_num)  # 创建一个列表项
            self.listWidget.addItem(item)  # 将该项添加到 QListWidget 中

        # 设置滚动区域
        scroll_area = QScrollArea(self)
        scroll_area.setWidget(self.listWidget)
        scroll_area.setWidgetResizable(True)

        # 添加滚动区域到主布局
        layout.addWidget(scroll_area)

        # 添加按钮
        button_layout = QHBoxLayout()
        add_button = QPushButton("Add", self)
        cancel_button = QPushButton("Cancel", self)
        
        add_button.clicked.connect(self.add_frames)
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(add_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)

        # 设置窗口初始大小
        self.resize(800, 600)  # 调整窗口大小，800x600为默认尺寸

    def extract_keyframes(self, st, ed):
        # 打开视频文件, 获取总帧数
        video_capture = cv2.VideoCapture(self.video_path)
        frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_len = len(str((int)(frame_count)))


        # 使用帧间差分法获取关键帧
        keyframe_id_set = getEffectiveFrame(self.video_path, st, ed)
        keyframe_id_list = sorted(keyframe_id_set)
        frames_dict = []
        for frame_id in keyframe_id_list:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # 设置帧的位置
            ret, frame = video_capture.read()  # 读取帧
            frame_num = ('%0{}d'.format(frame_len)) % frame_id
            if ret:
                frames_dict.append({'frame': frame, 'frame_num': str(frame_num)})  # 存储读取的帧
        print(keyframe_id_list)
        
        video_capture.release()  # 释放视频资源
        return frames_dict

    def extract_frames(self, st, ed):
        # 打开视频文件, 获取总帧数
        video_capture = cv2.VideoCapture(self.video_path)
        frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_len = len(str((int)(frame_count)))

        frames_dict = []
        # 确保给定的 start 和 end 帧在视频的有效范围内
        if st < 0 or ed >= frame_count or st > ed:
            print("Error: Invalid frame range.")
            return

        # 设置起始帧
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, st)

        # 提取指定范围的帧
        for frame_id in range(st, ed + 1):
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # 设置帧的位置
            ret, frame = video_capture.read()
            frame_num = ('%0{}d'.format(frame_len)) % frame_id
            if ret:
                frames_dict.append({'frame': frame, 'frame_num': str(frame_num)})  # 存储读取的帧
        
        video_capture.release()  # 释放视频资源
        return frames_dict
        

    def convert_frame_to_pixmap(self, frame):
        # 将 OpenCV 格式的图像（BGR）转换为 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 获取图像的高度、宽度和通道数
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        
        # 使用 QImage 将 OpenCV 图像转换为 QPixmap
        pixmap = QPixmap.fromImage(QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888))
        
        return pixmap

    def add_frames(self):
        # 获取选中的帧图并处理
        selected_items = self.listWidget.selectedItems()
        if selected_items:
            for item in selected_items:
                new_item = QListWidgetItem(item)
                self.parent.listWidget_addItem(new_item)
                # for row in range(self.parent.listWidget.count()):
                #     item = self.parent.listWidget.item(row)
                #     print(f"项 {row}: {item.text()}")
                # 获取被选中项的图像或者其他信息
                print("已选择帧图:", item.text())  # 根据需要处理所选帧图
        else:
            print("没有选择任何帧图")
        self.accept()  # 关闭窗口并返回