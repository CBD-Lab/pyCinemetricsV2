import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QColorDialog
from PySide6.QtCore import QThread, Signal
import os
import numpy as np
from PIL import Image
import pandas as pd


class TransNetThread(QThread):
    signal = Signal(int, int, int, str)
    finished = Signal(bool)

    def __init__(self, image_save_path=None, start=0, end=0,parent=None):
        super().__init__()
        self.image_save_path = image_save_path
        df = pd.read_csv(os.path.join(self.image_save_path, "shotlen.csv"))
        shot_len = df[['start', 'end', 'length']].values.tolist()
        self.shot_len = shot_len
        self.plot = TransNetPlot(self.shot_len, self.image_save_path)
        self.parent = parent
        self._stop = False  # 用于控制线程停止的标志位
        self.plot = None

    def run(self):
        self.plot = TransNetPlot(self.shot_len, self.image_save_path)
        self.plot.plotShotlen()  # 调用绘图逻辑
        self.plot.exec_()  # 显示窗口，阻塞线程直到窗口关闭
        self.signal.emit(101, 101, 101, "shot_len")
        self.finished.emit(True)

    def stop(self):
        """停止线程"""
        self._stop = True
        self.plot.close()  # 关闭窗口


class TransNetPlot(QDialog):
    def __init__(self, image_save_path,start,end, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Shot Length Visualization")
        self.setFixedSize(1280, 720)
        self.image_save_path = image_save_path
        df = pd.read_csv(os.path.join(self.image_save_path, "shotlength.csv"))
        self.shot_len = df[['start', 'end', 'length']].astype(int).values.tolist()
        # print(self.shot_len)
        self.start = int(start)
        self.end = int(end)
        # 主布局
        self.main_layout = QVBoxLayout(self)

        # 使用 Matplotlib 创建图表
        self.figure, self.ax = plt.subplots(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        self.main_layout.addWidget(self.canvas)

        # 按钮布局
        self.button_layout = QHBoxLayout()

        # 更改背景颜色按钮
        self.bg_color_button = QPushButton("Change Background Color")
        self.bg_color_button.clicked.connect(self.change_background_color)
        self.button_layout.addWidget(self.bg_color_button)

        # 更改条形颜色按钮
        self.bar_color_button = QPushButton("Change Bar Color")
        self.bar_color_button.clicked.connect(self.change_bar_color)
        self.button_layout.addWidget(self.bar_color_button)

        # 保存按钮
        self.save_img_button = QPushButton("Save and Overwrite")
        self.save_img_button.clicked.connect(self.save_img)
        self.button_layout.addWidget(self.save_img_button)

        # 添加按钮布局到主布局
        self.main_layout.addLayout(self.button_layout)

        # 初始化颜色
        self.bar_color = "blue"
        self.bg_color = "white"

        # 绘制图表
        self.plotShotlen()

    # def plotShotlen(self):
    #     """绘制条形图和嵌入图像"""
    #     self.ax.clear()
    #     # 提取数据
    #     shot_id = np.arange(len(self.shot_len))
    #     shot_length = np.array([shot[2] for shot in self.shot_len])
    #
    #     # 绘制条形图
    #     self.ax.bar(shot_id, shot_length, color=self.bar_color, width=0.8)
    #     self.ax.set_title("Shot Length Visualization", fontsize=16, fontweight='bold')
    #     self.ax.set_xlabel("Shot ID", fontsize=14)
    #     self.ax.set_ylabel("Shot Length (frames)", fontsize=14)
    #
    #     # 设置刻度字体大小
    #     self.ax.tick_params(axis='x', labelsize=12)
    #     self.ax.tick_params(axis='y', labelsize=12)
    #     if len(self.shot_len) < 30:
    #     # 添加镜头图像
    #         try:
    #             img_files = [f for f in os.listdir(os.path.join(self.image_save_path, "frame")) if f.endswith(".png")]
    #             for idx, (_, _, _) in enumerate(self.shot_len):
    #                 if idx < len(img_files):
    #                     image_path = os.path.join(self.image_save_path, "frame", img_files[idx])
    #                     img = Image.open(image_path).resize((96, 54))
    #                     imgbox = OffsetImage(np.array(img), zoom=0.5)
    #                     ab = AnnotationBbox(imgbox, (shot_id[idx], shot_length[idx] + 2), frameon=False)
    #                     self.ax.add_artist(ab)
    #         except Exception as e:
    #             print(f"Error adding images: {e}")
    #
    #     # 设置背景颜色
    #     self.figure.patch.set_facecolor(self.bg_color)
    #     self.ax.set_facecolor(self.bg_color)
    #
    #     # 刷新画布
    #     self.canvas.draw()

    def plotShotlen(self):
        """绘制 start 到 end 范围的条形图和嵌入图像"""
        self.ax.clear()

        # 筛选 start 到 end 范围的镜头
        filtered_shot_len = [
            (idx, shot[2]) for idx, shot in enumerate(self.shot_len)
            if self.start <= shot[0] <= self.end
        ]

        if not filtered_shot_len:
            print("No data in the specified range.")
            return

        # 提取镜头 ID 和长度
        shot_id, shot_length = zip(*filtered_shot_len)

        # 绘制条形图
        self.ax.bar(shot_id, shot_length, color=self.bar_color, width=0.8)
        self.ax.set_title(
            f"Shot Length Visualization (Start: {self.start}, End: {self.end})",
            fontsize=16, fontweight='bold'
        )
        self.ax.set_xlabel("Shot ID", fontsize=14)
        self.ax.set_ylabel("Shot Length (frames)", fontsize=14)

        # 设置刻度字体大小
        self.ax.tick_params(axis='x', labelsize=12)
        self.ax.tick_params(axis='y', labelsize=12)

        if len(filtered_shot_len) < 30:
            # 添加镜头图像
            try:
                img_files = [f for f in os.listdir(os.path.join(self.image_save_path, "frame")) if f.endswith(".png")]
                for idx, (shot_idx, _) in enumerate(filtered_shot_len):
                    if idx < len(img_files):
                        image_path = os.path.join(self.image_save_path, "frame", img_files[shot_idx])
                        img = Image.open(image_path).resize((96, 54))
                        imgbox = OffsetImage(np.array(img), zoom=0.5)
                        ab = AnnotationBbox(imgbox, (shot_idx, shot_length[idx] + 2), frameon=False)
                        self.ax.add_artist(ab)
            except Exception as e:
                print(f"Error adding images: {e}")

        # 设置背景颜色
        self.figure.patch.set_facecolor(self.bg_color)
        self.ax.set_facecolor(self.bg_color)

        # 刷新画布
        self.canvas.draw()

    def change_background_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.bg_color = color.name()
            self.plotShotlen()

    def change_bar_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.bar_color = color.name()
            self.plotShotlen()

    def save_img(self):
        if self.image_save_path:
            save_path = os.path.join(self.image_save_path, "shotlen_with_frames.png")
            self.figure.savefig(save_path, dpi=300)
            print(f"Image saved to {save_path}")

    # def closeEvent(self, event):
    #     """窗口关闭事件"""
    #     if self.parent():
    #         self.parent().stop()  # 停止线程
    #     event.accept()
