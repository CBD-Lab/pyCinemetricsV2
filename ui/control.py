import functools
import os
import shutil
import numpy as np
import pandas as pd
from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import (
    QDockWidget, QPushButton, QLabel, QFileDialog, QSlider, QMessageBox, QVBoxLayout,
    QWidget, QGridLayout, QInputDialog
)
from PySide2.QtCore import Qt
from algorithms.objectDetection import ObjectDetection
from algorithms.shotscale import ShotScale
from algorithms.pyqtgraph import TransNetPlot
from algorithms.shotcutTransNetV2 import TransNetV2
from algorithms.subtitleEasyOcr import SubtitleProcessor
from algorithms.subtitleWhisper import SubtitleProcessorWhisper
from algorithms.CrewEasyOcr import CrewProcessor
from algorithms.img2Colors import ColorAnalysis
from ui.ConcatFrameWindow import ConcatFrameWindow
from ui.CsvViewerDialog import CsvViewerDialog
from ui.progressbar import pyqtbar

class Control(QDockWidget):
    def __init__(self, parent, filename):
        super().__init__('Control', parent)
        self.parent = parent
        self.filename = filename
        self.AnalyzeImg = None
        self.AnalyzeImgPath = "" # 好像没有用到
        self.parent.filename_changed.connect(self.on_filename_changed)
        self.subtitleValue = 48    # 每多少帧检查一次字幕
        self.frameConcatValue = 10 # 拼接时每行的帧图个数
        self.init_ui()

    def init_ui(self):

        grid_layout = QGridLayout()

        # 按钮
        self.shotcut = self.create_function_button("Shot", self.shotcut_transNetV2)
        self.shotlenimgplot = self.create_function_button("ShotlenImgPlot", self.plot_transnet_pyqtgraph)
        self.show_shotlen_csv = self.create_function_button("ShowShotLen", lambda: self.show_csv("shotlen.csv"))

        self.frameconcat = self.create_function_button("ShotStitch", self.getframeconcat)
        
        self.subtitle = self.create_function_button("Subtitles", self.getsubtitles)
        self.credits = self.create_function_button("Crew", self.getcredits)
        self.show_subtitles_csv = self.create_function_button("ShowSubtitles", lambda: self.show_csv("subtitle.csv"))

        self.objects = self.create_function_button("Objects", self.object_detect)
        self.show_objects_csv = self.create_function_button("ShowObjects", lambda: self.show_csv("object.csv"))

        self.shotscale = self.create_function_button("ShotScale", self.getshotscale)
        self.show_shotscale_csv = self.create_function_button("ShowShotscale", lambda: self.show_csv("shotscale.csv"))

        self.colors = self.create_function_button("Colors", self.colorAnalyze)
        self.show_colors_csv = self.create_function_button("ShowColors", lambda: self.show_csv("colors.csv"))

        # 滑动条
        self.colorsSlider = QSlider(Qt.Horizontal, self)  # 水平方向
        self.colorsSlider.setMinimum(2)  # 设置最小值
        self.colorsSlider.setMaximum(20)  # 设置最大值
        self.colorsSlider.setSingleStep(1)  # 设置步长值
        self.colorsSlider.setValue(2)  # 设置当前值
        self.colorsSlider.setTickPosition(QSlider.TicksBelow)  # 设置刻度位置，在下方
        self.colorsSlider.setTickInterval(1)  # 设置刻度间隔
        self.colorsSlider.valueChanged.connect(self.colorChange)

        self.frameConcatSlider = QSlider(Qt.Horizontal, self)  # 水平方向
        self.frameConcatSlider.setMinimum(5)  # 设置最小值
        self.frameConcatSlider.setMaximum(30)  # 设置最大值
        self.frameConcatSlider.setSingleStep(1)  # 设置步长值
        self.frameConcatSlider.setValue(10)  # 设置当前值
        self.frameConcatSlider.setTickPosition(QSlider.TicksBelow)  # 设置刻度位置，在下方
        self.frameConcatSlider.setTickInterval(5)  # 设置刻度间隔
        self.frameConcatSlider.valueChanged.connect(self.frameConcatChange)

        # 滑动条的显示
        self.labelColors = QLabel("2", self)
        self.labelColors.setFixedWidth(30)  # 预留固定宽度，支持最多3位数字

        self.labelFrameConcat = QLabel("10", self)
        self.labelFrameConcat.setFixedWidth(30)  # 同样预留固定宽度

        # 创建一个网格布局，并将进度条和标签添加到网格中
        # 第零行， 分镜+csv显示
        grid_layout.addWidget(self.shotcut, 0, 0)
        grid_layout.addWidget(self.shotlenimgplot, 0, 1)
        grid_layout.addWidget(self.show_shotlen_csv, 0, 3)

        # 第一行，镜头拼接
        grid_layout.addWidget(self.frameconcat, 1, 0)
        grid_layout.addWidget(self.frameConcatSlider, 1, 1)
        grid_layout.addWidget(self.labelFrameConcat, 1, 2)

        # 第三行，字幕
        grid_layout.addWidget(self.subtitle, 2, 0)
        grid_layout.addWidget(self.credits, 2, 1)
        grid_layout.addWidget(self.show_subtitles_csv, 2, 3)

        # 第四行，目标检测
        grid_layout.addWidget(self.objects, 3, 0)
        grid_layout.addWidget(self.show_objects_csv, 3, 3)

        # 第五行，shotscale
        grid_layout.addWidget(self.shotscale, 4, 0)
        grid_layout.addWidget(self.show_shotscale_csv, 4, 3)

        # 第六行， Colors
        grid_layout.addWidget(self.colors, 5, 0)
        grid_layout.addWidget(self.colorsSlider, 5, 1)  
        grid_layout.addWidget(self.labelColors, 5, 2)  
        grid_layout.addWidget(self.show_colors_csv, 5, 3)

        # 创建一个QWidget，将主布局设置为这个QWidget的布局
        widget = QWidget()
        widget.setLayout(grid_layout)
        self.setWidget(widget)


    def create_function_button(self, label, function):
        """创建功能按钮并绑定功能函数"""
        button = QPushButton(label, self)
        button.setMaximumWidth(100)  # 为了使得第二列button不会变成Slider的宽度，设定按钮的最大宽度
        button.clicked.connect(lambda: self.toggle_buttons(False))  # 禁用所有按钮
        button.clicked.connect(function)
        return button

    def toggle_buttons(self, enable):
        """启用或禁用所有按钮"""
        self.shotcut.setEnabled(enable)
        self.shotlenimgplot.setEnabled(enable)
        self.show_shotlen_csv.setEnabled(enable)

        self.frameconcat.setEnabled(enable)

        self.subtitle.setEnabled(enable)
        self.credits.setEnabled(enable)
        self.show_subtitles_csv.setEnabled(enable)

        self.objects.setEnabled(enable)
        self.show_objects_csv.setEnabled(enable)

        self.shotscale.setEnabled(enable)
        self.show_shotscale_csv.setEnabled(enable)

        self.colors.setEnabled(enable)
        self.show_colors_csv.setEnabled(enable)
    
    def shotcut_toggle_buttons(self, enable):
        """启用或禁用所有按钮"""
        self.toggle_buttons(enable)
        shot_len_ImgPath = os.path.join(self.image_save, "shotlen.png")
        self.parent.analyze.add_tab_with_image("shot_len", shot_len_ImgPath)
    
    def imageAnalyze_toggle_buttons(self, enable,img_name:str=""):
        """启用或禁用所有按钮"""
        self.toggle_buttons(enable)
        # 分析之后显示图片，传入路径即可
        ImgPath = os.path.join(self.image_save, img_name)
        self.parent.analyze.add_tab_with_image(img_name, ImgPath)

    def subtitles_toggle_buttons(self, enable):
        """启用或禁用所有按钮"""
        self.toggle_buttons(enable)
        shot_len_ImgPath = os.path.join(self.image_save, "subtitle.png")
        self.parent.analyze.add_tab_with_image("subtitle", shot_len_ImgPath)

    def credits_toggle_buttons(self, enable):
        """启用或禁用所有按钮"""
        self.toggle_buttons(enable)
        # 色彩分析之后显示3D图，传入路径即可
        colors_3D_ImgPath = os.path.join(self.image_save, "Crew.png")
        self.parent.analyze.add_tab_with_image("Crew", colors_3D_ImgPath)
    
    def on_filename_changed(self,filename):
        self.filename = filename
        self.frame_save = "./img/" + str(os.path.basename(self.filename )[0:-4]) + "/frame"  # 图片存储路径
        self.image_save = "./img/" + str(os.path.basename(self.filename )[0:-4])
        self.parent.frame_save = self.frame_save
        self.parent.image_save = self.image_save

    '''
    功能模块写成if else 的原因是，在没有视频时或者在没有执行shotcut的情况下点击其他功能时，如果点击按钮，不希望所有按钮变成不能点击的灰色
    '''

    # 分镜 shot
    def shotcut_transNetV2(self):
        if self.filename:
            # 分镜
            model = TransNetV2(self.filename, self)
            # transNetV2_run(self.filename, self)
        else:
            self.shotcut_toggle_buttons(True)
            return
        # 分镜分析之后显示柱状图图，传入路径即可
        model.finished.connect(lambda: self.shotcut_toggle_buttons(True))
        bar = pyqtbar(model)

    # shot concat
    def getframeconcat(self):
        if not os.path.exists(self.frame_save):
            self.toggle_buttons(True)
            return
        concatframe_window = ConcatFrameWindow(self.frame_save, self)
        concatframe_window.finished.connect(lambda: self.toggle_buttons(True))
        concatframe_window.exec_()

    # shot_len图颜色更改
    def plot_transnet_pyqtgraph(self):
        plot_window = TransNetPlot(image_save_path=self.image_save)
        df = pd.read_csv(os.path.join(self.image_save, "shotlen.csv"))
        shot_len = df[['start', 'end', 'length']].values.tolist()
        plot_window.plot_transnet(shot_len)  # 传递数据进行绘图
        plot_window.finished.connect(lambda: self.toggle_buttons(True))
        plot_window.exec_()  # 显示窗口

    # 字幕
    def getsubtitles(self, filename):
        if os.path.exists(self.frame_save):
            imgpath = os.path.basename(self.filename)[0:-4]
            save_path = r"./img/" + imgpath + "/"
            subtitleprocesser = SubtitleProcessorWhisper(self.filename, save_path, self.subtitleValue, self.parent)
        else:
            self.toggle_buttons(True)
            return

        subtitleprocesser.subtitlesignal.connect(self.parent.subtitle.textSubtitle.setPlainText)
        subtitleprocesser.finished.connect(lambda: self.subtitles_toggle_buttons(True))

        self.video_path = self.filename  # 视频路径
        self.parent.shot_finished.emit()
        bar = pyqtbar(subtitleprocesser)
    
    # 演职员表
    def getcredits(self, filename):
        if os.path.exists("./img/"+os.path.basename(self.filename)[0:-4]+"/frame"):
            imgpath = os.path.basename(self.filename)[0:-4]
            save_path = r"./img/" + imgpath + "/"
            creditsprocesser = CrewProcessor(self.filename, save_path, self.subtitleValue, self.parent)
        else:
            self.toggle_buttons(True)
            return

        creditsprocesser.Crewsignal.connect(self.parent.subtitle.textSubtitle.setPlainText)
        creditsprocesser.finished.connect(lambda: self.credits_toggle_buttons(True))

        self.video_path = self.filename  # 视频路径
        self.parent.shot_finished.emit()
        bar = pyqtbar(creditsprocesser)

    # 目标检测
    def object_detect(self):
        if os.path.exists(self.frame_save):
            imgpath = os.path.basename(self.filename)[0:-4]
            objectdetection = ObjectDetection(self.filename, r"./img/" + imgpath)
        else:
            self.imageAnalyze_toggle_buttons(True, img_name="wordcloud.png")
            return

        objectdetection.finished.connect(lambda: self.imageAnalyze_toggle_buttons(True, img_name="wordcloud.png"))
        bar = pyqtbar(objectdetection)

    # ShotScale
    def getshotscale(self):
        if os.path.exists(self.frame_save):
            #csv_file = self.image_save + "shotscale.csv"
            shotscaleclass = ShotScale(25, self.image_save, self.frame_save)
        else:
            self.imageAnalyze_toggle_buttons(True, img_name="shotscale.png")
            return

        shotscaleclass.finished.connect(lambda: self.imageAnalyze_toggle_buttons(True, img_name="shotscale.png"))
        #万一视频名字不变内容变了呢？
        #if not os.path.exists(csv_file):
        png_file = self.image_save + "shotscale.png"
        self.AnalyzeImgPath = png_file
        self.AnalyzeImg = QPixmap(self.AnalyzeImgPath)
        self.AnalyzeImg = self.AnalyzeImg.scaled(300,250)
        bar = pyqtbar(shotscaleclass)
        # self.toggle_button.setVisible(False)  # 初始时不显示按钮
    
    # 颜色分析
    def colorAnalyze(self):
        if os.path.exists(self.frame_save):
            imgpath = os.path.basename(self.filename)[0:-4]
            coloranalysis = ColorAnalysis("", imgpath, self.parent.colorsC)
        else:
            self.imageAnalyze_toggle_buttons(True, img_name="colors.png")
            return
        coloranalysis.finished.connect(lambda: self.imageAnalyze_toggle_buttons(True, img_name="colors.png"))
        bar = pyqtbar(coloranalysis)
        
    #  滚动条变化
    def colorChange(self):
        print("current Color Category slider value:"+str(self.colorsSlider.value()))
        self.labelColors.setText(str(self.colorsSlider.value()))
        self.parent.colorsC = self.colorsSlider.value()

    def frameConcatChange(self):
        print("current frameconcat size Category slider value:" + str(self.frameConcatSlider.value()))
        self.frameConcatValue = self.frameConcatSlider.value()
        self.labelFrameConcat.setText(str(self.frameConcatValue))
    
    # 弹框显示csv文件
    def show_csv(self, csv_filename):
        csv_path = os.path.join(self.image_save, csv_filename)
        if not os.path.exists(csv_path):
            # Show a warning message if the file doesn't exist
            self.show_warning("File Not Found", f"The file at {csv_path} does not exist.")
        else:
            csv_dialog = CsvViewerDialog(csv_path)
            csv_dialog.finished.connect(lambda: self.toggle_buttons(True))
            csv_dialog.exec()

    # 弹出警告框
    def show_warning(self, title, message):
        """ Show a warning message box """
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.finished.connect(lambda: self.toggle_buttons(True))
        msg_box.exec()
    

        
        
        
