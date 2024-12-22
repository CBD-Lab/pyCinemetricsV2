import functools
import os
import shutil
from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import (
    QDockWidget, QPushButton, QLabel, QFileDialog, QSlider, QMessageBox, QVBoxLayout,
    QWidget, QGridLayout, QInputDialog
)
from PySide2.QtCore import Qt
from algorithms.objectDetection import ObjectDetection
from algorithms.shotscale import ShotScale
from algorithms.shotcutTransNetV2 import transNetV2_run
from algorithms.CrewEasyOcr import CrewProcessor
from algorithms.img2Colors import ColorAnalysis
from ui.progressbar import pyqtbar

class Control(QDockWidget):
    def __init__(self, parent, filename):
        super().__init__('Control', parent)
        self.parent = parent
        self.filename=filename
        self.AnalyzeImg= None
        self.AnalyzeImgPath=""
        self.parent.filename_changed.connect(self.on_filename_changed)
        # self.video_info_loaded.connect(self.update_table)
        self.colorsC = 5 #要分析的颜色类别的数量
        self.subtitleValue = 10
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        grid_layout = QGridLayout()

        self.shotcut = self.create_function_button("ShotCut", self.shotcut_transNetV2)
        self.colors = self.create_function_button("Colors", self.colorAnalyze)
        self.objects = self.create_function_button("Objects", self.object_detect)
        self.subtitleBtn = self.create_function_button("Subtitles", self.getsubtitles)
        self.shotscale = self.create_function_button("ShotScale", self.getshotscale)

        self.colorsSlider = QSlider(Qt.Horizontal, self)  # 水平方向
        self.colorsSlider.setMinimum(3)  # 设置最小值
        self.colorsSlider.setMaximum(20)  # 设置最大值
        self.colorsSlider.setSingleStep(1)  # 设置步长值
        self.colorsSlider.setValue(5)  # 设置当前值
        self.colorsSlider.setTickPosition(QSlider.TicksBelow)  # 设置刻度位置，在下方
        self.colorsSlider.setTickInterval(5)  # 设置刻度间隔
        self.colorsSlider.valueChanged.connect(self.colorChange)

        self.subtitleSlider = QSlider(Qt.Horizontal, self)  # 水平方向
        self.subtitleSlider.setMinimum(1)  # 设置最小值
        self.subtitleSlider.setMaximum(150)  # 设置最大值
        self.subtitleSlider.setSingleStep(1)  # 设置步长值
        self.subtitleSlider.setValue(48)  # 设置当前值
        self.subtitleSlider.setTickPosition(QSlider.TicksBelow)  # 设置刻度位置，在下方
        self.subtitleSlider.setTickInterval(5)  # 设置刻度间隔
        self.subtitleSlider.valueChanged.connect(self.subtitleChange)

        self.labelColors = QLabel("5", self)
        self.labelColors.setFixedWidth(30)  # 预留固定宽度，支持最多3位数字
        self.labelSubtitlevalue = QLabel("48", self)
        self.labelSubtitlevalue.setFixedWidth(30)  # 同样预留固定宽度


        # shotcut下载按钮
        self.download_shotcut_button = QPushButton(".csv", self)
        self.download_shotcut_button.clicked.connect(
            functools.partial(self.show_save_dialog, "shotlen.csv")
        )

        self.download_color_button = QPushButton(".csv", self)
        self.download_shotcut_button.clicked.connect(
            functools.partial(self.show_save_dialog, "colors.csv")
        )

        self.download_object_button = QPushButton(".csv", self)
        self.download_object_button.clicked.connect(
            functools.partial(self.show_save_dialog, "objects.csv")
        )

        self.download_subtitle_button = QPushButton(".csv", self)
        self.download_subtitle_button.clicked.connect(
            functools.partial(self.show_save_dialog, "Crew.csv")
        )

        self.download_shotscale_button = QPushButton(".csv", self)
        self.download_shotscale_button.clicked.connect(
            functools.partial(self.show_save_dialog, "shotscale.csv")
        )

        # 创建一个网格布局，并将进度条和标签添加到网格中
        grid_layout.addWidget(self.shotcut,0,0)
        grid_layout.addWidget(self.colors,1,0)
        grid_layout.addWidget(self.objects,2,0)
        grid_layout.addWidget(self.subtitleBtn,3,0)
        grid_layout.addWidget(self.shotscale,4,0)

        grid_layout.addWidget(self.colorsSlider, 1, 1)  # 第一行，第二列
        grid_layout.addWidget(self.labelColors, 1, 2)  # 第一行，第三列
        grid_layout.addWidget(self.subtitleSlider, 3, 1)  # 第四行，第二列
        grid_layout.addWidget(self.labelSubtitlevalue, 3, 2)  # 第四行，第三列

        grid_layout.addWidget(self.download_shotcut_button,0,3)
        grid_layout.addWidget(self.download_color_button,1,3)
        grid_layout.addWidget(self.download_object_button,2,3)
        grid_layout.addWidget(self.download_subtitle_button,3,3)
        grid_layout.addWidget(self.download_shotscale_button,4,3)

        # 创建一个QWidget，将主布局设置为这个QWidget的布局
        widget = QWidget()
        widget.setLayout(grid_layout)
        self.setWidget(widget)

    def create_function_button(self, label, function):
        """创建功能按钮并绑定功能函数"""
        button = QPushButton(label, self)
        button.clicked.connect(lambda: self.toggle_buttons(False))  # 禁用所有按钮
        button.clicked.connect(function)
        return button

    def toggle_buttons(self, enable):
        """启用或禁用所有按钮"""
        self.shotcut.setEnabled(enable)
        self.colors.setEnabled(enable)
        self.objects.setEnabled(enable)
        self.subtitleBtn.setEnabled(enable)
        self.shotscale.setEnabled(enable)

    def on_filename_changed(self,filename):
        self.filename=filename

    '''
    功能模块写成if else 的原因是，在没有视频时或者在没有执行shotcut的情况下点击其他功能时，如果点击按钮，不希望所有按钮变成不能点击的灰色
    '''
    def shotcut_transNetV2(self):
        if self.filename:
            self.video_path = self.filename  # 视频路径
            self.frame_save = "./img/" + str(os.path.basename(self.video_path)[0:-4]) + "/frame"  # 图片存储路径
            self.image_save = "./img/" + str(os.path.basename(self.video_path)[0:-4])

            self.AnalyzeImg = QPixmap(self.image_save + "/shotlen.png")
            self.AnalyzeImgPath = self.image_save + "/shotlen.png"
            self.AnalyzeImg = self.AnalyzeImg.scaled(300, 250)
            self.parent.analyze.labelAnalyze.setPixmap(self.AnalyzeImg)
            transNetV2_run(self.video_path, self.image_save, self)
        else:
            self.toggle_buttons(True)
            return

    def colorAnalyze(self):
        if os.path.exists("./img/"+os.path.basename(self.filename)[0:-4]+"/frame"):
            imgpath = os.path.basename(self.filename)[0:-4]
            coloranalysis = ColorAnalysis("", imgpath, self.colorsC)
        else:
            self.toggle_buttons(True)
            return

        coloranalysis.finished.connect(lambda: self.toggle_buttons(True))
        self.AnalyzeImg = QPixmap("img/" + imgpath + "/" + "colors.png")
        self.AnalyzeImgPath = "img/" + imgpath + "/" + "colors.png"
        self.AnalyzeImg = self.AnalyzeImg.scaled(
            300,250)
        self.parent.analyze.labelAnalyze.setPixmap(self.AnalyzeImg)
        bar = pyqtbar(coloranalysis)

    def object_detect(self):
        if os.path.exists("./img/"+os.path.basename(self.filename)[0:-4]+"/frame"):
            imgpath = os.path.basename(self.filename)[0:-4]
            objectdetection = ObjectDetection(r"./img/" + imgpath)
        else:
            self.toggle_buttons(True)
            return

        objectdetection.finished.connect(lambda: self.toggle_buttons(True))
        self.AnalyzeImg = QPixmap("img/" + imgpath + "/objects.png")
        self.AnalyzeImgPath = "img/" + imgpath + "/objects.png"
        self.AnalyzeImg = self.AnalyzeImg.scaled(
            300,250)
        self.parent.analyze.labelAnalyze.setPixmap(self.AnalyzeImg)
        bar = pyqtbar(objectdetection)

    def getsubtitles(self, filename):
        if os.path.exists("./img/"+os.path.basename(self.filename)[0:-4]+"/frame"):
            imgpath = os.path.basename(self.filename)[0:-4]
            save_path = r"./img/" + imgpath + "/"
            subtitleprocesser = CrewProcessor(self.filename, save_path, self.subtitleValue, self.parent)
        else:
            self.toggle_buttons(True)
            return

        subtitleprocesser.Crewsignal.connect(self.parent.subtitle.textSubtitle.setPlainText)
        subtitleprocesser.finished.connect(lambda: self.toggle_buttons(True))

        self.video_path = self.filename  # 视频路径
        self.parent.shot_finished.emit()

        self.AnalyzeImg = QPixmap("img/" + imgpath + "/" + "Crew.png")
        self.AnalyzeImgPath = "img/" + imgpath + "/" + "Crew.png"
        self.AnalyzeImg = self.AnalyzeImg.scaled(
            300, 250)
        self.parent.analyze.labelAnalyze.setPixmap(self.AnalyzeImg)
        bar = pyqtbar(subtitleprocesser)

    def getshotscale(self):
        if os.path.exists("./img/"+os.path.basename(self.filename)[0:-4]+"/frame"):
            image_save = "./img/" + str(os.path.basename(self.filename)[0:-4]) + "/"
            self.frame_save = "./img/" + str(os.path.basename(self.filename)[0:-4]) + "/frame/"  # 图片存储路径
            csv_file = image_save + "shotscale.csv"

            ss = ShotScale(25, image_save, self.frame_save)
        else:
            self.toggle_buttons(True)
            return

        ss.finished.connect(lambda: self.toggle_buttons(True))
        #万一视频名字不变内容变了呢？
        #if not os.path.exists(csv_file):
        png_file = image_save + "shotscale.png"
        self.AnalyzeImgPath = png_file
        self.AnalyzeImg = QPixmap(self.AnalyzeImgPath)
        self.AnalyzeImg = self.AnalyzeImg.scaled(300,250)
        self.parent.analyze.labelAnalyze.setPixmap(self.AnalyzeImg)
        bar = pyqtbar(ss)
        # self.toggle_button.setVisible(False)  # 初始时不显示按钮

    def show_save_dialog(self, file_name):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", "", options=options
        )

        if directory:
            self.download_resources(directory, file_name)

    def colorChange(self):
        print("current Color Category slider value:"+str(self.colorsSlider.value()))
        self.colorsC= self.colorsSlider.value()
        self.labelColors.setText(str(self.colorsC))
    def subtitleChange(self):
        print("current subtitle Category slider value:" + str(self.subtitleSlider.value()))
        self.subtitleValue = self.subtitleSlider.value()
        self.labelSubtitlevalue.setText(str(self.subtitleValue))
    def download_resources(self,directory,file_name):
        # 在这里编写复制资源的代码
        # 你需要将指定的资源文件复制到用户选择的 save_path
        # 例如：
        imgpath = os.path.basename(self.filename)[0:-4]
        resource_path ='./img/'+imgpath+"/"+file_name
        destination_path = os.path.join(directory, file_name)
        shutil.copy(resource_path, destination_path)

        QMessageBox.information(self, "Download", "Resource downloaded successfully!")
