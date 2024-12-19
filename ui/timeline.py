import os
import re
from pathlib import Path
from PySide2.QtWidgets import QDockWidget, QListWidget, QListWidgetItem
from PySide2.QtGui import QPixmap, QIcon
from PySide2.QtCore import Qt, QSize

from algorithms.img2Colors import ColorAnalysis


class Timeline(QDockWidget):
    def __init__(self, parent, colorsC):
        # 初始化时间轴面板，设置父组件和颜色配置
        super().__init__('Timeline/Storyboard', parent)
        self.parent = parent  # 父组件，通常是主窗口
        self.colorsC = colorsC  # 分析颜色的个数
        self.init_ui()  # 调用初始化UI的方法

        # 连接父组件发出的信号到相应的槽
        self.parent.filename_changed.connect(self.on_filename_changed)  # 切换打开的视频信号
        self.parent.shot_finished.connect(self.on_shot_finished)  # 视频分镜完成信号（该信号不传参所以额外加一个函数传参）

    def init_ui(self):
        # 初始化时间轴的UI组件
        self.listWidget = QListWidget(self)  # 创建一个QListWidget，用来展示视频帧
        self.listWidget.setViewMode(QListWidget.IconMode)  # 设置为图标模式显示
        self.listWidget.setResizeMode(QListWidget.Adjust)  # 调整列表项的大小
        self.listWidget.setFlow(QListWidget.LeftToRight)  # 列表项水平排列
        self.listWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # 需要时才显示水平滚动条
        self.listWidget.setWrapping(True)  # 自动换行
        self.listWidget.setIconSize(QSize(100, 67))  # 设置图标大小
        self.listWidget.itemSelectionChanged.connect(self.video_play)  # 连接单击图片事件到视频播放方法
        self.setWidget(self.listWidget)  # 设置时间轴面板的主体部件

        # 初始化其他变量
        self.currentImgIdx = 0  # 当前选中的图片索引
        self.currentImg = None  # 当前选中的图片
        self.paths = []  # 存储图片路径列表

    def showShot(self, name):
        # 展示该视频对应的分镜图片
        self.listWidget.clear()  # 清空当前的列表显示

        if name is None or name == '':  # 如果没有提供有效的名称，则不执行任何操作
            return

        try:
            # 获取指定目录中的所有图片文件（该目录下的所有帧）
            self.imglist = os.listdir('img/' + name + "/frame/")
        except Exception as e:  # 捕获可能的异常，如目录不存在
            print(f"Error reading directory 'img/{name}': {e}")
            return

        pattern = r"frame(\d+)\.png"
        if self.imglist:  # 如果目录中存在图片
            for img in self.imglist:  # 遍历每个图片文件
                img_path = 'img/' + name + "/frame/" + img  # 构建图片的完整路径
                img_frame_num = int(re.search(pattern, img).group(1));
                pixmap = QPixmap(img_path)  # 加载图片为 QPixmap 对象
                self.paths.append(img_path)  # 保存图片路径到 paths 列表
                item = QListWidgetItem(QIcon(pixmap), str(img_frame_num))  # 创建一个带图标的列表项
                self.listWidget.addItem(item)  # 将该项添加到列表中
            self.listWidget.itemDoubleClicked.connect(self.draw_pie)  # 双击图片时调用 draw_pie 方法

    def on_filename_changed(self, filename):
        # 当文件名发生变化时调用该方法
        self.listWidget.clear()  # 清空当前的列表
        if filename is None or filename == '':  # 如果文件名无效，则不做任何处理
            return
        # 调用 showShot 方法来显示与文件名对应的镜头
        self.showShot(Path(filename).resolve().stem)

    def on_shot_finished(self):
        # 当镜头分析完成时，重新调用 on_filename_changed 方法
        self.on_filename_changed(self.parent.filename)

    def video_play(self):
        # 当选择的镜头（帧）变化时调用此方法
        self.currentImgIdx = self.listWidget.currentIndex().row()  # 获取当前选中的图片索引
        if self.currentImgIdx in range(len(self.imglist)):  # 确保索引在有效范围内
            # 从图片文件名中提取出帧的 ID
            frameId = int(self.imglist[self.currentImgIdx][5:-4])
            # 发射视频播放状态变化信号，传递当前帧的 ID
            self.parent.video_play_changed.emit(frameId)

    def draw_pie(self):
        # 双击某个帧图时调用该方法
        imageid = self.listWidget.currentRow()  # 获取当前选中的帧的索引
        imgpath = self.paths[imageid]  # 获取当前帧的路径
        analysis = ColorAnalysis(self.currentImg, imgpath, self.colorsC)  # 创建颜色分析对象
        analysis.analysis1img(imgpath, self.colorsC)  # 对该帧进行颜色分析
        path = '/'.join(imgpath.split("/")[:2]) + '/colortmp.png'  # 分析结果的保存路径
        piximg = QPixmap(path)  # 加载分析结果图片
        piximg = piximg.scaled(300, 250)  # 调整图片大小
        # 更新控制面板上的分析图片
        self.parent.control.AnalyzeImgPath = path
        self.parent.analyze.labelAnalyze.setPixmap(piximg)
