import os
import re
import cv2
from pathlib import Path
from PySide2.QtWidgets import QDockWidget, QListWidget, QListWidgetItem, QMenu, QAction, QAbstractItemView
from PySide2.QtGui import QPixmap, QIcon
from PySide2.QtCore import Qt, QSize

from algorithms.img2Colors import ColorAnalysis
from ui.keyFrameWindow import KeyframeWindow

class Timeline(QDockWidget):
    def __init__(self, parent):
        # 初始化时间轴面板，设置父组件和颜色配置
        super().__init__('Timeline/Storyboard', parent)
        self.parent = parent  # 父组件，通常是主窗口
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
        self.listWidget.itemDoubleClicked.connect(self.draw_pie)  # 双击图片时调用 draw_pie 方法
        # 右键菜单事件
        self.listWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.listWidget.customContextMenuRequested.connect(self.show_right_click_menu)
        self.setWidget(self.listWidget)  # 设置时间轴面板的主体部件

        # 初始化其他变量
        self.currentImgIdx = 0  # 当前选中的图片索引
        self.currentImg = None  # 当前选中的图片
        self.paths = []  # 存储图片路径列表，会在draw_pie中用来获取图片路径并分析颜色

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
                img_frame_num = re.search(pattern, img).group(1);
                pixmap = QPixmap(img_path)  # 加载图片为 QPixmap 对象
                self.paths.append(img_path)  # 保存图片路径到 paths 列表
                item = QListWidgetItem(QIcon(pixmap), img_frame_num)  # 创建一个带图标的列表项
                self.listWidget.addItem(item)  # 将该项添加到列表中

    def on_filename_changed(self, filename):
        # 当文件名发生变化时调用该方法
        # self.listWidget.clear()  # 清空当前的列表
        # self.listWidget = QListWidget(self)
        if filename is None or filename == '':  # 如果文件名无效，则不做任何处理
            return
        # 调用 showShot 方法来显示与文件名对应的镜头
        self.showShot(Path(filename).resolve().stem)

    def on_shot_finished(self):
        # 当镜头分析完成时，重新调用 on_filename_changed 方法
        self.on_filename_changed(self.parent.filename)

    def video_play(self):
        # 获取当前选中的项
        current_item = self.listWidget.currentItem()
        self.parent.video_play_changed.emit(int(current_item.text()))

    def draw_pie(self):
        # 双击某个帧图时调用该方法
        item = self.listWidget.currentItem()    # 获取当前选中的帧
        # 遍历 self.paths 列表，根据frame_num找到相应路径
        for path in self.paths:
            # 提取路径中的编号部分，假设文件名的格式为 frameXXXX.png
            match = re.search(r'frame(\d+)\.png', path)
            if match and match.group(1) == item.text():
                imgpath = path
                break

        analysis = ColorAnalysis(self.currentImg, imgpath, self.parent.colorsC)  # 创建颜色分析对象
        analysis.analysis1img(imgpath, self.parent.colorsC)  # 对该帧进行颜色分析
        colors_pie_ImgPath = '/'.join(imgpath.split("/")[:2]) + '/colortmp.png'  # 分析结果的保存路径
        # piximg = QPixmap(path)  # 加载分析结果图片
        # piximg = piximg.scaled(300, 250)  # 调整图片大小

        # 更新控制面板上的分析图片
        self.parent.analyze.add_tab_with_image("frame" + item.text(), colors_pie_ImgPath)
        # self.parent.control.AnalyzeImgPath = path
        # self.parent.analyze.labelAnalyze.setPixmap(piximg)
    
    # 单机右键菜单
    def show_right_click_menu(self, pos):
        # 右键点击某个帧图时调用该方法
        # 获取当前点击的位置
        item = self.listWidget.itemAt(pos)
        if item:
            # 创建右键菜单
            contextMenu = QMenu(self)
            
            # 创建菜单项
            action1 = QAction("Add frame", self)
            action2 = QAction("Add key frame", self)
            action3 = QAction("Delete", self)

            # 绑定菜单项的点击事件"
            action1.triggered.connect(lambda: self.menu_add_action("all"))
            action2.triggered.connect(lambda: self.menu_add_action("key"))
            action3.triggered.connect(self.menu_delete_action)

            # 将菜单项添加到菜单中
            contextMenu.addAction(action1)
            contextMenu.addAction(action2)
            contextMenu.addAction(action3)
            
            # 显示菜单
            contextMenu.exec_(self.listWidget.mapToGlobal(pos))

    def menu_delete_action(self):
        # 获取当前选中的 QListWidgetItem
        item = self.listWidget.currentItem()
        if item:
            # 从 listWidget 中删除该项
            row = self.listWidget.row(item)  # 获取该项的行号
            self.listWidget.takeItem(row)  # 删除该行的项
            # 你也可以选择从 self.paths 列表中移除该路径

            # 遍历 self.paths 列表，删除包含该编号的路径
            for path in self.paths:
                # 提取路径中的编号部分，假设文件名的格式为 frameXXXX.png
                match = re.search(r'frame(\d+)\.png', path)
                if match and match.group(1) == item.text():
                    self.paths.remove(path) # 删除path中的路径
                    # 找到相应目录，并且删除图片
                    if os.path.exists(path):  # 确保文件存在
                        os.remove(path)  # 删除文件                       
                    break  # 找到并删除后退出循环

    def menu_add_action(self, mode = "key"):
        # 获取当前选中的帧的文本
        item = self.listWidget.currentItem()
        if not item:
            return

        st = int(item.text())  # 起始帧
        # 获取下一个帧作为结束帧
        next_item = self.listWidget.item(self.listWidget.row(item) + 1)  # 获取下一个列表项
        if next_item:
            ed = int(next_item.text())  # 结束帧
        else:
            ed = st  # 如果没有下一个帧，结束帧就是当前帧
        video_path = self.parent.filename
        print(f'st--ed:{st}-{ed}')

        # 打开一个新的窗口显示这些关键帧
        self.show_keyframe_window(video_path, st, ed, mode)

    def show_keyframe_window(self, video_path, st, ed, mode = "key"):
        # 创建一个新的窗口显示提取的关键帧
        keyframe_window = KeyframeWindow(video_path, st, ed, mode, self)
        keyframe_window.exec_()
    
    
    def sort_listWidget(self):
        # 获取所有的 QListWidgetItem
        items = []
        for i in range(self.listWidget.count()):
            item = self.listWidget.item(i)
            new_item = QListWidgetItem(item)
            items.append(new_item)

        # 根据 item 的文本转换为整数进行排序
        sorted_items = sorted(items, key=lambda item: int(item.text()))

        # 清空原来的列表
        self.listWidget.clear()

        # 按照排序后的顺序重新添加项
        for item in sorted_items:
            self.listWidget.addItem(item)

    # 为keyFrameWindow中的添加操作提供接口
    def listWidget_addItem(self, item):
        frame_num = item.text()     # 如"0170"
        self.listWidget.addItem(item)
        self.sort_listWidget() # 添加之后重新排序
        # 添加后应该保存该帧图像
        self.save_frame(frame_num)

    def save_frame(self, frame_num: str):
        img_save_path = os.path.join(self.parent.frame_save, f"frame{frame_num}.png")
        self.paths.append(img_save_path)
        cap = cv2.VideoCapture(self.parent.filename)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
        _, img = cap.read()
        cv2.imwrite(img_save_path, img)

        
