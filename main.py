import sys
import os
import qdarktheme
import cv2

# 自动设置 Qt 环境变量
try:
    from PySide6 import QtCore
    qt_plugin_path = QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.PluginsPath)
    os.environ['QT_PLUGIN_PATH'] = qt_plugin_path
    print(f"已自动设置 QT_PLUGIN_PATH: {qt_plugin_path}")
except Exception as e:
    print(f"无法自动设置 Qt 环境变量: {e}")

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QMessageBox, QProgressBar
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, Signal
from ui.timeline import Timeline
from ui.info import Info
from ui.analyze import Analyze
from ui.subtitle import Subtitle
from concurrent.futures import ThreadPoolExecutor
from ui.vlcPlayer import VLCPlayer
from ui.control import Control



# from ui.subtitleEasyOcr import getsubtitleEasyOcr,subtitle2Srt

class MainWindow(QMainWindow):
    # 定义信号
    filename_changed = Signal(str)  # 文件名更改信号，传递字符串参数
    shot_finished = Signal()        # 表示视频分析完成的信号
    shot_changed = Signal(str)         # 表示在timeline中有添加/删除操作，需要更新info
    video_play_changed = Signal(int)  # 视频播放状态改变信号，传递整数参数

    def __init__(self):
        super().__init__()
        # 创建线程池，用于处理多线程任务
        self.threadpool = ThreadPoolExecutor()
        self.filename = ''  # 当前文件名
        self.frame_save = ""  # 图片存储路径
        self.image_save = ""
        self.AnalyzeImgPath = ''  # Analyze窗口中要显示的图像的路径
        self.colorsC = 2
        self.init_ui()  # 初始化界面
        self.frameCnt = 0

    def init_ui(self):

        # 信号与槽的连接
        self.filename_changed.connect(self.on_filename_changed)  # 文件名更改信号连接到处理方法
        self.on_filename_changed()  # 初始化时调用一次

        # 初始化窗口界面
        # 设置应用图标
        # self.setWindowIcon(QIcon(resource_path('resources/icon.ico')))

        # 延迟导入 VLC，创建 VLC 播放器实例
        self.player = VLCPlayer(self)
        self.setCentralWidget(self.player)  # 设置 VLC 播放器为主窗口的中心部件
        self.video_play_changed.connect(self.player.on_video_play_changed)  # 点击timeline中的图片从图片处开始播放

        # 创建其他功能窗口并添加到停靠区域
        self.info = Info(self)  # 信息窗口
        self.addDockWidget(Qt.LeftDockWidgetArea, self.info)  # 停靠到左侧

        self.subtitle = Subtitle(self, self.filename)  # 字幕窗口
        self.addDockWidget(Qt.LeftDockWidgetArea, self.subtitle)
        
        self.control = Control(self, self.filename)  # 控制窗口
        self.addDockWidget(Qt.RightDockWidgetArea, self.control)# 停靠到右侧
        self.control.image_save = self.image_save  # 传递image_save属性

        self.analyze = Analyze(self, self.filename)  # 分析窗口
        self.addDockWidget(Qt.RightDockWidgetArea, self.analyze)

        self.timeline = Timeline(self)  # 时间轴窗口
        self.addDockWidget(Qt.BottomDockWidgetArea, self.timeline)# 停靠到下侧

        # 创建菜单栏
        menu = self.menuBar()

        # 文件菜单
        file_menu = menu.addMenu('&File')  # 添加文件菜单
        open_action = QAction('&Open', self)
        open_action.triggered.connect(lambda: self.player.open_file())  # 打开视频
        exit_action = QAction('&Exit', self)
        exit_action.triggered.connect(lambda: self.close()) # 关闭软件
        file_menu.addAction(open_action)  # 添加到文件菜单
        file_menu.addSeparator()  # 添加分隔线
        file_menu.addAction(exit_action)

        # 帮助菜单
        help_menu = menu.addMenu('&Help')  # 添加帮助菜单
        manual_action = QAction('&Manual', self)  # 使用说明菜单项
        manual_action.triggered.connect(
            lambda: QMessageBox.about(
                self,
                'PyCinemetrics V2.0',
                ' 1-Open Video Play(VLC)\n'
                ' 2-ShotCut(TransnetV2)\n'
                ' 3-Color Analyze(Kmeans)\n'
                ' 4-Subtitle(Whisper)\n'
                ' 5-Object Detection(GIT-base)\n'
                ' 6-Field of view(OpenPose)\n'
            )
        )
        about_action = QAction('&About', self)  # 关于菜单项
        about_action.triggered.connect(
            lambda: QMessageBox.about(self, 'PyCinemetrics', 'PyCinemetrics V2.0 \nHttp://movie.yingshinet.com')
        )
        help_menu.addAction(manual_action)  # 添加使用说明菜单项
        help_menu.addAction(about_action)  # 添加关于菜单项

        # 状态栏
        status_bar = self.statusBar()
        self.progressBar = QProgressBar()  # 进度条
        status_bar.showMessage('')  # 显示空信息
        status_bar.addPermanentWidget(self.progressBar)  # 在状态栏添加进度条
        self.progressBar.hide()  # 默认隐藏进度条

        # 设置窗口尺寸和位置
        app = QApplication.instance()
        screen = app.primaryScreen()
        geometry = screen.availableGeometry()  # 获取屏幕可用区域
        self.setGeometry(
            int(geometry.width() * 0.1),   # 距屏幕左边的距离
            int(geometry.height() * 0.1),  # 距屏幕上方的距离
            int(geometry.width() * 0.8),   # 窗口宽度
            int(geometry.height() * 0.8)   # 窗口高度
        )
        self.showMaximized()  # 窗口最大化

        # 调整停靠窗口的尺寸
        # 调整高度
        self.resizeDocks([self.info, self.subtitle], [int(self.height() * 0.3), int(self.height() * 0.2)], Qt.Vertical)
        self.resizeDocks([self.control, self.analyze], [int(self.height() * 0.3), int(self.height() * 0.2)], Qt.Vertical)

        # 调整宽度
        self.resizeDocks([self.info, self.subtitle, self.control, self.analyze], [int(self.width() * 0.3)] * 4, Qt.Horizontal)
        # 下方的timeline
        self.resizeDocks([self.timeline],
                         [int(self.height() * 0.5)], Qt.Vertical)  # 设置底部时间轴窗口的高度

    def on_filename_changed(self, filename=None):
        # 文件名改变时更新窗口标题
        if filename is None or filename == '':
            self.setWindowTitle('PyCinemetrics')  # 无文件时设置默认标题
        else:
            self.setWindowTitle('PyCinemetrics - %s' % filename)  # 显示当前文件名
            self.filename = filename
            
            # 创建必要的目录
            base_dir = "./img/" + str(os.path.basename(self.filename)[0:-4])
            self.frame_save = base_dir + "/frame"  # 图片存储路径
            self.image_save = base_dir
            
            # 确保目录存在
            os.makedirs(base_dir, exist_ok=True)
            os.makedirs(self.frame_save, exist_ok=True)
            
            try:
                cap = cv2.VideoCapture(self.filename)
                self.frameCnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
            except Exception as e:
                print(f"Error opening video file: {e}")
                self.frameCnt = 0

def main():
    # 启用 Qt 高分辨率显示支持
    qdarktheme.enable_hi_dpi()
    # 创建 Qt 应用程序实例
    app = QApplication(sys.argv)
    # 设置 QDarkTheme 主题（深色或浅色主题）
    qdarktheme.setup_theme()

    # 创建主窗口实例
    _ = MainWindow()

    # 启动应用程序事件循环
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
