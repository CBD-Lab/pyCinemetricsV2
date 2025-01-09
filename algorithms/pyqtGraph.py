import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QDialog, QPushButton, QVBoxLayout, QHBoxLayout, QColorDialog
from pyqtgraph.Qt import QtWidgets


class TransNetPlot(QDialog):
    def __init__(self, image_save_path=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Shot Length Visualization")
        self.setFixedSize(1280, 720)  # 设置固定窗口大小
        self.image_save_path = image_save_path

        # 主布局：垂直布局，上部分为图形区域，下部分为按钮区域
        self.main_layout = QVBoxLayout(self)

        # 上部分：图形区域
        self.win = pg.GraphicsLayoutWidget(title="Shot Length Visualization")
        self.win.setWindowTitle("Shot Length Plot")
        self.win.setBackground('black')  # 默认背景色为黑色

        # 添加 plot 项
        self.plot = self.win.addPlot(title="Shot Length", row=0, col=0,font=pg.QtGui.QFont('Arial', 30))
        self.plot.setLabel('left', 'Shot Length')
        self.plot.setLabel('bottom', 'Shot ID')

        # 将图形窗口添加到主布局的上部分
        self.main_layout.addWidget(self.win)

        # 下部分：按钮区域
        self.button_layout = QHBoxLayout()

        # 左侧：切换背景颜色的按钮
        self.left_button_layout = QVBoxLayout()
        self.button1 = QPushButton("Change Background Color")  # 修改按钮文本
        self.left_button_layout.addWidget(self.button1)

        # 右侧：改变颜色的按钮
        self.right_button_layout = QVBoxLayout()
        self.color_button = QPushButton("Change Bar Color")
        self.right_button_layout.addWidget(self.color_button)

        # save按钮
        self.save_img_button_layout = QVBoxLayout()
        self.save_img_button = QPushButton("Save and overwrite")
        self.save_img_button_layout.addWidget(self.save_img_button)

        # 将左右两部分的按钮布局合并到下部分布局
        self.button_layout.addLayout(self.left_button_layout)
        self.button_layout.addLayout(self.right_button_layout)
        self.button_layout.addLayout(self.save_img_button_layout)

        # 将按钮区域添加到主布局的下部分
        self.main_layout.addLayout(self.button_layout)

        # 默认条形图颜色
        self.bar_color = 'blue'

        # 按钮点击连接到相应的函数
        self.button1.clicked.connect(self.change_background_color)  # 改为调用 change_background_color 方法
        self.color_button.clicked.connect(self.update_bar_color)
        self.save_img_button.clicked.connect(self.save_img)

        # 存储数据
        self.shot_len_data = []  # 用于存储图表数据
        self.bg = None

    def plot_transnet(self, shot_len):
        """绘制图表"""
        # 提取 shot ID 和 shot 长度
        shot_id = np.arange(len(shot_len))
        shot_length = np.array([shot[2] for shot in shot_len])

        # 存储数据
        self.shot_len_data = list(zip(shot_id, shot_length))

        # 创建条形图
        self.bg = pg.BarGraphItem(x=shot_id, height=shot_length, width=0.8, brush=self.bar_color)
        self.plot.addItem(self.bg)

        # 可选：导出图表到文件
        if self.image_save_path:
            img = self.win.grab()  # 捕获整个窗口
            img.save(self.image_save_path + '/shotlen.png')

    def update_bar_color(self):
        """更新条形图颜色"""
        color = QColorDialog.getColor()
        if color.isValid():
            self.bar_color = color.name()
            self.plot.clear()  # 清除当前图表
            shot_id, shot_length = zip(*self.shot_len_data)
            self.bg = pg.BarGraphItem(x=shot_id, height=shot_length, width=0.8, brush=self.bar_color)
            self.plot.addItem(self.bg)

    def change_background_color(self):
        """切换背景颜色"""
        color = QColorDialog.getColor()
        if color.isValid():
            self.win.setBackground(color.name())  # 更新背景色为用户选择的颜色

    def save_img(self):
        # 可选：导出图表到文件
        if self.image_save_path:
            img = self.win.grab()  # 捕获整个窗口
            img.save(self.image_save_path + '/shotlen.png')


# if __name__ == "__main__":
#     app = QtWidgets.QApplication([])

#     plot_window = TransNetPlot(image_save_path="./img")
#     plot_window.show()

#     app.exec_()
