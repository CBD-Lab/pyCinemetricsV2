from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import (
    QDockWidget, QLabel, QDialog, QVBoxLayout, QTabWidget, QWidget, QHBoxLayout, QPushButton
)
from PySide2.QtCore import Qt


class Analyze(QDockWidget):
    def __init__(self,parent,filename):
        super().__init__('Analyze', parent)
        self.parent = parent
        self.filename = filename
        self.parent.filename_changed.connect(self.on_filename_changed)
        self.init_analyze()

    def init_analyze(self):
        # 使用 QTabWidget 来支持多个标签页
        self.tab_widget = QTabWidget(self)
        self.setWidget(self.tab_widget)  # 将 QTabWidget 设置为 QDockWidget 的中央部件


    def on_filename_changed(self):
        # 清空当前显示的所有标签页
        self.tab_widget.clear()

    def add_tab_with_image(self, tab_name, image_path):
        # 创建一个标签页来显示图片
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return  # 如果图片为空，则返回

        # 创建标签并设置图片
        label = QLabel(self)
        label.setPixmap(pixmap.scaled(300, 250, Qt.KeepAspectRatio))  # 设置缩放后的图片大小
        label.mousePressEvent = lambda event: self.on_analyze_image_click(event, image_path)  # 支持点击图片刷新

        # 创建一个布局并将标签添加到布局中
        layout = QVBoxLayout()

        # 创建关闭按钮并添加到布局
        close_button_layout = QHBoxLayout()
        close_button = QPushButton("❎", self)
        close_button.setFixedSize(30, 30)
        close_button.clicked.connect(lambda: self.close_tab())
        close_button_layout.addWidget(close_button, alignment=Qt.AlignRight)

        # 将关闭按钮布局添加到垂直布局的顶部
        layout.addLayout(close_button_layout)

        # 将图片标签添加到垂直布局中
        layout.addWidget(label)

        # 创建一个 QWidget 作为标签页的内容
        tab_widget_content = QWidget(self)
        tab_widget_content.setLayout(layout)

        # 添加标签页到 QTabWidget
        self.tab_widget.addTab(tab_widget_content, tab_name)

        # 自动聚焦到当前新建的标签页
        self.tab_widget.setCurrentIndex(self.tab_widget.count() - 1)

    def close_tab(self):
        # 获取当前选中的标签页并移除
        current_index = self.tab_widget.currentIndex()
        if current_index != -1:
            self.tab_widget.removeTab(current_index)

    def on_analyze_image_click(self, event, image_path):
        # print(image_path)
        if event.button() == Qt.LeftButton:
            # 获取原始图片
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # 获取原始宽度和高度
                original_width = pixmap.width()
                original_height = pixmap.height()

                # 如果图片的宽度大于1600，按比例缩小
                target_width = 1600
                if original_width > target_width:
                    # 根据目标宽度计算新的高度，保持长宽比
                    target_height = int((original_height * target_width) / original_width)
                    # 缩放图像，保持比例
                    scaled_pixmap = pixmap.scaled(target_width, target_height, Qt.KeepAspectRatio)
                else:
                    # 如果图片的宽度小于或等于800，保持原尺寸
                    scaled_pixmap = pixmap
                self.show_zoomed_image(scaled_pixmap)
    
    def show_zoomed_image(self, pixmap):
        # 创建一个新的对话框来显示放大后的图片
        zoomed_dialog = QDialog(self)
        layout = QVBoxLayout()
        zoomed_label = QLabel()
        zoomed_label.setPixmap(pixmap)
        layout.addWidget(zoomed_label)
        zoomed_dialog.setLayout(layout)
        zoomed_dialog.setWindowTitle('Zoomed Image')
        zoomed_dialog.exec_()  # 进入对话框的主事件循环
