import os
from PySide2.QtCore import Qt
from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import QApplication, QDialog, QTabWidget, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QPushButton

class ImageDialog(QDialog):
    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        
        # 初始化 QTabWidget
        self.tab_widget = QTabWidget(self)
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        
        # 设置对话框布局
        layout = QVBoxLayout(self)
        layout.addWidget(self.tab_widget)
        
        # 根据提供的图像路径列表添加标签页
        for i, image_path in enumerate(image_paths):
            tab_name = os.path.basename(image_path)[0:-4]
            self.add_tab_with_image(tab_name, image_path)
        
        self.setLayout(layout)
        self.setWindowTitle("Image Viewer")
        # # 设置窗口初始大小
        # self.resize(800, 600)  # 调整窗口大小，800x600为默认尺寸

    def add_tab_with_image(self, tab_name, image_path):
        # 创建一个标签页来显示图片
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return  # 如果图片为空，则返回

        # 创建标签并设置图片
        label = QLabel(self)
        label.setPixmap(pixmap)  # 设置缩放后的图片大小

        # 创建布局并将标签添加到布局中
        layout = QVBoxLayout()

        # 将图片标签添加到垂直布局中
        layout.addWidget(label)

        # 创建一个 QWidget 作为标签页的内容
        tab_widget_content = QWidget(self)
        tab_widget_content.setLayout(layout)

        # 添加标签页到 QTabWidget
        self.tab_widget.addTab(tab_widget_content, tab_name)

        # # 自动聚焦到当前新建的标签页
        # self.tab_widget.setCurrentIndex(self.tab_widget.count() - 1)
      
    def close_tab(self):
        # 获取当前选中的标签页并移除
        current_index = self.tab_widget.currentIndex()
        if current_index != -1:
            self.tab_widget.removeTab(current_index)
