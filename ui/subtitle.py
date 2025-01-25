import os
import csv
from PySide6.QtWidgets import (
    QDockWidget, QTextEdit, QVBoxLayout, QWidget
)
from PySide6.QtCore import Qt
from pathlib import Path

class Subtitle(QDockWidget):
    def __init__(self, parent,filename):
        super().__init__('Subtitle', parent)
        self.parent = parent
        self.filename=filename
        self.parent.filename_changed.connect(self.on_filename_changed)
        self.init_subtitle()
    def init_subtitle(self):
        # self.textSubtitle = QTextEdit(
        #     "Subtitle…… ", self)
        # self.textSubtitle.setGeometry(10, 30, 300, 300)
        # Create the text edit widget and set it as the central widget of the dock
        self.textSubtitle = QTextEdit(self)
        self.textSubtitle.setPlainText("Subtitle…… ")

        # Create a layout for the dock widget
        layout = QVBoxLayout()
        layout.addWidget(self.textSubtitle)

        # Create a QWidget for the content and set the layout
        widget = QWidget(self)
        widget.setLayout(layout)

        # Set the QWidget as the main widget of the QDockWidget
        self.setWidget(widget)


    def on_filename_changed(self,filename):
        # 当文件名发生变化时调用该方法
        if filename is None or filename == '':  # 如果文件名无效，则不做任何处理
            return
        self.showSubtitle(Path(filename).resolve().stem)

    def read_subtitle(self, filename, col):
        # 用于存储最终的所有文本
        data = ""
        try:
            with open(filename, 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                # 跳过第一行（列名）
                next(csv_reader)
                for row in csv_reader:
                    if len(row) > col:  # 确保目标列存在
                        # 累加每行的目标列内容，并加上换行符
                        data += row[col].strip() + "\n"
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return
        return data
    
    def showSubtitle(self, name):
        if name is None or name == '':  # 如果没有提供有效的名称，则不执行任何操作
            return
        
        subtitle_path = os.path.join(self.parent.image_save, 'subtitle.csv')
        if os.path.exists(subtitle_path):
            self.textSubtitle.setPlainText(self.read_subtitle(subtitle_path, 2))
        else:
            self.textSubtitle.setPlainText("Subtitle…… ")

