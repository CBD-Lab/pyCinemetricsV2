from PySide6.QtWidgets import (
    QDockWidget, QTextEdit, QVBoxLayout, QWidget
)
from PySide6.QtCore import Qt

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
        self.filename=filename
        self.textSubtitle.setPlainText("Subtitle…… ")
