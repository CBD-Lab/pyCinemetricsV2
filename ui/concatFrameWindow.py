import sys
import os
import re
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout, QFileDialog, QPushButton, QWidget
from PIL import Image

class ConcatFrameWindow(QDialog):
    def __init__(self, folder_path, st, ed, parent=None):
        super().__init__(parent)

        self.folder_path = folder_path
        self.parent = parent
        self.st = st
        self.ed = ed

        self.setWindowTitle("Concated Frame")
        # 设置窗口初始大小
        # self.resize(800, 600)  # 调整窗口大小，800x600为默认尺寸

        # 处理文件夹中的图片
        self.process_images()

        # 显示拼接后的图片
        self.show_image_in_dialog()

    def resize_and_pad(self, image_path, target_width, target_height):
        try:
            print(image_path)
            image = Image.open(image_path)
            # 获取原始图片尺寸
            width, height = image.size
            # 计算宽高比
            aspect_ratio = width / height

            # 计算缩放比例
            if aspect_ratio > target_width / target_height:
                # 按宽度缩放
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                # 按高度缩放
                new_height = target_height
                new_width = int(target_height * aspect_ratio)

            # 调整图片大小
            image = image.resize((new_width, new_height))

            # 创建目标大小的背景图像，填充为黑色
            new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
            # 计算放置图片的位置，将其放置在中心
            left = (target_width - new_width) // 2
            top = (target_height - new_height) // 2
            new_image.paste(image, (left, top))
        except:
            return
        return new_image

    def process_images(self):
        try:
            # 获取文件夹中的所有图片文件
            image_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # 定义一个正则表达式，用于提取文件名中的数字部分
            frame_pattern = re.compile(r'frame(\d+)\.png')

            # 筛选出编号在 [st, ed] 范围内的图片路径
            images = [
                self.resize_and_pad(os.path.join(self.folder_path, img), 48 * 3, 27 * 3) for img in image_files
                if frame_pattern.match(img) and self.st <= int(frame_pattern.match(img).group(1)) <= self.ed
            ]

            # images = [self.resize_and_pad(image) for image in images]
            # images = [
            #     self.resize_and_pad(Image.open(os.path.join(self.folder_path, img)), 48 * 3, 27 * 3, padding_color=(0, 0, 0)) for img in image_files
            #     if frame_pattern.match(img) and self.st <= int(frame_pattern.match(img).group(1)) <= self.ed
            # ]
            print(f"images_len:{len(images)}")

            
            # 设置每行的图片数量
            images_per_row = self.parent.frameConcatValue
            
            # 拼接图像
            rows = []
            for i in range(0, len(images), images_per_row):
                row_images = images[i:i + images_per_row]
                row_width = sum(img.width for img in row_images)
                row_height = max(img.height for img in row_images)
                row = Image.new("RGB", (row_width, row_height))
                
                x_offset = 0
                for img in row_images:
                    row.paste(img, (x_offset, 0))
                    x_offset += img.width
                
                rows.append(row)
            
            # 总拼接图像高度
            total_height = sum(row.height for row in rows)

            self.final_image = Image.new("RGB", (rows[0].width, total_height))
            
            y_offset = 0
            for row in rows:
                self.final_image.paste(row, (0, y_offset))
                y_offset += row.height
            
            # 保存拼接后的图片
            self.Concated_frame_path = os.path.join(self.parent.image_save, "Concated_frame.png")
            self.final_image.save(self.Concated_frame_path)
        except:
            return

    def show_image_in_dialog(self):
        try:
            # 创建标签来显示图片
            label = QLabel(self)
            pixmap = QPixmap(self.Concated_frame_path)

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
                    # 如果图片的宽度小于或等于1600，保持原尺寸
                    scaled_pixmap = pixmap
            label.setPixmap(scaled_pixmap)
            # label.setPixmap(pixmap.scaled(600, 400, Qt.KeepAspectRatio))
            label.setAlignment(Qt.AlignCenter)

            # 设置布局
            layout = QVBoxLayout()
            layout.addWidget(label)
            self.setLayout(layout)
        except:
            return


# class MainWindow(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Image Combiner")
#         self.setGeometry(300, 100, 400, 200)

#         # 布局
#         layout = QVBoxLayout()
#         self.button = QPushButton("选择文件夹并拼接图片")
#         self.button.clicked.connect(self.show_dialog)
#         layout.addWidget(self.button)
        
#         self.setLayout(layout)

#     def show_dialog(self):
#         folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
#         if folder_path:
#             self.open_concat_window(folder_path)

#     def open_concat_window(self, folder_path):
#         self.concat_window = ConcatFrameWindow(folder_path, self)
#         self.concat_window.exec_()


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())
