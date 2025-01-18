import os
import shutil
import subprocess
import cv2
import csv
import math
import numpy as np
import chardet
import pandas as pd
from insightface.app import FaceAnalysis#注意导入的顺序否则可能会报错
from sklearn.cluster import DBSCAN
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLabel, QScrollArea, QLineEdit, QMessageBox, QFileDialog, QInputDialog, QDialog, QGridLayout, QComboBox
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QTimer
from ui.progressBar import *

class FaceDetection(QThread):
    signal = Signal(int, int, int, str)  # 进度更新信号
    finished = Signal(bool)        # 任务完成信号
    is_stop = 0                    # 是否中断标志

    def __init__(self, image_dir, output_dir):
        super(FaceDetection, self).__init__()
        self.image_dir = image_dir
        self.output_dir = output_dir

    def run(self):
        app = self.initialize_model()
        image_paths = [os.path.join(self.image_dir, img) for img in os.listdir(self.image_dir) if img.endswith(('.jpg', '.png'))]
        
        # 提取特征
        features, face_images, original_images, face_poses, eye_statuses = self.extract_features(app, image_paths)
        if len(features) == 0:
            print("No faces detected!",image_paths)
            self.finished.emit(False)
            return
        # 聚类
        labels = self.cluster_faces(features)
        
        # 保存最终结果
        code = self.save_final_faces(labels, face_images, original_images, face_poses, eye_statuses, self.output_dir)
        if code == 404:
            self.finished.emit(False)
        else:
            self.finished.emit(True)
            print(len(features))
            print(f"Clustering and selection complete! Results saved in {self.output_dir}")
        
    # 初始化模型
    def initialize_model(self):
        app = FaceAnalysis(root = "../models", providers=['CPUExecutionProvider'])  # 使用 CPU
        app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id = -1 强制使用 CPU
        return app

    # 计算人脸检测框面积
    def calculate_bbox_area(self, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        return (x2 - x1) * (y2 - y1)

    # 计算亮度得分
    def calculate_brightness_score(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        face_region = image[y1:y2, x1:x2]  # 裁剪人脸区域
        if face_region.size == 0:
            return 0  # 如果区域为空，返回 0
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)  # 转为灰度图
        return np.mean(gray)  # 返回灰度均值（亮度得分）

    # 判断是否睁开眼睛（假设支持关键点，需结合 EAR）
    def calculate_ear(self, eye_points):
        # 计算 EAR
        A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        ear = (A + B) / (2.0 * C)
        return ear

    def is_eyes_open(self, landmarks, eye_threshold=0.2):
        left_eye = landmarks[36:42]  # 左眼关键点
        right_eye = landmarks[42:48]  # 右眼关键点
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        return left_ear > eye_threshold and right_ear > eye_threshold

    # 判断是否为正脸
    def is_frontal_face(self, pose, yaw_threshold=15, pitch_threshold=15):
        yaw, pitch, roll = pose
        return abs(yaw) < yaw_threshold and abs(pitch) < pitch_threshold

    # 计算检测框对角线长度
    def calculate_diagonal(self, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # 生成圆形照片
    def create_circular_crop(self, image, bbox):
        """
        对图片进行圆形裁剪，并返回只有圆形区域内容的图像，其他部分为透明
        :param image: 输入图像 (numpy array)
        :param bbox: 边界框 (x1, y1, x2, y2)
        :return: 纯圆形裁剪图像 (带透明背景)
        """
        x1, y1, x2, y2 = map(int, bbox)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius = int(self.calculate_diagonal((x1, y1, x2, y2)) / 2)

        # 创建圆形掩码
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)

        # 提取圆形区域
        result = cv2.bitwise_and(image, image, mask=mask)

        # 创建一个带透明度的输出图像 (4 通道)
        size = 2 * radius
        circular_image = np.zeros((size, size, 4), dtype=np.uint8)  # 4 通道，支持透明

        # 将裁剪区域添加到画布中
        circular_region = result[center_y - radius:center_y + radius, center_x - radius:center_x + radius]
        circular_image[:circular_region.shape[0], :circular_region.shape[1], :3] = circular_region  # RGB 通道

        # 设置透明背景
        mask_region = mask[center_y - radius:center_y + radius, center_x - radius:center_x + radius]
        circular_image[:circular_region.shape[0], :circular_region.shape[1], 3] = mask_region  # Alpha 通道

        return circular_image

    # 提取人脸特征
    def extract_features(self, app, image_paths):
        features = []
        face_images = []
        original_images = []
        face_poses = []  # 保存姿态信息
        eye_statuses = []  # 保存眼睛状态

        # 进度条设置
        total_number = len(image_paths)  # 总任务数
        task_id = 0  # 子任务序号

        for img_path in image_paths:
            if self.is_stop:
                break
            img = cv2.imread(img_path)
            if img is None:
                continue
            faces = app.get(img)
            for face in faces:
                features.append(face.normed_embedding)  # 提取特征
                face_images.append(face.bbox)         # 获取人脸边框
                original_images.append((img, img_path))  # 原图及其路径保存
                face_poses.append(face.pose)          # 保存姿态信息
                eye_statuses.append(self.is_eyes_open(face.landmark_2d_106))  # 判断眼睛状态
            task_id += 1
            percent = round(float(task_id / total_number) * 100)
            self.signal.emit(percent, task_id, total_number, "faceDetect")  # 发送实时任务进度和总任务进度
        self.signal.emit(101, 101, 101, "faceDetect")  # 完成后发送信号
        return np.array(features), face_images, original_images, face_poses, eye_statuses

    # 聚类人脸
    def cluster_faces(self, features, eps=0.6, min_samples=2):
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(features)
        return labels

    # 保存最终结果
    def save_final_faces(self, labels, face_images, original_images, face_poses, eye_statuses, output_dir):
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # 移除噪声类别

        if not unique_labels:
            print("No valid clusters found. Exiting save_final_faces.")
            return 404 # 如果没有有效的聚类，直接退出

        csv_data = []
        # 计算每个聚类的人脸数量
        label_counts = {label: np.sum(labels == label) for label in unique_labels}
        print('label_counts:', label_counts)

        # 获取最大值，并计算最小阈值
        max_count = max(label_counts.values())
        min_threshold = max_count / 10  # 小于最大值的十分之一

        # 过滤掉数量小于阈值的聚类
        filtered_labels = {label: count for label, count in label_counts.items() if count >= min_threshold}
        print('Filtered label counts:', filtered_labels)

        for label, count in filtered_labels.items():
            cluster_faces = []  # 保存当前聚类中的所有人脸信息
            for idx, lbl in enumerate(labels):
                if lbl == label:
                    cluster_faces.append({
                        "image": original_images[idx][0],
                        "path": original_images[idx][1],
                        "bbox": face_images[idx],
                        "brightness": self.calculate_brightness_score(original_images[idx][0], face_images[idx]),
                        "area": self.calculate_bbox_area(face_images[idx]),
                        "eyes_open": eye_statuses[idx],
                        "is_frontal": self.is_frontal_face(face_poses[idx])
                    })

            # 按优先级排序：睁眼 > 检测框面积 > 亮度 > 正脸
            cluster_faces.sort(
                key=lambda x: (
                    x["eyes_open"],  # 是否睁眼
                    x["area"],       # 检测框面积
                    x["brightness"], # 亮度
                    x["is_frontal"]  # 是否正脸
                ),
                reverse=True
            )
            # 保存优先级最高的两张图片
            top_faces = cluster_faces[:2]

            cluster_dir = output_dir

            for i, face in enumerate(top_faces):
                img_name = f"top_{i+1}_" + os.path.basename(face["path"])
                output_path = os.path.join(cluster_dir, img_name)
                x1, y1, x2, y2 = map(int, face["bbox"])
                img_copy = face["image"].copy()
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img_copy,
                    f"Cluster {label} - Top {i+1}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                # cv2.imwrite(output_path, img_copy)
                print(f"Saved top {i+1} face for Cluster {label} to {output_path}")

            # 保存圆形照片（优先级最高的图片）
            top_face = top_faces[0]
            circular_crop = self.create_circular_crop(top_face["image"], top_face["bbox"])
            circular_output_path = os.path.join(cluster_dir, f"FaceCluster_{label}.png")
            cv2.imwrite(circular_output_path, circular_crop)

            print(f"Saved circular cropped image for Cluster {label} to {circular_output_path}")

            # 收集当前聚类的图片名称最后数字
            image_numbers = [
                int(os.path.basename(face["path"]).split('_')[-1].split('.')[0])
                for face in cluster_faces
            ]

            # 统计人脸数量并保存到 CSV 数据中
            csv_data.append([
                f"FaceCluster_{label}",  # Name
                count,                  # Count
                int(os.path.basename(cluster_faces[0]["path"]).split('_')[2]) + 1,  # ShotNumber
                sorted(image_numbers)   # 图片编号集合
            ])

        # 保存 CSV 文件
        csv_output_path = os.path.join(output_dir, "face_cluster_statistics.csv")
        try:
            df = pd.DataFrame(csv_data, columns=["Name", "Count", "ShotNumber", "Frames"])
            df.to_csv(csv_output_path, index=False)
            print(f"Saved face cluster statistics to {csv_output_path}")
        except PermissionError:
            return 404

    def stop(self):
        self.is_stop = 1



class ImageDialog(QDialog):
    """显示放大的图片，支持以鼠标位置为中心缩放"""
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Viewer")
        self.setGeometry(300, 300, 800, 800)

        # 初始化缩放比例
        self.scale_factor = 1.0
        self.image_path = image_path

        # 主布局
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # 滚动区域
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

        # 图片标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)

        # 加载图片
        self.original_pixmap = QPixmap(self.image_path)
        if self.original_pixmap.isNull():
            raise FileNotFoundError(f"无法加载图片：{self.image_path}")
        self.load_image()

    def load_image(self):
        """加载并显示图片，根据当前缩放比例调整大小"""
        if self.original_pixmap:
            scaled_pixmap = self.original_pixmap.scaled(
                self.original_pixmap.size() * self.scale_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def wheelEvent(self, event):
        """鼠标滚轮事件，用于以鼠标位置为中心缩放图片"""
        if event.angleDelta().y() > 0:
            self.scale_factor *= 1.1  # 放大
        else:
            self.scale_factor /= 1.1  # 缩小

        # 限制缩放比例
        self.scale_factor = max(0.1, min(self.scale_factor, 5.0))
        self.load_image()


class ClickableImageLabel(QLabel):
    """可点击的图片标签"""
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            dialog = ImageDialog(self.image_path, self)
            dialog.exec_()


class ImageDialog(QDialog):
    """显示放大的图片，支持以鼠标位置为中心缩放"""
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Viewer")
        self.setGeometry(300, 300, 800, 600)

        # 初始化缩放比例
        self.scale_factor = 1.0
        self.image_path = image_path

        # 主布局
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # 滚动区域
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

        # 图片标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)

        # 加载图片
        self.original_pixmap = QPixmap(self.image_path)
        if self.original_pixmap.isNull():
            raise FileNotFoundError(f"无法加载图片：{self.image_path}")
        self.load_image()

    def load_image(self):
        """加载并显示图片，根据当前缩放比例调整大小"""
        if self.original_pixmap:
            scaled_pixmap = self.original_pixmap.scaled(
                self.original_pixmap.size() * self.scale_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def wheelEvent(self, event):
        """鼠标滚轮事件，用于以鼠标位置为中心缩放图片"""
        # 获取鼠标位置并转换为整数坐标
        mouse_position = event.position().toPoint() - self.scroll_area.geometry().topLeft()
        old_scroll_values = self.scroll_area.horizontalScrollBar().value(), self.scroll_area.verticalScrollBar().value()

        # 判断缩放方向
        if event.angleDelta().y() > 0:
            self.scale_factor *= 1.1  # 放大
        else:
            self.scale_factor /= 1.1  # 缩小

        # 限制缩放比例
        self.scale_factor = max(0.1, min(self.scale_factor, 5.0))
        self.load_image()

        # 调整滚动条位置以实现以鼠标为中心的缩放
        new_pixmap_size = self.original_pixmap.size() * self.scale_factor
        delta_x = (new_pixmap_size.width() - self.image_label.pixmap().width()) / 2
        delta_y = (new_pixmap_size.height() - self.image_label.pixmap().height()) / 2

        self.scroll_area.horizontalScrollBar().setValue(
            old_scroll_values[0] + delta_x * (mouse_position.x() / self.scroll_area.width())
        )
        self.scroll_area.verticalScrollBar().setValue(
            old_scroll_values[1] + delta_y * (mouse_position.y() / self.scroll_area.height())
        )



class MappingApp(QDialog):
    def __init__(self, image_folder, input_images_dir, parent):
        super().__init__()
        self.image_folder = image_folder
        self.input_images_dir = input_images_dir
        self.imageshows = {}  # 存储文件夹的原始名称和对应的编辑框
        self.init_ui()
        self.parent = parent

    def init_ui(self):
        self.setWindowTitle("Editable Folder Mapping with Save and Merge")
        self.setFixedWidth(1200)  # 固定窗口宽度

        # 主布局
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)  # 设置为 QDialog 的主布局

        # 顶部按钮区域：增加一个 "Run Recognition" 按钮
        top_layout = QHBoxLayout()
        top_layout.setAlignment(Qt.AlignLeft)

        # 添加 "Run Recognition" 按钮
        run_recognition_button = QPushButton("Face Recognition")
        run_recognition_button.setFixedWidth(150)
        
        # 创建下拉框并设置内容
        self.combo_box = QComboBox()
        self.combo_box.addItem("frame")  # 添加选项1
        self.combo_box.addItem("ImagetoText")  # 添加选项2
        
        # 按钮点击事件处理
        run_recognition_button.clicked.connect(self.run_recognition)
        
        # 将按钮和下拉框添加到布局
        top_layout.addWidget(run_recognition_button)
        top_layout.addWidget(self.combo_box)


        # 添加 "打开文件夹目录" 按钮
        open_dir_button = QPushButton("Open Folder")
        open_dir_button.setFixedWidth(150)
        open_dir_button.clicked.connect(self.open_directory)
        top_layout.addWidget(open_dir_button)

        #展示字幕
        show_subtitle_button = QPushButton("Show MetaData/Credits")
        show_subtitle_button.setFixedWidth(200)
        show_subtitle_button.clicked.connect(self.show_subtitle)
        top_layout.addWidget(show_subtitle_button)

        # 添加 "Generate New CSV" 按钮
        # generate_csv_button = QPushButton("Update CSV")
        # generate_csv_button.setFixedWidth(150)
        # generate_csv_button.clicked.connect(self.generate_new_csv)
        # top_layout.addWidget(generate_csv_button)

        # 将顶部布局添加到主布局
        main_layout.addLayout(top_layout)

        # 滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        content_widget = QWidget()
        self.grid_layout = QGridLayout()  # 使用网格布局替代垂直布局
        self.grid_layout.setSpacing(10)  # 设置控件间的间距
        content_widget.setLayout(self.grid_layout)
        self.scroll_area.setWidget(content_widget)
        main_layout.addWidget(self.scroll_area)

        # 底部按钮区域
        bottom_layout = QHBoxLayout()
        bottom_layout.setAlignment(Qt.AlignLeft)

        # 添加 "保存" 按钮
        save_button = QPushButton("Update Changes")
        save_button.setFixedWidth(150)
        save_button.clicked.connect(self.save_changes)
        bottom_layout.addWidget(save_button)

        # 添加 "刷新" 按钮
        refresh_button = QPushButton("Refresh")
        refresh_button.setFixedWidth(150)
        refresh_button.clicked.connect(lambda: self.refresh_images(True))
        bottom_layout.addWidget(refresh_button)

        # 将底部布局添加到主布局
        main_layout.addLayout(bottom_layout)

        # 加载文件夹
        self.load_images()

    def load_images(self):
        """加载图片并创建可编辑控件。"""
        if not os.path.exists(self.image_folder):
            return
        images = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            self.run_recognition()

        self.clear_grid_layout()  # 清除现有网格布局内容

        row, col = 0, 0
        for image_name in images:
            # 创建单元格容器
            container = QWidget()
            container_layout = QVBoxLayout()  # 使用垂直布局排列控件
            container.setLayout(container_layout)
            container_layout.setAlignment(Qt.AlignCenter)

            # 显示缩略图
            image_path = os.path.join(self.image_folder, image_name)
            pixmap = QPixmap(image_path).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label = ClickableImageLabel(image_path)
            label.setPixmap(pixmap)
            container_layout.addWidget(label)

            # 水平布局：用于放置文本框和删除按钮
            horizontal_layout = QHBoxLayout()
            horizontal_layout.setAlignment(Qt.AlignLeft)

            # 可编辑的文本框
            name_edit = QLineEdit(image_name)
            name_edit.setFixedWidth(200)
            name_edit.textChanged.connect(lambda text, btn=name_edit: self.clear_red_border(btn))
            horizontal_layout.addWidget(name_edit)

            # 删除图片按钮，放在文本框右侧
            delete_button = QPushButton("Delete")
            delete_button.setFixedWidth(70)
            delete_button.clicked.connect(lambda checked=False, image_name=image_name: self.delete_image(image_name))
            horizontal_layout.addWidget(delete_button)

            # 将水平布局添加到垂直布局中
            container_layout.addLayout(horizontal_layout)

            # 将容器添加到网格布局
            self.grid_layout.addWidget(container, row, col)
            self.imageshows[image_name] = name_edit

            # 控制布局为四列
            col += 1
            if col == 4:  # 每行四个单元后换行
                col = 0
                row += 1


    def clear_grid_layout(self):
        """清除网格布局中的所有控件。"""
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def run_recognition(self):
        """运行人脸识别"""
        # 获取下拉框的选中项
        selected_option = self.combo_box.currentText()
        image_dir = os.path.dirname(self.input_images_dir) + "//" +selected_option

        # 禁用按钮，防止再次点击
        button = self.sender()  # 获取触发信号的按钮
        if button is not None:
            button.setEnabled(False)

        # 检测 image_dir 是否存在
        if not os.path.exists(image_dir):
            # 如果目录不存在，给出错误提示
            print(f"Error: The directory '{image_dir}' does not exist.")
            QMessageBox.warning(self, "Refresh Failed", f"Error: The directory '{image_dir}' does not exist.")
            # 重新启用按钮
            if button is not None:
                button.setEnabled(True)
            return

        facedetection = FaceDetection(image_dir, self.image_folder)
        bar = pyqtbar(facedetection)

        # 连接识别完成的信号，刷新图片显示
        facedetection.finished.connect(self.refresh_images)
        facedetection.finished.connect(self.generate_new_csv)
        # 识别完成后重新启用按钮
        if button is not None:
            facedetection.finished.connect(lambda: button.setEnabled(True))

    def refresh_images(self, success=True):
        """刷新图片显示。"""
        if success:
            print("Detection finished successfully. Refreshing images...")
            # 清空现有图片显示的网格布局
            self.clear_grid_layout()
            self.imageshows.clear()
            self.load_images()  # 重新加载图片
        else:
            print("Detection failed. No images to refresh.")
            QMessageBox.warning(self, "Refresh Failed", "The detection process failed or there are no images to refresh.Please change the target folder for detection to 'ImagetoText' or close the CSV file.")

    def clear_red_border(self, line_edit):
        """清除编辑框的红色边框"""
        line_edit.setStyleSheet("")

    def save_changes(self):
        """保存图片名字的更改，同时更新 face_cluster_statistics.csv 文件。"""
        new_names = {}
        conflicts = []

        # 遍历所有图片的编辑框
        for original_name, edit_box in self.imageshows.items():
            new_name = edit_box.text().strip()
            if new_name in new_names or new_name == "":  # 检测名称冲突或空名称
                conflicts.append(edit_box)
                edit_box.setStyleSheet("border: 2px solid red;")
            else:
                new_names[new_name] = original_name
                edit_box.setStyleSheet("")

        if conflicts:
            QMessageBox.warning(self, "Name Conflict", "Some names are invalid or duplicated. Please resolve the conflicts.")
            return

        # 执行文件重命名
        for new_name, original_name in new_names.items():
            if new_name != original_name:
                original_path = os.path.join(self.image_folder, original_name)
                new_path = os.path.join(self.image_folder, new_name)
                os.rename(original_path, new_path)

        # 更新 face_cluster_statistics.csv 文件
        csv_path = os.path.join(self.image_folder, "face_cluster_statistics.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)

                # 定义允许的图片后缀
                valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

                # 获取文件夹中所有图片名字（去掉后缀，过滤仅限图片）
                existing_images = set(
                    os.path.splitext(file)[0] for file in os.listdir(self.image_folder)
                    if os.path.splitext(file)[1].lower() in valid_extensions
                )
                print(f"Existing images in folder (without extensions): {existing_images}")  # 调试信息

                # 检查 CSV 是否包含 Name 列
                if 'Name' not in df.columns:
                    QMessageBox.warning(self, "Error", "CSV file does not contain 'Name' column.")
                    return
                
                # 更新 CSV 中的 Name 列
                for new_name, original_name in new_names.items():
                    # 去除后缀
                    new_name_no_ext = os.path.splitext(new_name)[0]
                    original_name_no_ext = os.path.splitext(original_name)[0]
                    
                    print('new_name', new_name_no_ext, 'original_name', original_name_no_ext)  # 调试信息
                    
                    if original_name_no_ext in df['Name'].values:
                        print(f"Replacing {original_name_no_ext} with {new_name_no_ext}")  # 调试信息
                        df['Name'] = df['Name'].replace(original_name_no_ext, new_name_no_ext)

                # 删除 CSV 中不存在于文件夹中的图片名字对应的行
                df = df[df['Name'].isin(existing_images)]
                print(f"Filtered DataFrame:\n{df}")  # 调试信息

                # 保存更新后的 CSV
                df.to_csv(csv_path, index=False)
                print(f"Updated {csv_path} with new image names and removed missing entries.")
                self.generate_new_csv()

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to update CSV file: {e}")
                return
        else:
            QMessageBox.warning(self, "Error", f"CSV file not found")
            return
        QMessageBox.information(self, "Success", "Image names and CSV file have been successfully updated!")
        self.refresh_images()

    def delete_image(self, image_name):
        """删除指定图片。"""
        reply = QMessageBox.question(self, "Delete Image", f"Are you sure you want to delete the image '{image_name}'?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            image_path = os.path.join(self.image_folder, image_name)
            os.remove(image_path)
            self.refresh_images()

    def open_directory(self):
        print(f"Trying to open folder: {self.image_folder}")
        if os.path.exists(self.image_folder):
            if os.path.isdir(self.image_folder):
                try:
                    if os.name == 'nt':  # Windows
                        subprocess.Popen(['explorer', os.path.normpath(self.image_folder)])
                    elif os.name == 'posix':  # macOS or Linux
                        command = ['open', self.image_folder] if sys.platform == 'darwin' else ['xdg-open', self.image_folder]
                        subprocess.Popen(command)
                except Exception as e:
                    print(f"Error opening folder: {e}")
                    QMessageBox.warning(self, "Error", f"Failed to open folder: {e}")
            else:
                QMessageBox.warning(self, "Error", "The path is not a valid folder!")
        else:
            QMessageBox.warning(self, "Error", "The folder path does not exist!")
    
    def generate_new_csv(self, success: bool = True):
        if not success:
            print("任务失败，不执行生成新 CSV 的操作。")
            return  # 如果任务失败，则不执行 CSV 生成操作
        """根据原始 CSV 文件生成新的 CSV 文件"""
        # 原始 CSV 文件路径
        csv_path = os.path.join(self.image_folder, "face_cluster_statistics.csv")
        if not os.path.exists(csv_path):
            QMessageBox.warning(self, "Error", "Original CSV file not found.")
            return

        try:
            # 读取原始 CSV 文件
            df = pd.read_csv(csv_path)

            # 创建新的 DataFrame
            new_data = []

            # 遍历原始数据的每一行
            for _, row in df.iterrows():
                name = row["Name"]
                shot_number = row["ShotNumber"]
                frames = eval(row["Frames"])  # 将帧号列表字符串转换为实际列表

                for frame in frames:
                    # 检查当前帧是否已经在 new_data 中
                    existing_entry = next((entry for entry in new_data if entry["Frames"] == frame), None)

                    if existing_entry:
                        # 如果当前帧已经存在，追加 Name 和 ShotNumber
                        existing_entry["Names"].append(name)
                    else:
                        # 如果当前帧不存在，创建新条目
                        new_data.append({"Frames": frame, "Names": [name], "ShotNumber": shot_number})

            # 按帧号排序
            new_data = sorted(new_data, key=lambda x: x["Frames"])

            # 转换为 DataFrame
            new_df = pd.DataFrame([
                {"Frames": entry["Frames"], "Names": ", ".join(entry["Names"]), "ShotNumber": entry["ShotNumber"]}
                for entry in new_data
            ])

            # 保存新的 CSV 文件
            new_csv_path = os.path.join(self.image_folder, "new_face_cluster_statistics.csv")
            new_df.to_csv(new_csv_path, index=False)

            QMessageBox.information(self, "Success", f"New CSV file has been generated: {new_csv_path}")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to generate new CSV: {e}")

    def show_subtitle(self):
        # 构造matedata.csv的完整路径
        csv_file_path = os.path.join(os.path.dirname(self.image_folder), "matedata.csv")

        # 检查文件是否存在
        if os.path.exists(csv_file_path):
            subtitles = []  # 用于存储第二列的每行内容

            # 自动检测文件编码
            with open(csv_file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']

            try:
                # 打开并读取 CSV 文件，使用检测到的编码
                with open(csv_file_path, mode='r', encoding=encoding) as file:
                    csv_reader = csv.reader(file)
                    for row in csv_reader:
                        if len(row) > 1:  # 确保第二列存在
                            subtitles.append(row[1])  # 提取第二列的内容

                # 将提取出来的内容拼接为一个带换行符的字符串
                subtitle_text = "\n".join(subtitles)

                # 发送字幕内容
                self.parent.subtitle.textSubtitle.setPlainText(subtitle_text)
            
            except UnicodeDecodeError:
                print(f"Error decoding the file with {encoding}.")
                # 处理编码错误的情况
        else:
            QMessageBox.warning(self, "Error", f"Please generate the CSV first.")
            print(f"File '{csv_file_path}' does not exist.")
