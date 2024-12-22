import os
import numpy as np
import csv
import torch
import torch.nn
import torchvision.models as models
import torch.cuda
from torchvision import transforms
from algorithms.wordcloud2frame import WordCloud2Frame
from ui.progressbar import *
from collections import Counter
from PIL import Image

class ObjectDetection(QThread):
    signal = Signal(int, int, int)  # 进度更新信号
    finished = Signal(bool)        # 任务完成信号
    is_stop = 0                    # 是否中断标志

    def __init__(self, image_path):
        super(ObjectDetection, self).__init__()
        self.is_stop = 0
        self.image_path = image_path
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

    def make_model(self):
        # 加载 ResNet50 模型结构
        model = models.resnet50(pretrained=False)  # 不下载权重，只加载模型结构

        # 加载本地权重文件
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_weights_path = os.path.join(base_dir,"../models/resnet50-0676ba61.pth")  # 替换为实际路径
        state_dict = torch.load(local_weights_path)

        # 将权重加载到模型中
        model.load_state_dict(state_dict)
        model = model.eval()
        if torch.cuda.is_available():
            model.cuda()
        return model

    def run(self):
        model = self.make_model()
        if self.image_path is None or self.image_path == '':
            return

        frame_dir = os.path.join(self.image_path, "frame")
        file_list = os.listdir(frame_dir)
        framelist = []

        #  1000 个 ImageNet 类别名称
        with open('./files/imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]

        # 进度条设置
        total_number = len(file_list)  # 总任务数
        task_id = 0  # 子任务序号

        for file_name in file_list:
            if self.is_stop:
                self.finished.emit(True)
                break
            if os.path.splitext(file_name)[-1] in ['.jpg', '.png', '.bmp']:
                img_path = os.path.join(frame_dir, file_name)
                img_t = self.transform(Image.open(img_path).convert('RGB'))

                if torch.cuda.is_available():
                    batch_t = torch.unsqueeze(img_t, 0).cuda()
                else:
                    batch_t = torch.unsqueeze(img_t, 0)

                out = model(batch_t)
                _, indices = torch.sort(out, descending=True)
                percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

                # 每帧选取，10个物体
                for idx in indices[0][:10]:
                    frame_id = file_name[5:-4]
                    framelist.append([frame_id, (classes[idx], percentage[idx].item())[0]])

                percent = round(float(task_id / total_number) * 100)
                self.signal.emit(percent, task_id, total_number)  # 发送实时任务进度和总任务进度

                task_id += 1

        self.signal.emit(101, 101, 101)  # 完成后发送信号

        if self.is_stop:
            self.finished.emit(True)
            pass
        else:
            self.object_detection_csv(framelist, self.image_path)

    def object_detection_csv(self, framelist, save_path):
        
        csv_file = open(os.path.join(save_path, 'objects.csv'), "w+", newline='')
        name = ['FrameId', 'Top1-Objects']

        try:
            writer = csv.writer(csv_file)
            writer.writerow(name)
            datarow = []

            for i in range(len(framelist)):
                datarow = [framelist[i][0]]
                datarow.append(framelist[i][1])
                writer.writerow(datarow)
        finally:
            csv_file.close()
            wc2f = WordCloud2Frame()
            tf = wc2f.wordfrequency(os.path.join(save_path, 'objects.csv'))
            wc2f.plotwordcloud(tf, save_path, '/objects')
            self.finished.emit(True)

    def stop(self):
        self.is_stop = 1