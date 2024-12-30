import os
from collections import Counter

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.vq import vq, kmeans, whiten
import math
from algorithms.resultsave import Resultsave
from ui.progressbar import *

class ColorAnalysis(QThread):
    #  通过类成员对象定义信号对象
    signal = Signal(int, int, int, str)
    #线程中断
    flag = 0
    # 线程结束信号
    finished = Signal(bool)

    def __init__(self, filename, imgpath, colorsC):
        super(ColorAnalysis, self).__init__()
        self.filename = filename
        self.imgpath = imgpath
        self.colorsC = colorsC

    def load_image(self):
        img = Image.open(self.filename)
        img = img.rotate(-90)
        img.thumbnail((200, 200))
        w, h = img.size
        points = []
        for count, color in img.getcolors(w * h):
            points.append(color)
        return points

    def kmeans(self,imgdata, n):
        data = np.array(imgdata, dtype=float)
        centers, loss = kmeans(data, n)
        centers = np.array(centers, dtype=int)
        return centers

    def calculate_distances(self, centers):
        imgdata = self.load_image()
        result = []
        for one_center in centers:
            dis = math.sqrt(one_center[0] ** 2 + one_center[1] ** 2 + one_center[2] ** 2)
            flag = -1
            for index, one_color in enumerate(imgdata):
                temp = math.sqrt((one_color[0] - one_center[0]) ** 2 + (one_color[1] - one_center[1]) ** 2 + (
                        one_color[2] - one_center[2]) ** 2)
                if temp < dis:
                    dis = temp
                    flag = index
            result.append(list(imgdata[flag]))
        return result

    def rgb_to_hex(self, real_color):
        colors_16 = []
        for one_color in real_color:
            color_16 = '#'
            for one in one_color:
                color_16 += str(hex(one))[-2:].replace('x', '0').upper()
            colors_16.append(color_16)
        return colors_16

    def run(self):
        imglist = os.listdir("img/" + self.imgpath + '/frame/')
        color_16_list = []
        allcolors_rgb = []
        allcolors_rgb_list = []
        allcolors_16 = []

        # 进度条设置
        total_number = len(imglist)  # 总任务数
        task_id = 1  # 子任务序号

        for i in imglist:
            if self.flag:
                self.finished.emit(True)
                break
            self.filename=("img/" + self.imgpath + "/frame/" + i)
            imgdata = self.load_image()
            if len(imgdata) < self.colorsC:
                color_rgb = [list(imgdata[0])] * self.colorsC
            else:
                colors = self.kmeans(imgdata, self.colorsC)  # 提取几种色彩
                color_rgb = self.calculate_distances(colors)
            color_16 = self.rgb_to_hex(color_rgb)

            allcolors_16 += color_16
            allcolors_rgb += color_rgb
            allcolors_rgb_list.append((list)(color_rgb))  # 用于plot3D

            color_16_list.append([i, color_16])

            percent = round(float(task_id / total_number) * 100)
            self.signal.emit(percent, task_id, total_number, "ColorAnalyze")  # 发送实时任务进度和总任务进度

            task_id += 1
        self.signal.emit(101, 101, 101, "ColorAnalyze")  # 完事了再发一次

        if self.flag:
            self.finished.emit(True)
            pass
        else:
            # 创建一个py文件 骗一下matplot让它以为在主线程里
            rs = Resultsave("./img/"+self.imgpath+"/")
            rs.color_csv(color_16_list, self.colorsC)
            rs.plot_scatter_3d(allcolors_rgb_list)
            self.finished.emit(True)

    def stop(self):
        self.flag = 1
        
    def analysis1img(self, imgpath, colorC):
        self.filename = imgpath
        imgdata = self.load_image()
        if len(imgdata) < self.colorsC:
            color_rgb = [list(imgdata[0])] * self.colorsC
        else:
            colors = self.kmeans(imgdata, self.colorsC)  # 提取几种色彩
            color_rgb = self.calculate_distances(colors)
        color_16 = self.rgb_to_hex(color_rgb)
        self.drawpie(imgdata, color_rgb, color_16)

    def drawpie(self, imgdata, colors, colors_16):
        cluster1, _ = vq(imgdata, colors)
        result = Counter(cluster1.tolist())
        plt.clf()
        plt.style.use("dark_background")
        plt.pie(x=[result[i] for i in range(self.colorsC)],  # 指定绘图数据
                colors=colors_16[:self.colorsC],  # 为饼图添加标签说明
                # wedgeprops=dict(width=0.2, edgecolor='w'),  # 设置环图
                wedgeprops=dict(edgecolor='w'),  # 设置饼图
                labels=colors_16,
                autopct='%1.2f%%',
                )
        plt.savefig('/'.join(self.filename.split("/")[:2])+ '/colortmp.png')
        # plt.show()
