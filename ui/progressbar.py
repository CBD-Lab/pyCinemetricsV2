from PySide2 import QtWidgets, QtCore
from PySide2.QtCore import *

class ProcessBar(QtWidgets.QDialog):
    def __init__(self, work):
        super().__init__()
        self.work = work
        self.run_work()
    def call_backlog(self, msg, task_number, total_task_number):
        if task_number == 0 and total_task_number == 0:
            self.setWindowTitle(self.tr('Processing...'))
        elif msg == 101 and task_number == 101 and total_task_number == 101:
            print("thread quit")
            self.close()
        else:
            label = "Processing：" + "task No" + str(task_number) + "/" + str(total_task_number)
            self.setWindowTitle(self.tr(label))  # 顶部的标题
        self.pbar.setValue(int(msg))  # 将线程的参数传入进度条
        #QtWidgets.QApplication.processEvents() # 实时刷新显示

    def run_work(self):
        # 创建线程
        # 连接信号
        self.work.signal.connect(self.call_backlog)  # 进程连接回传到GUI的事件#很奇怪为什么这样就可以连接上
        # 开始线程
        self.work.start()

        # 进度条设置
        self.pbar = QtWidgets.QProgressBar(self)
        self.pbar.setMinimum(0)  # 设置进度条最小值
        self.pbar.setMaximum(100)  # 设置进度条最大值
        self.pbar.setValue(0)  # 进度条初始值为0
        self.pbar.setGeometry(QRect(1, 3, 499, 28))  # 设置进度条在 QDialog 中的位置 [左，上，右，下]
        self.show()
        # 窗口初始化
        # self.setGeometry(300, 300, 500, 32)
        # self.setWindowTitle('正在处理中')
        # self.show()
        # self.work = None  # 初始化线程

    def closeEvent(self, event):#抛出异常不好使，有待改进
        self.work.stop()
        #self.work.terminate()# 强制


class pyqtbar():
    def __init__(self, work):
        self.myshow = ProcessBar(work)
        work.signal.connect(self.myshow.call_backlog())
        #self.myshow.show()
