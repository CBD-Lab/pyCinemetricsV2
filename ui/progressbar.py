from PySide2 import QtWidgets, QtCore
from PySide2.QtCore import *

class ProcessBar(QtWidgets.QDialog):
    def __init__(self, work):
        super().__init__()
        self.work = work
        self.run_work()
        
    def call_backlog(self, msg, task_number, total_task_number, task_name=""):
        if task_number == 0 and total_task_number == 0:
            self.setWindowTitle(self.tr('Processing...'))
        elif msg == 101 and task_number == 101 and total_task_number == 101:
            print("Thread quit")
            self.close()
        else:
            # 显示任务名称和进度
            label = f"Processing: {task_name} (Task {task_number}/{total_task_number})"
            self.setWindowTitle(self.tr(label))  # 更新窗口标题
        
        # 更新进度条的值
        self.pbar.setValue(int(msg))
        # QtWidgets.QApplication.processEvents() # 实时刷新显示

    def run_work(self):
        # 创建线程并连接信号
        self.work.signal.connect(self.call_backlog)  # 进程连接到 GUI 的事件
        self.work.start()

        # 进度条设置
        self.pbar = QtWidgets.QProgressBar(self)
        self.pbar.setMinimum(0)  # 设置进度条最小值
        self.pbar.setMaximum(100)  # 设置进度条最大值
        self.pbar.setValue(0)  # 初始值为 0
        self.pbar.setGeometry(QRect(1, 3, 499, 28))  # 设置位置 [左, 上, 宽, 高]
        self.show()

    def closeEvent(self, event):
        # 停止线程
        self.work.stop()

class pyqtbar():
    def __init__(self, work):
        self.myshow = ProcessBar(work)
        work.signal.connect(lambda msg, task_number, total_task_number, task_name: 
                            self.myshow.call_backlog(msg, task_number, total_task_number, task_name))

