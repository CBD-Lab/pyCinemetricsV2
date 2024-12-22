import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from algorithms.resultsave import Resultsave
from ui.progressbar import pyqtbar
from ui.progressbar import *
from algorithms.Transnetv1_ECA import TransNet


def cpu():
    """Get the CPU device.

    Defined in :numref:`sec_use_gpu`"""
    return torch.device('cpu')


def gpu(i=0):
    """Get a GPU device.

    Defined in :numref:`sec_use_gpu`"""
    return torch.device(f'cuda:{i}')


def num_gpus():
    """Get the number of available GPUs.

    Defined in :numref:`sec_use_gpu`"""
    return torch.cuda.device_count()


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()


class TransNetV2(QThread):
    #  通过类成员对象定义信号对象
    signal = Signal(int, int, int)
    # 线程中断
    is_stop = 0
    video_fn: str
    image_save: str
    # 线程结束信号
    finished = Signal(bool)

    def __init__(self, video_f, image_sav, parent, para_path=None):
        super(TransNetV2, self).__init__()
        self.is_stop = 0
        self.video_fn = video_f
        self.image_save = image_sav
        self.parent = parent
        if para_path is None:
            para_path = os.path.join(os.path.dirname(__file__),
                                     "../models/transnetv1_ECA/BEST_ECA_357k_F1_ECA_lr_0.001-wd_0.001-bs_32_48273264_0.3_iters_5000.pth")

            if not os.path.exists(para_path):  # 判断文件或目录是否存在
                raise FileNotFoundError(f"[TransNetV2] ERROR: {para_path} does not exist.")
            else:
                print(f"[TransNetV2] Using weights from {para_path}.")

        self._input_size = (27, 48, 3)

        if num_gpus():
            devices = [try_gpu(i) for i in range(num_gpus())]
            result_dict = torch.load(para_path)
        else:
            devices = [cpu()]
            result_dict = torch.load(para_path, map_location=torch.device(devices[0]))

        self.model = TransNet(test=True).to(devices[0])
        self.model = nn.DataParallel(self.model, device_ids=devices)
        self.model.load_state_dict(result_dict["net"])

    # def predict_raw(self, frames: np.ndarray):
    #     assert len(frames.shape) == 5 and frames.shape[2:] == self._input_size, \
    #         "[TransNetV2] Input shape must be [batch, frames, height, width, 3]."
    #     frames = tf.cast(frames, tf.float32)

    #     logits, dict_ = self._model(frames)
    #     single_frame_pred = tf.sigmoid(logits)
    #     all_frames_pred = tf.sigmoid(dict_["many_hot"])

    #     return single_frame_pred, all_frames_pred

    def predict_raw(self, frames):
        assert len(frames.shape) == 5 and frames.shape[2:] == self._input_size, \
            "[TransNetV2] Input shape must be [batch, frames, height, width, 3]."
        return self.model(frames)

    def predict_video(self, frames: np.ndarray):
        print("-------------START predict_video -------------")
        input_width = self.model.module.INPUT_WIDTH
        input_height = self.model.module.INPUT_HEIGHT
        pre = self.model.module.pre
        window = self.model.module.window
        look_window = self.model.module.lookup_window
        print(f'pre:{pre}, window: {window}, look_wiodow: {look_window}')
        assert len(frames.shape) == 4 and list(frames.shape[1:]) == [input_height, input_width, 3], \
            "[TransNet] Inputs shape must be [frames, height, width, 3]."

        def input_iterator():
            lookup_window = pre * 2 + window
            # return windows of size 128 where the first/last 36 frames are from the previous/next batch
            # the first and last window must be padded by copies of the first and last frame of the video
            no_padded_frames_start = pre
            no_padded_frames_end = pre + window - (len(frames) % window if len(frames) % window != 0 else window)

            start_frame = frames[0].unsqueeze(0)  # Unsqueezing to add batch dimension
            end_frame = frames[-1].unsqueeze(0)  # Unsqueezing to add batch dimension
            padded_inputs = torch.cat(
                [start_frame.repeat(no_padded_frames_start, 1, 1, 1),
                 frames,
                 end_frame.repeat(no_padded_frames_end, 1, 1, 1)],
                dim=0
            )

            ptr = 0
            while ptr + lookup_window <= len(padded_inputs):
                out = padded_inputs[ptr: ptr + lookup_window]
                ptr += window  # 每次指针前进32帧
                yield out[np.newaxis]

        predictions = []
        # 进度条设置
        total_number = len(frames)  # 总任务数

        for inp in input_iterator():
            if self.is_stop:
                self.finished.emit(True)
                break

            single_frame_pred = self.predict_raw(inp)

            predictions_window = single_frame_pred.detach().numpy().reshape(-1)
            predictions.append(predictions_window)

            print("\r[TransNetV2] Processing video frames {}/{}".format(
                min(len(predictions) * window, len(frames)), len(frames)
            ), end="")

            # ----------------------
            # 实现实时插入
            pred = np.concatenate([single_ for single_ in predictions])
            if (np.any(predictions_window > 0.3)):
                self.save_pred(pred[:min(len(pred), len(frames))])

            # 结束 -----------------

            percent = round(float(min(len(predictions) * window, len(frames)) / len(frames)) * 100)
            self.signal.emit(percent, min(len(predictions) * window, len(frames)), total_number)  # 发送实时任务进度和总任务进度

            # percent = round(float(min(len(predictions) * 50, len(frames))/ len(frames)) * 100)
            # bar.set_value(min(len(predictions) * 50, len(frames)), len(frames), percent)  # 刷新进度条

        if self.is_stop:
            self.finished.emit(True)
            pass
        else:
            single_frame_pred = np.concatenate([single_ for single_ in predictions])
            self.single_frame = single_frame_pred[:len(frames)]
            # return single_frame_pred[:len(frames)], all_frames_pred[:len(frames)]  # remove extra padded frames

    def run(self):
        # print("[TransNetV2] Extracting frames from {}".format(video_fn))
        # 进度条设置
        total_number = 0  # 总任务数
        task_id = 0  # 子任务序号

        try:
            import cv2
        except ModuleNotFoundError:
            raise ModuleNotFoundError("For `predict_video` function `cv2` needs to be installed in order to extract "
                                      "individual frames from video file. Install `cv2` command line tool and then "
                                      "install python wrapper by `pip install opencv-python`.")

        print("[TransNetV2] Extracting frames from {}".format(self.video_fn))
        cap = cv2.VideoCapture(self.video_fn)
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (48, 27))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        self.video = np.array(frames)
        # self.video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])
        self.predict_video(torch.from_numpy(self.video))
        self.signal.emit(101, 101, 101)  # 完事了再发一次

        if self.is_stop:
            self.finished.emit(True)
            pass
        else:
            self.run_moveon()
        # if self.isRunning():
        # self.terminate()
        # print(video)
        # return (video, *self.predict_frames(video))

    @staticmethod
    def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.3):
        predictions = (predictions > threshold).astype(np.uint8)

        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        # just fix if all predictions are 1
        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)

    @staticmethod
    def visualize_predictions(frames: np.ndarray, predictions):
        from PIL import Image, ImageDraw

        if isinstance(predictions, np.ndarray):
            predictions = [predictions]

        ih, iw, ic = frames.shape[1:]
        width = 25

        # pad frames so that length of the video is divisible by width
        # pad frames also by len(predictions) pixels in width in order to show predictions
        pad_with = width - len(frames) % width if len(frames) % width != 0 else 0
        frames = np.pad(frames, [(0, pad_with), (0, 1), (0, len(predictions)), (0, 0)])

        predictions = [np.pad(x, (0, pad_with)) for x in predictions]
        height = len(frames) // width

        img = frames.reshape([height, width, ih + 1, iw + len(predictions), ic])
        img = np.concatenate(np.split(
            np.concatenate(np.split(img, height), axis=2)[0], width
        ), axis=2)[0, :-1]

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        # iterate over all frames
        for i, pred in enumerate(zip(*predictions)):
            x, y = i % width, i // width
            x, y = x * (iw + len(predictions)) + iw, y * (ih + 1) + ih - 1

            # we can visualize multiple predictions per single frame
            for j, p in enumerate(pred):
                color = [0, 0, 0]
                color[(j + 1) % 3] = 255

                value = round(p * (ih - 1))
                if value != 0:
                    draw.line((x + j, y, x + j, y - value), fill=tuple(color), width=1)
        return img

    def stop(self):
        self.is_stop = 1

    def save_pred(self, pred):

        scenes = self.predictions_to_scenes(pred)
        number = [sublist[1] for sublist in scenes]
        number.pop()
        print(number)
        cap = cv2.VideoCapture(self.video_fn)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_len = len(str((int)(frame_count)))

        frame_save = os.path.join(self.image_save, "frame")
        os.makedirs(frame_save, exist_ok=True)

        # 后续的分镜图片
        _, img1 = cap.read()
        for i in number:
            i = i + 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            _, img2 = cap.read()
            j = ('%0{}d'.format(frame_len)) % i
            png_save_path = os.path.join(frame_save, f"frame{str(j)}.png")
            if not os.path.exists(png_save_path):
                cv2.imwrite(png_save_path, img2)
            img1 = img2
            self.parent.parent.shot_finished.emit()

    def run_moveon(self):

        # 保存路径
        frame_save = os.path.join(self.image_save, "frame")

        # 删除旧的分镜
        if not (os.path.exists(self.image_save)):
            os.mkdir(self.image_save)
        if not (os.path.exists(frame_save)):
            os.mkdir(frame_save)
        else:
            imgfiles = os.listdir(os.path.join(os.getcwd(), frame_save))
            for f in imgfiles:
                os.remove(os.path.join(os.getcwd(), frame_save, f))

        video_frames = self.video
        single_frame_predictions = self.single_frame
        # video_frames, single_frame_predictions, all_frame_predictions = \
        #     pyqtbar(model)#model.predict_video(file)

        scenes = self.predictions_to_scenes(single_frame_predictions)

        np.savetxt(os.path.join(self.image_save, "video.txt"), scenes, fmt="%d")

        # pil_image = self.visualize_predictions(
        #     video_frames, predictions=(single_frame_predictions, all_frame_predictions))
        # pil_image.save(file + ".vis.png")

        number = []
        number = getFrame_number(os.path.join(self.image_save, "video.txt"))
        number.pop()

        cap = cv2.VideoCapture(self.video_fn)

        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_len = len(str((int)(frame_count)))
        shot_len = []
        start = 0
        # 第一帧的图片
        i = 0
        _, img1 = cap.read()
        frameid = ""
        for j in range(frame_len - len(str(i))):
            frameid = frameid + "0"
        cv2.imwrite(os.path.join(frame_save, f"/frame{frameid}{i}.png"), img1)

        # 后续的分镜图片
        _, img1 = cap.read()
        for i in number:
            i = i + 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            _, img2 = cap.read()
            j = ('%0{}d'.format(frame_len)) % i
            cv2.imwrite(os.path.join(frame_save, f"frame{str(j)}.png"), img2)
            shot_len.append([start, i, i - start])
            start = i
            img1 = img2
            # self.parent.parent.shot_finished.emit()
        print("TransNetV2 completed")  # 把画图放进来
        # 发送shot_finished信号，进行处理
        self.parent.parent.shot_finished.emit()

        rs = Resultsave(self.image_save + "/")
        rs.plot_transnet_shotcut(shot_len)
        rs.diff_csv(0, shot_len)

        self.finished.emit(True)
        # self.parent.shotcut.clicked.connect(lambda: self.parent.colors.setEnabled(True))


def getFrame_number(f_path):
    f = open(f_path, 'r')
    Frame_number = []

    i = 0
    for line in f:
        NumList = [int(n) for n in line.split()]
        Frame_number.append(NumList[1])

    print(Frame_number)
    return Frame_number


def transNetV2_run(v_path, image_save, parent):  # parent定义有点奇怪
    import sys
    import argparse

    file = v_path
    if os.path.exists(file + ".predictions.txt") or os.path.exists(file + ".scenes.txt"):
        print(f"[TransNetV2] {file}.predictions.txt or {file}.scenes.txt already exists. "
              f"Skipping video {file}.", file=sys.stderr)

    # 模型跑完了生成一个分镜帧号的txt
    model = TransNetV2(file, image_save, parent)
    model.finished.connect(parent.shotcut.setEnabled)
    model.finished.connect(parent.colors.setEnabled)
    model.finished.connect(parent.objects.setEnabled)
    model.finished.connect(parent.subtitleBtn.setEnabled)
    model.finished.connect(parent.shotscale.setEnabled)
    bar = pyqtbar(model)


