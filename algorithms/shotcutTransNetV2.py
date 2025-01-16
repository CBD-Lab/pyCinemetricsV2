import os
import numpy as np
import tensorflow as tf
import cv2
from algorithms.resultSave import Resultsave
from ui.progressBar import *

class TransNetV2(QThread):
    #  通过类成员对象定义信号对象
    signal = Signal(int, int, int, str)
    #线程中断
    is_stop = 0
    video_fn:str
    image_save:str
    #线程结束信号
    finished = Signal(bool)

    def __init__(self, video_f, parent, model_dir=None):
        super(TransNetV2, self).__init__()
        self.is_stop = 0
        self.video_fn = video_f
        self.image_save = parent.image_save
        self.frame_save = parent.frame_save
        self.parent = parent

        self.INPUT_WIDTH = 48
        self.INPUT_HEIGHT = 27
        self.pre = 25
        self.window = 50
        self.lookup_window = self.pre * 2 + self.window
        if model_dir is None:
            model_dir = "models/transnetv2-weights/"
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(f"[TransNetV2] ERROR: {model_dir} is not a directory.")
            else:
                print(f"[TransNetV2] Using weights from {model_dir}.")

        self._input_size = (27, 48, 3)
        self.model_dir = model_dir
        # try:
        #     self.model = tf.saved_model.load(model_dir)
        #     print(model_dir)
        # except OSError as exc:
        #     raise IOError(f"[TransNetV2] It seems that files in {model_dir} are corrupted or missing. "
        #                   f"Re-download them manually and retry. For more info, see: "
        #                   f"https://github.com/soCzech/TransNetV2/issues/1#issuecomment-647357796") from exc

    def predict_raw(self, frames: np.ndarray):
        assert len(frames.shape) == 5 and frames.shape[2:] == self._input_size, \
            "[TransNetV2] Input shape must be [batch, frames, height, width, 3]."
        frames = tf.cast(frames, tf.float32)

        logits, dict_ = self.model(frames)
        single_frame_pred = tf.sigmoid(logits)
        all_frames_pred = tf.sigmoid(dict_["many_hot"])

        return single_frame_pred, all_frames_pred

    def predict_frames(self, frames: np.ndarray):
        assert len(frames.shape) == 4 and frames.shape[1:] == self._input_size, \
            "[TransNetV2] Input shape must be [frames, height, width, 3]."

        def input_iterator():
            # return windows of size 100 where the first/last 25 frames are from the previous/next batch
            # the first and last window must be padded by copies of the first and last frame of the video
            no_padded_frames_start = 25
            no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)  # 25 - 74

            start_frame = np.expand_dims(frames[0], 0)
            end_frame = np.expand_dims(frames[-1], 0)
            padded_inputs = np.concatenate(
                [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
            )

            ptr = 0
            while ptr + 100 <= len(padded_inputs):
                out = padded_inputs[ptr:ptr + 100]
                ptr += 50
                yield out[np.newaxis]

        predictions = []

        # 进度条设置
        total_number = len(frames) # 总任务数

        for inp in input_iterator():
            if self.is_stop:
                self.finished.emit(True)
                break
            single_frame_pred, all_frames_pred = self.predict_raw(inp)
            predictions.append((single_frame_pred.numpy()[0, 25:75, 0],
                                all_frames_pred.numpy()[0, 25:75, 0]))

            print("\r[TransNetV2] Processing video frames {}/{}".format(
                min(len(predictions) * 50, len(frames)), len(frames)
            ), end="")
            # percent = round(float(min(len(predictions) * 50, len(frames))/ len(frames)) * 100)
            # self.signal.emit(percent, min(len(predictions) * 50, len(frames)),total_number,"LoadingVideo")  # 发送实时任务进度和总任务进度
            # percent = round(float(min(len(predictions) * 50, len(frames))/ len(frames)) * 100)
            # bar.set_value(min(len(predictions) * 50, len(frames)), len(frames), percent)  # 刷新进度条
        if self.is_stop:
            self.finished.emit(True)
            pass
        else:
            single_frame_pred = np.concatenate([single_ for single_, all_ in predictions])
            self.single_frame=single_frame_pred[:len(frames)]
            all_frames_pred = np.concatenate([all_ for single_, all_ in predictions])
            self.all_frames=all_frames_pred[:len(frames)]
            #return single_frame_pred[:len(frames)], all_frames_pred[:len(frames)]  # remove extra padded frames

    def predict_video(self, frames: np.ndarray):
        print("-------------START predict_video -------------")
        input_width = self.INPUT_WIDTH
        input_height = self.INPUT_HEIGHT
        pre = self.pre
        window = self.window
        look_window = self.lookup_window
        print(f'pre:{pre}, window: {window}, look_window: {look_window}')

        assert len(frames.shape) == 4 and list(frames.shape[1:]) == [input_height, input_width, 3], \
            "[TransNet] Inputs shape must be [frames, height, width, 3]."

        def input_iterator():
            lookup_window = pre * 2 + window
            # return windows of size 128 where the first/last 36 frames are from the previous/next batch
            # the first and last window must be padded by copies of the first and last frame of the video
            no_padded_frames_start = pre
            no_padded_frames_end = pre + window - (len(frames) % window if len(frames) % window != 0 else window)

            start_frame = np.expand_dims(frames[0], 0)
            end_frame = np.expand_dims(frames[-1], 0)
            padded_inputs = np.concatenate(
                [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
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

            single_frame_pred, all_frames_pred = self.predict_raw(inp)
            predictions_window = single_frame_pred.numpy()[0, pre: pre + window, 0]

            # ----------------------
            # 实现实时插入
            # 将第一帧处理
            if (len(predictions) == 0):
                arr = np.zeros(window, dtype=int)
                arr[0] = 1
                self.save_pred(arr, -1)
            # 其他帧判断处理
            elif (np.any(predictions_window > 0.3)):
                self.save_pred(predictions_window, min(len(predictions) * window, len(frames)))

            # 结束 -----------------

            predictions.append(predictions_window)
            print("\r[TransNetV2] Processing video frames {}/{}".format(
                min(len(predictions) * window, len(frames)), len(frames)
            ), end="")

            percent = round(float(min(len(predictions) * window, len(frames)) / len(frames)) * 100)
            self.signal.emit(percent, min(len(predictions) * window, len(frames)), total_number, "shotcut")  # 发送实时任务进度和总任务进度

        if self.is_stop:
            self.finished.emit(True)
            pass
        else:
            single_frame_pred = np.concatenate([single_ for single_ in predictions])
            self.single_frame = single_frame_pred[:len(frames)]
            # return single_frame_pred[:len(frames)], all_frames_pred[:len(frames)]  # remove extra padded frames

    def run(self):
        self.signal.emit(0, 0, 0,"shotcut")

        try:
            self.model = tf.saved_model.load(self.model_dir)
            print(self.model_dir)
        except OSError as exc:
            raise IOError(f"[TransNetV2] It seems that files in {self.model_dir} are corrupted or missing. "
                          f"Re-download them manually and retry. For more info, see: "
                          f"https://github.com/soCzech/TransNetV2/issues/1#issuecomment-647357796") from exc
        # 删除旧的分镜
        if not (os.path.exists(self.image_save)):
            os.mkdir(self.image_save)
        if not (os.path.exists(self.frame_save)):
            os.mkdir(self.frame_save)
        else:
            imgfiles = os.listdir(os.path.join(os.getcwd(), self.frame_save))
            for f in imgfiles:
                os.remove(os.path.join(os.getcwd(), self.frame_save, f))

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
        self.predict_video(self.video)
        self.signal.emit(101, 101, 101,"shotcut")  # 完事了再发一次
        if self.is_stop:
            self.finished.emit(True)
            pass
        else:
            self.run_moveon()

    @staticmethod
    def pred_window_to_shotList(predictions: np.ndarray, prev_cnt, threshold: float = 0.3):
        predictions = (predictions > threshold).astype(np.uint8)

        shotList = []
        if prev_cnt == 0:
            shotList.append(0)

        for i, t in enumerate(predictions):
            if t == 1:
                shotList.append(i + prev_cnt + 1)
        return shotList

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

        # return np.array(scenes, dtype=np.int32)
        return scenes

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

    def save_pred(self, pred, prev_count):

        number = self.pred_window_to_shotList(pred, prev_count)
        print(number)

        cap = cv2.VideoCapture(self.video_fn)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_len = len(str((int)(frame_count)))
        os.makedirs(self.frame_save, exist_ok=True)

        # 后续的分镜图片
        for i in number:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            _, img = cap.read()
            j = ('%0{}d'.format(frame_len)) % i
            png_save_path = os.path.join(self.frame_save, f"frame{str(j)}.png")
            if not os.path.exists(png_save_path):
                cv2.imwrite(png_save_path, img)
            self.parent.parent.shot_finished.emit()

    # 画图 和 保存
    def run_moveon(self):

        video_frames = self.video
        single_frame_predictions = self.single_frame

        scenes = self.predictions_to_scenes(single_frame_predictions)

        np.savetxt(os.path.join(self.image_save, "video.txt"), scenes, fmt="%d")

        number = []
        number = getFrame_number(os.path.join(self.image_save, "video.txt"))

        shot_len = []
        start = -1
        print(number)
        for idx, i in enumerate(number):

            if idx == (len(number) - 1):
                shot_len.append([start, i, i - start + 1])
            elif idx != 0:
                shot_len.append([start, i - 1, i - start])
            start = i

        print("TransNetV2 completed")  # 把画图放进来
        # 发送shot_finished信号，进行处理
        self.parent.parent.shot_finished.emit()
        rs = Resultsave(self.image_save + "/")
        rs.plot_transnet_shotcut(shot_len)
        rs.diff_csv(0, shot_len)
        self.finished.emit(True)

def getFrame_number(f_path):
    f = open(f_path, 'r')
    Frame_number = []

    for line in f:
        NumList = [int(n) for n in line.split()]
        Frame_number.append(NumList[0])

    return Frame_number


def transNetV2_run(v_path, parent):#parent定义有点奇怪
    import sys
    import argparse

    file = v_path
    if os.path.exists(file + ".predictions.txt") or os.path.exists(file + ".scenes.txt"):
        print(f"[TransNetV2] {file}.predictions.txt or {file}.scenes.txt already exists. "
              f"Skipping video {file}.", file=sys.stderr)

    # 模型跑完了生成一个分镜帧号的txt
    model = TransNetV2(file, parent)
    model.finished.connect(parent.shotcut.setEnabled)
    model.finished.connect(parent.colors.setEnabled)
    model.finished.connect(parent.objects.setEnabled)
    model.finished.connect(parent.subtitleBtn.setEnabled)
    model.finished.connect(parent.shotscale.setEnabled)
    bar = pyqtbar(model)


