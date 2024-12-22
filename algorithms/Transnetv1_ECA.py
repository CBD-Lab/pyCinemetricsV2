import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as functional
from torch.utils.checkpoint import checkpoint


# Batch Normalization只出现在输入位置


# ------------------------- 定义 TransNet+Res+BatchNorm+NonLocal3+RGB+AvgPool_1283218_56 训练的网络结构 ------------------------------
class TransNet(nn.Module):
    
    def __init__(self,
                 F=8, L=3, S=2, D=128,
                 INPUT_WIDTH=48, INPUT_HEIGHT=27, lookup_window=128,
                 use_color_histograms=True,
                 pre=32, window=64, eca=True, test=False):
                 # F=16, L=3, S=2, D=256,
                 # INPUT_WIDTH=48, INPUT_HEIGHT=27):
        super(TransNet, self).__init__()
        
        self.INPUT_WIDTH = INPUT_WIDTH
        self.INPUT_HEIGHT = INPUT_HEIGHT
        self.lookup_window = lookup_window
        self.compress_w = INPUT_WIDTH // 8
        self.compress_h = INPUT_HEIGHT // 8
        self.pre = pre
        self.window = window
        self.test= test
        
        self.SDDCNN = nn.ModuleList(
            [StackedDDCNN(in_filters=3, n_blocks=S, filters=F, eca=eca, k_size=3)] +
            [StackedDDCNN(in_filters=(F * 2 ** (i - 1)) * 4, n_blocks=S, filters=F * 2 ** i,
                           eca=eca, k_size = 3 + 2 * i) for i in range(1, L)]
        )
        # self.SDDCNN = nn.ModuleList(
        #     [StackedDDCNN(in_filters=3, n_blocks=S, filters=F, eca=eca, k_size=5)] +
        #     [StackedDDCNN(in_filters=(F * 2 ** (i - 1)) * 4, n_blocks=S, filters=F * 2 ** i,
        #                    eca=eca, k_size = 5) for i in range(1, L)]
        # )
        self.color_hist_layer = ColorHistograms(lookup_window=self.lookup_window + 1, output_dim=32) if use_color_histograms else None
        
        output_dim = ((F * 2 ** (L - 1)) * 4) * self.compress_h * self.compress_w  # 2x4 for spatial dimensions
        # output_dim = ((F * 2 ** (L - 1)) * 4) * 3 * 6  # 3x6 for spatial dimensions
        if use_color_histograms:
            output_dim += 32
        self.fc1 = nn.Linear(output_dim, D)
        self.fc2 = nn.Linear(D, 2)
    
    def forward(self, inputs):
        # print("\n[TransNet] Creating ops.")
        # uint8 of shape [B, T, H, W, 3] to float of shape [B, 3, T, H, W]
        x = inputs.permute([0, 4, 1, 2, 3]).float()
        # x = x.div_(255.)  # 20240829 将这行代码注释掉了
        # print("\n", " " * 9, "Input: ", x.shape)
        
        # --------- Start of Batch Normalization ---------
        batch_size, channels, time, height, width = x.shape
        x = x.view(batch_size, channels, -1, self.lookup_window, height, width)  # 这里的128需要根据设置的窗口帧数进行调整（下同）
        # 计算每组128帧的均值和标准差
        mean = x.mean(dim=(3, 4, 5), keepdim=True)
        std = x.std(dim=(3, 4, 5), keepdim=True)
        # 归一化
        x = (x - mean) / (std + 1e-5)
        # 恢复原始形状
        x = x.view(batch_size, channels, time, height, width)
        # --------- End of Batch Normalization ---------
        
        for block in self.SDDCNN:
            x = block(x)
        
        # uint8 of shape [B, C, T, H, W] to float of shape [B, T, H, W, C]
        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # print("\n", " " * 9, "Flatten: ", x.shape)

        if self.color_hist_layer:
            x = torch.cat([self.color_hist_layer(inputs), x], 2)
            
        x = functional.relu(self.fc1(x))
        # print(" " * 10, "Dense_1: ", x.shape)
        
        x = self.fc2(x)
        # print(" " * 10, "Dense_2: ", x.shape)

        
        if self.test:
            x = functional.softmax(x, dim=-1)
            # print(" " * 10, "Softmax: ", x.shape)
        
        return x[:, :, 1][:, self.pre: self.pre + self.window]

        
# ------------------------- 定义 TransNet+Res+BatchNorm+NonLocal3+RGB+AvgPool_1283218_32 内部所用到的网络结构 ------------------------------
class StackedDDCNN(nn.Module):
    
    def __init__(self,
                 in_filters,
                 n_blocks,
                 filters,
                 shortcut=True,  # 残差
                 eca=False, k_size=3):  # Non-Local Block
        super(StackedDDCNN, self).__init__()
        
        self.shortcut = shortcut
        self.eac = eca
        self.DDCNN = nn.ModuleList([
            DilatedDCNN(in_filters if i == 1 else filters * 4, filters) for i in range(1, n_blocks + 1)
        ])
        
        self.eac_layer = eca_layer_3d(k_size=k_size)
        
        self.avg_pool = nn.AvgPool3d(kernel_size=(1, 2, 2))
    
    def forward(self, inputs):
        x = inputs
        # print("\n", " " * 9, "> SDDCNN: ")
        shortcut = None
        
        for block in self.DDCNN:
            x = block(x)  # x被更新为第二次DDCNN的结果
            if shortcut is None:
                shortcut = x  # 将第一次DDCNN的结果存入shortcut中
        
        if self.shortcut:
            x += shortcut  # 残差连接
        
        if self.eac:
            x = self.eac_layer(x)

        x = self.avg_pool(x)

        # print(" " * 10, "AvgPool: ", x.shape)
        return x


class DilatedDCNN(nn.Module):
    
    def __init__(self,
                 in_filters,
                 filters):
        super(DilatedDCNN, self).__init__()
        
        self.Conv3D_1 = Conv3DConfigurable(in_filters, filters, 1, use_bias=True)
        self.Conv3D_2 = Conv3DConfigurable(in_filters, filters, 2, use_bias=True)
        self.Conv3D_4 = Conv3DConfigurable(in_filters, filters, 4, use_bias=True)
        self.Conv3D_8 = Conv3DConfigurable(in_filters, filters, 8, use_bias=True)
    
    def forward(self, inputs):
        conv1 = self.Conv3D_1(inputs)
        # print(" " * 20, "conv1: ", conv1.shape)
        
        conv2 = self.Conv3D_2(inputs)
        # print(" " * 20, "conv2: ", conv2.shape)
        
        conv3 = self.Conv3D_4(inputs)
        # print(" " * 20, "conv3: ", conv3.shape)
        
        conv4 = self.Conv3D_8(inputs)
        # print(" " * 20, "conv4: ", conv4.shape)
        
        x = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        # print(" " * 10, ">> DDCNN: ", x.shape)
        
        return x


class eca_layer_3d(nn.Module):
    """Constructs a 3D ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=5):
        super(eca_layer_3d, self).__init__()
        # Global average pooling across time, height, and width
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # Adapts to 5D input (batch, channel, time, height, width)
        # 1D convolution to capture channel-wise dependencies
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global average pooling on spatial and temporal dimensions
        y = self.avg_pool(x)  # Shape: [batch_size, channels, 1, 1, 1]

        # Squeeze and transpose to apply 1D convolution over channels
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2))  # Shape: [batch_size, 1, channels]

        # Transpose back and reshape the output
        y = y.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, channels, 1, 1, 1]

        # Apply sigmoid activation
        y = self.sigmoid(y)

        # Element-wise multiplication to apply channel attention
        return x * y.expand_as(x)


class Conv3DConfigurable(nn.Module):
    
    def __init__(self,
                 in_filters,
                 filters,
                 dilation_rate,
                 use_bias=True):
        super(Conv3DConfigurable, self).__init__()
        
        self.conv = nn.Conv3d(in_filters, filters, kernel_size=3,
                              dilation=(dilation_rate, 1, 1), padding=(dilation_rate, 1, 1), bias=use_bias)
    
    def forward(self, inputs):
        x = functional.relu(self.conv(inputs))
        
        return x


class ColorHistograms(nn.Module):
    
    def __init__(self,
                 lookup_window=101,
                 output_dim=None):
        super(ColorHistograms, self).__init__()
        
        self.fc = nn.Linear(lookup_window, output_dim) if output_dim is not None else None
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"
    
    @staticmethod
    def compute_color_histograms(frames):
        frames = frames.int()
        
        def get_bin(frames):
            # returns 0 .. 511
            # 提取R、G、B三个通道并将其右移5位，保证每个通道的值都在0-7之间
            # 保留最高的3位，将颜色值映射到一个较小的范围，从而减少直方图的维度
            R, G, B = frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]
            R, G, B = R >> 5, G >> 5, B >> 5
            # 返回每个像素的bin值，将R、G、B三个通道的值合并为一个0到511之间的值（9位）
            return (R << 6) + (G << 3) + B
        
        batch_size, time_window, height, width, no_channels = frames.shape
        assert no_channels == 3
        frames_flatten = frames.view(batch_size * time_window, height * width, 3)
        
        # 获取每个像素点的bin值
        binned_values = get_bin(frames_flatten)  # torch.Size([batch_size * time_window, height * width])
        # 创建帧的bin前缀，将批次大小和时间窗口的信息合并到bin值中
        # 左移9位=乘512，为每个帧分配一个唯一的前缀，确保每个帧的像素值在最终的bin值计算中不会冲突
        frame_bin_prefix = (torch.arange(0, batch_size * time_window, device=frames.device) << 9).view(-1,
                                                                                                       1)  # torch.Size([batch_size * time_window, 1])
        binned_values = (binned_values + frame_bin_prefix).view(
            -1)  # torch.Size([batch_size * time_window * height * width])
        
        # 初始化直方图张量，大小为batch_size * time_window * 512（每一帧对应一个512维的直方图），用于存储每个bin的计数
        histograms = torch.zeros(batch_size * time_window * 512, dtype=torch.int32, device=frames.device)
        # 使用scatter_add_方法填充直方图张量，计算每个bin的计数，得到每一帧的颜色直方图
        histograms.scatter_add_(0, binned_values,
                                torch.ones(len(binned_values), dtype=torch.int32, device=frames.device))
        
        # 调整直方图的形状，恢复batch_size和time_window的维度
        histograms = histograms.view(batch_size, time_window, 512).float()
        # 对直方图进行归一化处理，使得每个直方图的L2范数为1
        histograms_normalized = functional.normalize(histograms, p=2, dim=2)
        return histograms_normalized
    
    def forward(self, inputs):
        # 计算输入frames的颜色直方图
        x = self.compute_color_histograms(inputs)  # torch.Size([batch_size, time_window, 512])
        
        batch_size, time_window = x.shape[0], x.shape[1]
        # 计算每个时间窗口内的相似性，使用的是批量矩阵乘法
        # [batch_size, time_window, 512] * [batch_size, 512, time_window] = [batch_size, time_window, time_window]
        similarities = torch.bmm(x, x.transpose(1, 2))
        # 填充相似性矩阵，使得每个时间窗口都有相同的查找窗口大小，torch.Size([batch_size, time_window, time_window + self.lookup_window - 1])
        similarities_padded = functional.pad(similarities,
                                             [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2])
        
        # 创建批次索引，形状为[batch_size, time_window, lookup_window]
        batch_indices = torch.arange(0, batch_size, device=x.device).view([batch_size, 1, 1]).repeat(
            [1, time_window, self.lookup_window])
        # 创建时间索引，形状为[batch_size, time_window, lookup_window]
        time_indices = torch.arange(0, time_window, device=x.device).view([1, time_window, 1]).repeat(
            [batch_size, 1, self.lookup_window])
        # 创建查找索引，形状为[batch_size, time_window, lookup_window]
        lookup_indices = torch.arange(0, self.lookup_window, device=x.device).view([1, 1, self.lookup_window]).repeat(
            [batch_size, time_window, 1]) + time_indices
        
        # 获取填充后的相似性矩阵，形状为[batch_size, time_window, lookup_window]
        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]
        
        if self.fc is not None:
            return functional.relu(self.fc(similarities))
        return similarities
