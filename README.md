# pyCinemetrics_V2.0

## English Version

1. After cloning, the only missing folder is `models`. After downloading the `models` folder from the cloud drive, just place it in the directory, and it will run.
2. Files shared through the cloud drive: 
    Model files and test_videos:
   - Link: [Baidu Netdisk](https://pan.baidu.com/s/1GMlOYvglimvSoIcIowuM0A?pwd=1234), [Google Drive](https://drive.google.com/drive/folders/1ho48Bx6KF-fZewnwBpoHmdI5F1XCY6Xm?usp=sharing)  
3. Model and functionality correspondences:
    - Video boundary detection: transnetv2
    - Face recognition: buffalo_l
    - Speech-to-text subtitles: faster-whisper-base
    - Subtitle detection：paddleocr
    - Object detection: git-base
    - Translation: opus-mt-en-zh
    - Shot type recognition：pose net
4. Potential issues:
    - If `pip install insightface` fails, please install it manually. [Link to GitHub](https://github.com/Gourieff/Assets/tree/main/Insightface)
    - Some models are outdated, so the latest numpy cannot be used. Version 1.26.0 is compatible.
5.License:
    - This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License (LGPL) as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.
---

## 中文版本

1. clone之后仅缺少`models`文件夹，网盘下载`models`文件夹后放进目录即可运行。
2. 通过网盘分享的文件：
    模型文件和测试视频：
   - 链接: [Baidu Netdisk](https://pan.baidu.com/s/1GMlOYvglimvSoIcIowuM0A?pwd=1234), [Google Drive](https://drive.google.com/drive/folders/1ho48Bx6KF-fZewnwBpoHmdI5F1XCY6Xm?usp=sharing)  
3. 模型与功能对应
    - 视频边界检测：transnetv2
    - 人脸识别：buffalo_l
    - 语音识别字幕：faster-whisper-base
    - 字幕检测：paddleocr
    - 目标检测：git-base
    - 翻译： opus-mt-en-zh
    - 镜头类型识别：pose net
4. 可能会遇到的问题
    - 如果pip install insightface出错，请手动安装。 [Link to GitHub](https://github.com/Gourieff/Assets/tree/main/Insightface)
    - 因为有些模型比较旧，所以不能使用最新的numpy，1.26.0是可用的。

[![CI Build](https://github.com/CBD-Lab/pyCinemetricsV2/actions/workflows/ci.yml/badge.svg)](https://github.com/CBD-Lab/pyCinemetricsV2/actions/workflows/ci.yml)
