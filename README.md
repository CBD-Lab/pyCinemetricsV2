# pyCinemetrics_V2.0 
[![CI Build](https://github.com/CBD-Lab/pyCinemetricsV2/actions/workflows/ci.yml/badge.svg)](https://github.com/CBD-Lab/pyCinemetricsV2/actions/workflows/ci.yml)
## Paper
https://www.sciencedirect.com/science/article/pii/S2352711025002651
@article{LI2025102299,
title = {PyCinemetricsV2: Interactive computational film software based on transformers and PySide6},
journal = {SoftwareX},
volume = {31},
pages = {102299},
year = {2025},
issn = {2352-7110},
doi = {https://doi.org/10.1016/j.softx.2025.102299},
url = {https://www.sciencedirect.com/science/article/pii/S2352711025002651},
author = {Chunfang Li and Yalv Fan and Yushi Shen and Kun Wang and Yuhe Hu and Fei Zhang and Yuchen Pei and Tongtong Zheng and Zhuoqi Shi},
keywords = {Git-base, Faster-whisper, PaddleOCR, InsightFace, PySide6},
abstract = {Based on feedback from film scholars, we developed PyCinemetricsV2, an upgraded film analytics software. Built on pre-trained AIGC models and PySide6, it offers features such as removing false positives and adding false negatives after TransNetV2 Shot-Boundary Detection (SBD), as well as generating configurable shot mosaic. Using the multimodal Git-base model, it provides descriptions of shot frames and enables open-domain object recognition. Using Faster-Whisper, it delivers more accurate dialog recognition, while PaddleOCR is employed to identify film metadata and intertitles. Based on InsightFace, facial recognition is implemented for the shot-first frames or for fine-grained frames, generating a list of credits. The case study demonstrates its accuracy, efficiency, and improved user experience in film analytics.}
}
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

