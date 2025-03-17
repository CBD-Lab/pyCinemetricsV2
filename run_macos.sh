#!/bin/bash
# macOS运行脚本 - 用于直接运行PyCinemetricsV2（不打包）

# 确保脚本在错误时退出
set -e

echo "===== 启动PyCinemetricsV2 ====="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 找不到python3命令"
    echo "请安装Python 3.10或更高版本"
    exit 1
fi

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv .venv
    
    # 激活虚拟环境
    echo "激活虚拟环境..."
    source .venv/bin/activate
    
    # 安装依赖
    echo "安装依赖..."
    pip install -r requirements.txt
else
    # 激活虚拟环境
    echo "激活虚拟环境..."
    source .venv/bin/activate
fi

# 自动设置 Qt 环境变量
echo "设置 Qt 环境变量..."
if [ -f "qt_env.sh" ]; then
    source qt_env.sh
else
    # 如果 qt_env.sh 不存在，则自动生成
    QT_PLUGIN_PATH=$(python -c "import sys; from PySide6 import QtCore; print(QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.PluginsPath))" 2>/dev/null || echo "")
    if [ -n "$QT_PLUGIN_PATH" ]; then
        echo "export QT_PLUGIN_PATH=$QT_PLUGIN_PATH" > qt_env.sh
        export QT_PLUGIN_PATH=$QT_PLUGIN_PATH
        echo "Qt 插件路径已设置为: $QT_PLUGIN_PATH"
    else
        echo "警告: 无法自动设置 Qt 环境变量，可能需要运行 fix_mac_issues.sh"
    fi
fi

# 检查Homebrew
if ! command -v brew &> /dev/null; then
    echo "警告: 找不到Homebrew，这可能会导致VLC和FFmpeg安装问题"
    echo "建议安装Homebrew: https://brew.sh/"
else
    # 检查VLC
    if ! brew list --formula | grep -q vlc; then
        echo "警告: 未安装VLC，视频播放可能无法正常工作"
        echo "建议运行: brew install vlc"
    fi
    
    # 检查FFmpeg
    if ! brew list --formula | grep -q ffmpeg; then
        echo "警告: 未安装FFmpeg，视频处理可能无法正常工作"
        echo "建议运行: brew install ffmpeg"
    fi
fi

# 检查models文件夹
if [ ! -d "models" ]; then
    echo "警告: 找不到models文件夹"
    echo "请从网盘下载models文件夹并放置在项目根目录"
    echo "百度网盘: https://pan.baidu.com/s/1GMlOYvglimvSoIcIowuM0A?pwd=1234"
    echo "Google Drive: https://drive.google.com/drive/folders/1ho48Bx6KF-fZewnwBpoHmdI5F1XCY6Xm?usp=sharing"
fi

# 确保img目录存在
mkdir -p img

# 运行应用
echo "启动应用..."
python main.py

# 退出虚拟环境
deactivate 