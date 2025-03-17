#!/bin/bash
# 修复Mac上的问题

# 确保脚本在错误时退出
set -e

echo "===== 修复PyCinemetricsV2 Mac问题 ====="

# 激活虚拟环境
echo "激活虚拟环境..."
source .venv/bin/activate

# 修复shapely版本问题
echo "修复shapely版本问题..."
pip uninstall -y shapely
pip install shapely==1.8.5

# 修复PySide6问题
echo "修复PySide6问题..."
pip uninstall -y PySide6 PySide6_Addons PySide6_Essentials
pip install PySide6==6.5.0 # 使用较旧但稳定的版本

# 确保VLC和python-vlc兼容
echo "确保VLC和python-vlc兼容..."
pip uninstall -y python-vlc
pip install python-vlc

# 设置Qt插件路径
echo "设置Qt插件路径..."
QT_PLUGIN_PATH=$(python -c "import sys; from PySide6 import QtCore; print(QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.PluginsPath))")
echo "export QT_PLUGIN_PATH=$QT_PLUGIN_PATH" > qt_env.sh

echo "修复完成！"
echo "请运行以下命令启动应用："
echo "source qt_env.sh && ./run_macos.sh"

# 退出虚拟环境
deactivate 