#!/bin/bash
# 设置Qt环境变量并运行应用

# 激活虚拟环境
source .venv/bin/activate

# 获取Qt插件路径
QT_PLUGIN_PATH=$(python -c "from PySide6 import QtCore; print(QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath))")
QT_QPA_PLATFORM_PLUGIN_PATH=$(python -c "from PySide6 import QtCore; print(QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath) + '/platforms')")

# 设置环境变量
export QT_PLUGIN_PATH=$QT_PLUGIN_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$QT_QPA_PLATFORM_PLUGIN_PATH
export QT_DEBUG_PLUGINS=1

echo "Qt插件路径: $QT_PLUGIN_PATH"
echo "Qt平台插件路径: $QT_QPA_PLATFORM_PLUGIN_PATH"

# 运行应用
python main.py
