#!/bin/bash
# 修复 PySide6 包问题的脚本

# 确保脚本在错误时退出
set -e

echo "===== 修复 PySide6 包问题 ====="

# 激活虚拟环境
echo "激活虚拟环境..."
source .venv/bin/activate

# 清理可能损坏的 PySide6 安装
echo "清理 PySide6 安装..."
rm -rf .venv/lib/python3.10/site-packages/PySide6* 2>/dev/null || true
rm -rf .venv/lib/python3.10/site-packages/pyside6* 2>/dev/null || true
rm -rf .venv/lib/python3.10/site-packages/shiboken6* 2>/dev/null || true

# 重新安装 PySide6
echo "重新安装 PySide6 6.5.0..."
pip install PySide6==6.5.0 --no-cache-dir --force-reinstall

# 设置Qt插件路径
echo "设置Qt插件路径..."
QT_PLUGIN_PATH=$(python -c "import sys; from PySide6 import QtCore; print(QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.PluginsPath))" 2>/dev/null || echo "无法获取Qt插件路径")

if [ "$QT_PLUGIN_PATH" != "无法获取Qt插件路径" ]; then
    echo "export QT_PLUGIN_PATH=$QT_PLUGIN_PATH" > qt_env.sh
    echo "Qt插件路径已设置为: $QT_PLUGIN_PATH"
else
    echo "警告: 无法获取Qt插件路径，可能需要手动设置"
fi

echo "PySide6 修复完成！"

# 退出虚拟环境
deactivate 