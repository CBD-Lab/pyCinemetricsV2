#!/bin/bash
# 修复Qt插件问题的脚本

# 激活虚拟环境
source .venv/bin/activate

# 查找Qt插件路径
echo "查找Qt插件路径..."
python -c "
import sys
from PySide6 import QtCore
print('Qt版本:', QtCore.__version__)
print('Qt库路径:', QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.LibrariesPath))
print('Qt插件路径:', QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath))
"

# 创建启动包装脚本
echo "创建启动包装脚本..."
cat > run_with_qt_env.sh << 'EOF'
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
EOF

chmod +x run_with_qt_env.sh

echo "修复完成！"
echo "请运行以下命令启动应用："
echo "./run_with_qt_env.sh"

# 退出虚拟环境
deactivate 