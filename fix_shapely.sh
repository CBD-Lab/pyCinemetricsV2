#!/bin/bash
# 修复 Shapely 包问题的脚本

# 确保脚本在错误时退出
set -e

echo "===== 修复 Shapely 包问题 ====="

# 激活虚拟环境
echo "激活虚拟环境..."
source .venv/bin/activate

# 清理可能损坏的 Shapely 安装
echo "清理 Shapely 安装..."
rm -rf .venv/lib/python3.10/site-packages/Shapely* 2>/dev/null || true
rm -rf .venv/lib/python3.10/site-packages/shapely 2>/dev/null || true

# 重新安装 Shapely
echo "重新安装 Shapely 1.7.1 (与 PaddleOCR 兼容的版本)..."
pip install shapely==1.8.5 --no-cache-dir

echo "Shapely 修复完成！"

# 退出虚拟环境
deactivate 