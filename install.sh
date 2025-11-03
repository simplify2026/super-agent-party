#!/usr/bin/env bash
# 保存为 setup.sh
# 一键完成 uv sync + npm install + static/npm install

set -euo pipefail   # 任何一步失败立即退出，未定义变量/管道错误也终止

echo "[1/4] uv sync ..."
uv sync

echo "[2/4] npm install ..."
npm install

echo "[3/4] cd static && npm install ..."
[[ -d static ]] || { echo "目录 static 不存在，脚本终止"; exit 1; }
(
  cd static
  npm install
)

echo "[4/4] 全部完成！"