@echo off
rem 保存为例如：setup.bat
rem 一键完成 uv sync + npm install + static\npm install

echo [1/4] uv sync ...
uv sync
if errorlevel 1 (
    echo uv sync 失败，脚本终止
    pause
    exit /b 1
)

echo [2/4] npm install ...
call npm install
if errorlevel 1 (
    echo 根目录 npm install 失败，脚本终止
    pause
    exit /b 1
)

echo [3/4] cd static && npm install ...
if not exist "static" (
    echo 目录 static 不存在，脚本终止
    pause
    exit /b 1
)
pushd static
call npm install
if errorlevel 1 (
    echo static 目录 npm install 失败，脚本终止
    pause
    popd
    exit /b 1
)

echo [4/4] 返回上级目录
popd

echo 全部完成！