@echo off
setlocal enableextensions

set "IMAGE_NAME=hetero-fedlearn-worker-dfs:test"
set "CONTAINER_NAME=worker-node"
set "WORKER_ID=%~1"
set "HOST_PORT=%~2"

if "%WORKER_ID%"=="" set "WORKER_ID=worker-node"
if "%HOST_PORT%"=="" set "HOST_PORT=5000"

docker info >nul 2>&1
if errorlevel 1 (
    echo Please start Docker Desktop before running start_worker.bat
    exit /b 1
)

if not exist storage mkdir storage

docker rm -f %CONTAINER_NAME% >nul 2>&1
docker build -t %IMAGE_NAME% -f worker\Dockerfile_extended worker
if errorlevel 1 exit /b 1

docker run -d --restart unless-stopped ^
  --name %CONTAINER_NAME% ^
  -e WORKER_ID=%WORKER_ID% ^
  -p %HOST_PORT%:5000 ^
  -v "%cd%\storage:/app/datanode_storage" ^
  %IMAGE_NAME%
if errorlevel 1 exit /b 1

echo DFS-lite worker started on http://127.0.0.1:%HOST_PORT%
start http://127.0.0.1:%HOST_PORT%
exit /b 0
