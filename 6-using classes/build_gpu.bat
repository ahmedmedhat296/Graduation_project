@echo off
echo Setting up MSVC environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"

echo Building PPO GPU Agent for EGX Market...
echo GPU: GTX 1650 (Compute 7.5), CUDA 13.1
echo.

nvcc -o ppo_gpu.exe ^
     ppo_main.cpp ^
     src/GpuNet.cu ^
     src/GpuPPOAgent.cu ^
     src/TradingEnv.cpp ^
     src/matrix.cpp ^
     src/layer.cpp ^
     src/net.cpp ^
     src/trainSet.cpp ^
     -I include ^
     -std=c++17 ^
     -O3 ^
     -arch=sm_75 ^
     -lcublas ^
     -Xcompiler "/O2 /EHsc" ^
     2>&1

if %errorlevel%==0 (
    echo.
    echo Build SUCCESSFUL: ppo_gpu.exe
    echo.
    echo Usage:
    echo   .\ppo_gpu.exe --train    Train the agent (ESC to stop)
    echo   .\ppo_gpu.exe --test     Test on most recent EGX data
) else (
    echo.
    echo Build FAILED. Check errors above.
)
pause
