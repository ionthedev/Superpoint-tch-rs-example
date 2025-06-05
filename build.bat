@echo off
echo 🔧 Building SuperPoint (Windows)

REM Clean previous build
echo 📁 Cleaning previous build...
cargo clean

REM Build release version
echo 🚀 Building release version...
cargo build --release

echo ✅ Build complete!
echo.
echo 📋 Quick start:
echo   cargo run --release -- -i input.png -o output.png
echo.
echo 📋 Download model (if needed):
echo   curl -L https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth -o superpoint_v2.pt
echo.
echo 📋 Binary location:
echo   .\target\release\superpoint.exe
pause 