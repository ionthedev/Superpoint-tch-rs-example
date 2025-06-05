#!/bin/bash
set -e

echo "🔧 Building SuperPoint (Cross-Platform)"
echo "Platform: $(uname -s)"

# Clean previous build
echo "📁 Cleaning previous build..."
cargo clean

# Build release version
echo "🚀 Building release version..."
cargo build --release

echo "✅ Build complete!"
echo ""
echo "📋 Quick start:"
echo "  cargo run --release -- -i input.png -o output.png"
echo ""
echo "📋 Download model (if needed):"
echo "  wget https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth -O superpoint_v2.pt"
echo ""
echo "📋 Binary location:"
echo "  ./target/release/superpoint" 