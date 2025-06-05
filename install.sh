#!/bin/bash

set -e

echo "🚀 Building SuperPoint Rust..."

# Clean and build
cargo clean
cargo build --release

echo "📦 Setting up library dependencies..."

# Find libtorch libraries
LIBTORCH_LIBS=$(find target -name "libtorch*.dylib" 2>/dev/null | head -1 | xargs dirname)

if [ -n "$LIBTORCH_LIBS" ]; then
    echo "Found libtorch libraries in: $LIBTORCH_LIBS"
    
    # Create deps directory in release if it doesn't exist
    mkdir -p target/release/deps
    
    # Copy all libtorch libraries to the release deps directory
    echo "Copying libraries..."
    find target -name "libtorch*.dylib" -exec cp {} target/release/deps/ \; 2>/dev/null || true
    find target -name "*torch*.dylib" -exec cp {} target/release/deps/ \; 2>/dev/null || true
    find target -name "*omp*.dylib" -exec cp {} target/release/deps/ \; 2>/dev/null || true
    
    # Also check common system locations for OpenMP
    for omp_path in "/usr/local/lib/libomp.dylib" "/opt/homebrew/lib/libomp.dylib"; do
        if [ -f "$omp_path" ]; then
            echo "Found OpenMP at $omp_path, copying..."
            cp "$omp_path" target/release/deps/ 2>/dev/null || true
        fi
    done
    
    # Update binary to look in the right place
    if command -v install_name_tool >/dev/null 2>&1; then
        echo "Updating library paths in binary..."
        for lib in target/release/deps/*.dylib; do
            if [ -f "$lib" ]; then
                lib_name=$(basename "$lib")
                install_name_tool -change "@rpath/$lib_name" "@executable_path/deps/$lib_name" target/release/superpoint 2>/dev/null || true
            fi
        done
    fi
    
    echo "✅ Library setup complete"
else
    echo "⚠️  Warning: Could not find libtorch libraries"
    echo "   The binary may need to be run with cargo run"
fi

# Make the binary executable
chmod +x target/release/superpoint

echo "🎉 Installation complete!"
echo ""
echo "Usage:"
echo "  ./target/release/superpoint -i input.png -o output.png"
echo "  ./target/release/superpoint -i input.png -c config.toml"
echo ""
echo "If the binary doesn't work, use:"
echo "  cargo run --release -- -i input.png -o output.png" 