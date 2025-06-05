#!/bin/bash

# SuperPoint Wrapper Script
# This script sets up the correct library paths and runs SuperPoint

# Find the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find libtorch libraries
LIBTORCH_LIBS=$(find "$SCRIPT_DIR/target" -name "libtorch*.dylib" | head -1 | xargs dirname 2>/dev/null)

if [ -z "$LIBTORCH_LIBS" ]; then
    echo "Warning: LibTorch libraries not found. Trying to build..."
    cargo build --release
    LIBTORCH_LIBS=$(find "$SCRIPT_DIR/target" -name "libtorch*.dylib" | head -1 | xargs dirname 2>/dev/null)
fi

if [ -n "$LIBTORCH_LIBS" ]; then
    export DYLD_LIBRARY_PATH="$LIBTORCH_LIBS:$DYLD_LIBRARY_PATH"
fi

# Run SuperPoint with all passed arguments
exec "$SCRIPT_DIR/target/release/superpoint" "$@" 