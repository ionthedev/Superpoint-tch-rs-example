# SuperPoint Build Commands

# Build release version
build:
    @echo "🔧 Building SuperPoint (Release)"
    cargo build --release

# Build debug version  
build-debug:
    @echo "🔧 Building SuperPoint (Debug)"
    cargo build

# Clean and rebuild
rebuild:
    @echo "📁 Cleaning previous build..."
    cargo clean
    @echo "🚀 Building release version..."
    cargo build --release

# Run with default parameters
run INPUT OUTPUT="output.png":
    cargo run --release -- -i {{INPUT}} -o {{OUTPUT}}

# Run with heatmap
run-heatmap INPUT OUTPUT="output.png":
    cargo run --release -- -i {{INPUT}} -o {{OUTPUT}} --save-heatmap

# Download SuperPoint model
download-model:
    @echo "📥 Downloading SuperPoint model..."
    @if command -v wget >/dev/null 2>&1; then \
        wget https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth -O superpoint_v2.pt; \
    elif command -v curl >/dev/null 2>&1; then \
        curl -L https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth -o superpoint_v2.pt; \
    else \
        echo "❌ Neither wget nor curl found. Please download manually."; \
    fi

# Run tests
test:
    cargo test

# Run clippy linter
lint:
    cargo clippy

# Format code
fmt:
    cargo fmt

# Clean build artifacts
clean:
    cargo clean

# Show help
help:
    @just --list 