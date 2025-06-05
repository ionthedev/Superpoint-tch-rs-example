# SuperPoint Keypoint Detection in Rust

A robust Rust implementation of the SuperPoint keypoint detection algorithm with modern architecture and cross-platform support.

## Features

- 🏗️ **Modular Architecture**: Clean separation of concerns across multiple modules
- ⚙️ **Flexible Configuration**: TOML-based configuration with runtime overrides
- 🖥️ **CLI Interface**: Full command-line interface with comprehensive options
- 🎯 **Advanced Processing**: Non-Maximum Suppression (NMS) and parallel processing
- 🎨 **Rich Visualization**: Score-based keypoint colors and heatmap generation
- 🛡️ **Robust Error Handling**: Custom error types with detailed messages
- 🌍 **Cross-Platform**: Works on Windows, macOS, and Linux

## Prerequisites

- Rust (latest stable version)
- Git

No additional dependencies required! PyTorch is automatically downloaded during build.

## Quick Start

### 1. Clone and Build

```bash
git clone <repository-url>
cd Superpoint-tch-rs-example
cargo build --release
```

### 2. Download a SuperPoint Model

```bash
# Download the official SuperPoint model
wget https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth -O superpoint_v2.pt
```

### 3. Run Keypoint Detection

```bash
# Basic usage
cargo run --release -- -i input.png -o output.png

# With heatmap visualization
cargo run --release -- -i input.png -o output.png --save-heatmap

# Custom parameters
cargo run --release -- -i input.png -o output.png -t 0.01 --max-keypoints 500
```

## Command Line Options

```
SuperPoint Keypoint Detector

USAGE:
    superpoint [OPTIONS] --input <FILE>

OPTIONS:
    -i, --input <FILE>           Input image path (required)
    -o, --output <FILE>          Output image path [default: output_keypoints.png]
    -m, --model <FILE>           Path to SuperPoint model (.pt file) [default: ./superpoint_v2.pt]
    -c, --config <FILE>          Configuration file (TOML format)
    -t, --threshold <FLOAT>      Keypoint detection threshold
        --max-keypoints <INT>    Maximum number of keypoints to detect
        --no-cuda               Disable CUDA acceleration
        --save-heatmap          Save heatmap visualization
        --save-config <FILE>    Save current configuration to file
    -h, --help                  Print help information
    -V, --version               Print version information
```

## Configuration

Create a `config.toml` file in the project root for persistent settings:

```toml
[model]
path = "./superpoint_v2.pt"
use_cuda = true

[image]
width = 640
height = 480

[keypoint]
threshold = 0.05
max_keypoints = 1000
nms_radius = 4.0

[visualization]
circle_radius = 1
```

## Architecture

The project is organized into focused modules:

- **`lib.rs`** - Public API and module exports
- **`error.rs`** - Custom error types with thiserror
- **`config.rs`** - TOML configuration management
- **`model.rs`** - SuperPoint model wrapper
- **`keypoint.rs`** - Keypoint data structures
- **`preprocessing.rs`** - Image preprocessing pipeline
- **`postprocessing.rs`** - Keypoint extraction and NMS
- **`visualization.rs`** - Advanced visualization features

## Cross-Platform Notes

### Windows
- Works out of the box with the MSVC toolchain
- PyTorch CPU version is automatically downloaded
- For GPU support, ensure CUDA is installed

### macOS  
- Works on both Intel and Apple Silicon
- No additional dependencies required
- Metal GPU acceleration supported on Apple Silicon

### Linux
- Works with any modern Linux distribution
- For GPU support, install appropriate CUDA drivers
- Ubuntu/Debian: All dependencies handled automatically

## Build Options

```bash
# Debug build (faster compilation, slower runtime)
cargo build

# Release build (slower compilation, faster runtime)
cargo build --release

# Clean rebuild
cargo clean && cargo build --release

# Run with logging
RUST_LOG=info cargo run --release -- -i input.png -o output.png
```

## Development

### Running Tests
```bash
cargo test
```

### Linting
```bash
cargo clippy
```

### Formatting
```bash
cargo fmt
```

## Performance

- **Parallel Processing**: Keypoint extraction uses rayon for multi-core processing
- **Memory Efficient**: Optimized tensor operations with minimal allocations
- **NMS Filtering**: Fast non-maximum suppression for keypoint deduplication
- **GPU Support**: Automatic CUDA acceleration when available

## Troubleshooting

### "Model file not found"
Ensure the SuperPoint model file exists at the specified path:
```bash
ls -la superpoint_v2.pt
```

### "Image loading failed"
Verify the input image format is supported (PNG, JPEG, etc.):
```bash
file input.png
```

### Build Issues
Clean and rebuild:
```bash
cargo clean
cargo build --release
```

Bombardino Crocodilo's days are numbered....

Links
---
[https://huggingface.co/magic-leap-community/superpoint](https://huggingface.co/magic-leap-community/superpoint)

[https://crates.io/crates/tch](https://crates.io/crates/tch)
