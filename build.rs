fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    
    // The tch crate with download-libtorch feature handles PyTorch installation
    // automatically across all platforms. No additional configuration needed.
    
    // On macOS, add some common rpath locations for better compatibility
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
    }
} 