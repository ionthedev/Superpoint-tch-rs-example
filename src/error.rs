use thiserror::Error;
use tch::TchError;

#[derive(Error, Debug)]
pub enum SuperPointError {
    #[error("Model loading failed: {0}")]
    ModelLoading(String),
    
    #[error("Image processing failed: {0}")]
    ImageProcessing(String),
    
    #[error("Inference failed: {0}")]
    Inference(String),
    
    #[error("Tensor operation failed: {0}")]
    TensorOp(#[from] TchError),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),
    
    #[error("Invalid configuration: {0}")]
    Config(String),
    
    #[error("Keypoint extraction failed: {0}")]
    KeypointExtraction(String),
} 