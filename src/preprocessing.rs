use crate::config::ImageConfig;
use crate::error::SuperPointError;
use image::{DynamicImage, GrayImage, ImageBuffer};
use tch::{Device, Tensor};

pub struct ImagePreprocessor {
    config: ImageConfig,
    device: Device,
}

impl ImagePreprocessor {
    pub fn new(config: ImageConfig, device: Device) -> Self {
        Self { config, device }
    }

    pub fn load_and_preprocess(&self, image_path: &str) -> Result<(Tensor, DynamicImage), SuperPointError> {
        // Load the original image for later use
        let original_image = image::open(image_path)
            .map_err(|e| SuperPointError::ImageProcessing(format!("Failed to load image '{}': {}", image_path, e)))?;
        
        // Create tensor for model input
        let tensor = self.create_tensor_from_image(&original_image)?;
        
        Ok((tensor, original_image))
    }
    
    pub fn create_tensor_from_image(&self, image: &DynamicImage) -> Result<Tensor, SuperPointError> {
        // Convert to grayscale
        let gray_image = image.to_luma8();
        
        // Resize to model input dimensions
        let resized = image::imageops::resize(
            &gray_image,
            self.config.width as u32,
            self.config.height as u32,
            image::imageops::FilterType::Lanczos3,
        );
        
        // Convert to tensor
        let tensor = self.image_to_tensor(&resized)?;
        
        Ok(tensor)
    }
    
    fn image_to_tensor(&self, image: &GrayImage) -> Result<Tensor, SuperPointError> {
        let (width, height) = image.dimensions();
        let pixels: Vec<f32> = image
            .pixels()
            .map(|pixel| {
                let value = pixel[0] as f32 / 255.0;
                if self.config.normalize {
                    // Normalize to [-1, 1] or [0, 1] depending on model requirements
                    value
                } else {
                    pixel[0] as f32
                }
            })
            .collect();

        let tensor = Tensor::from_slice(&pixels)
            .view((1, 1, height as i64, width as i64))
            .to_device(self.device);

        Ok(tensor)
    }
    
    pub fn tensor_to_image(&self, tensor: &Tensor) -> Result<GrayImage, SuperPointError> {
        let tensor_cpu = tensor.to_device(Device::Cpu);
        let dims = tensor_cpu.size();
        
        if dims.len() != 2 {
            return Err(SuperPointError::ImageProcessing(
                "Expected 2D tensor for image conversion".to_string()
            ));
        }
        
        let height = dims[0] as u32;
        let width = dims[1] as u32;
        
        let data: Vec<f32> = Vec::try_from(tensor_cpu)
            .map_err(|e| SuperPointError::ImageProcessing(format!("Failed to convert tensor to vec: {}", e)))?;
        
        let pixels: Vec<u8> = data
            .iter()
            .map(|&value| (value.clamp(0.0, 1.0) * 255.0) as u8)
            .collect();
        
        ImageBuffer::from_raw(width, height, pixels)
            .ok_or_else(|| SuperPointError::ImageProcessing("Failed to create image buffer".to_string()))
    }
} 