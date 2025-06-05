use crate::config::VisualizationConfig;
use crate::error::SuperPointError;
use crate::keypoint::Keypoint;
use image::{DynamicImage, Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut};

pub struct Visualizer {
    config: VisualizationConfig,
}

impl Visualizer {
    pub fn new(config: VisualizationConfig) -> Self {
        Self { config }
    }

    pub fn draw_keypoints(&self, image: &DynamicImage, keypoints: &[Keypoint]) -> Result<RgbImage, SuperPointError> {
        let mut rgb_image = image.to_rgb8();
        let color = Rgb(self.config.circle_color);
        
        for keypoint in keypoints {
            // Ensure coordinates are within image bounds
            let x = keypoint.x.round() as i32;
            let y = keypoint.y.round() as i32;
            
            if x >= 0 && y >= 0 && (x as u32) < rgb_image.width() && (y as u32) < rgb_image.height() {
                // Draw filled circle for keypoint
                draw_filled_circle_mut(
                    &mut rgb_image,
                    (x, y),
                    self.config.circle_radius as i32,
                    color,
                );
                
                // Optionally draw a cross for better visibility
                self.draw_cross(&mut rgb_image, x, y, color);
            }
        }
        
        Ok(rgb_image)
    }
    
    pub fn draw_keypoints_with_scores(&self, image: &DynamicImage, keypoints: &[Keypoint]) -> Result<RgbImage, SuperPointError> {
        let mut rgb_image = image.to_rgb8();
        
        for keypoint in keypoints {
            let x = keypoint.x.round() as i32;
            let y = keypoint.y.round() as i32;
            
            if x >= 0 && y >= 0 && (x as u32) < rgb_image.width() && (y as u32) < rgb_image.height() {
                // Color intensity based on score (higher score = brighter red)
                let intensity = (keypoint.score.clamp(0.0, 1.0) * 255.0) as u8;
                let color = Rgb([intensity, 0, 0]);
                
                draw_filled_circle_mut(
                    &mut rgb_image,
                    (x, y),
                    self.config.circle_radius as i32,
                    color,
                );
                
                self.draw_cross(&mut rgb_image, x, y, color);
            }
        }
        
        Ok(rgb_image)
    }
    
    pub fn draw_keypoint_matches(
        &self,
        image1: &DynamicImage,
        image2: &DynamicImage,
        keypoints1: &[Keypoint],
        keypoints2: &[Keypoint],
        matches: &[(usize, usize)],
    ) -> Result<RgbImage, SuperPointError> {
        let img1 = image1.to_rgb8();
        let img2 = image2.to_rgb8();
        
        let (w1, h1) = img1.dimensions();
        let (w2, h2) = img2.dimensions();
        
        // Create combined image (side by side)
        let combined_width = w1 + w2;
        let combined_height = h1.max(h2);
        
        let mut combined = RgbImage::new(combined_width, combined_height);
        
        // Copy first image
        for (x, y, pixel) in img1.enumerate_pixels() {
            combined.put_pixel(x, y, *pixel);
        }
        
        // Copy second image (offset by width of first image)
        for (x, y, pixel) in img2.enumerate_pixels() {
            combined.put_pixel(x + w1, y, *pixel);
        }
        
        // Draw keypoints
        let kp_color = Rgb(self.config.circle_color);
        for kp in keypoints1 {
            let x = kp.x.round() as i32;
            let y = kp.y.round() as i32;
            if x >= 0 && y >= 0 && (x as u32) < w1 && (y as u32) < h1 {
                draw_filled_circle_mut(&mut combined, (x, y), self.config.circle_radius as i32, kp_color);
            }
        }
        
        for kp in keypoints2 {
            let x = (kp.x.round() as u32 + w1) as i32;
            let y = kp.y.round() as i32;
            if x >= w1 as i32 && y >= 0 && (x as u32) < combined_width && (y as u32) < h2 {
                draw_filled_circle_mut(&mut combined, (x, y), self.config.circle_radius as i32, kp_color);
            }
        }
        
        // Draw match lines
        let line_color = Rgb([0, 255, 0]); // Green for matches
        for &(idx1, idx2) in matches {
            if idx1 < keypoints1.len() && idx2 < keypoints2.len() {
                let kp1 = &keypoints1[idx1];
                let kp2 = &keypoints2[idx2];
                
                let start = (kp1.x.round() as f32, kp1.y.round() as f32);
                let end = (kp2.x.round() as f32 + w1 as f32, kp2.y.round() as f32);
                
                draw_line_segment_mut(&mut combined, start, end, line_color);
            }
        }
        
        Ok(combined)
    }
    
    fn draw_cross(&self, image: &mut RgbImage, x: i32, y: i32, color: Rgb<u8>) {
        let size = (self.config.circle_radius / 2).max(1) as i32;
        
        // Horizontal line
        for dx in -size..=size {
            let px = x + dx;
            if px >= 0 && (px as u32) < image.width() && y >= 0 && (y as u32) < image.height() {
                image.put_pixel(px as u32, y as u32, color);
            }
        }
        
        // Vertical line
        for dy in -size..=size {
            let py = y + dy;
            if x >= 0 && (x as u32) < image.width() && py >= 0 && (py as u32) < image.height() {
                image.put_pixel(x as u32, py as u32, color);
            }
        }
    }
    
    pub fn create_heatmap_visualization(&self, heatmap_tensor: &tch::Tensor) -> Result<RgbImage, SuperPointError> {
        use tch::Device;
        
        // Convert heatmap to CPU and extract values
        let heatmap_cpu = heatmap_tensor.to_device(Device::Cpu);
        let dims = heatmap_cpu.size();
        
        if dims.len() != 2 {
            return Err(SuperPointError::ImageProcessing(
                "Expected 2D heatmap tensor".to_string()
            ));
        }
        
        let height = dims[0] as u32;
        let width = dims[1] as u32;
        
        let data: Vec<f32> = Vec::try_from(heatmap_cpu)
            .map_err(|e| SuperPointError::ImageProcessing(format!("Failed to convert heatmap: {}", e)))?;
        
        // Find min/max for normalization
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;
        
        let mut heatmap_image = RgbImage::new(width, height);
        
        for (i, &value) in data.iter().enumerate() {
            let x = (i as u32) % width;
            let y = (i as u32) / width;
            
            // Normalize to [0, 1]
            let normalized = if range > 0.0 {
                (value - min_val) / range
            } else {
                0.5
            };
            
            // Convert to heatmap color (blue -> red)
            let color = self.value_to_heatmap_color(normalized);
            heatmap_image.put_pixel(x, y, color);
        }
        
        Ok(heatmap_image)
    }
    
    fn value_to_heatmap_color(&self, value: f32) -> Rgb<u8> {
        let value = value.clamp(0.0, 1.0);
        
        if value < 0.5 {
            // Blue to green
            let t = value * 2.0;
            Rgb([0, (t * 255.0) as u8, ((1.0 - t) * 255.0) as u8])
        } else {
            // Green to red
            let t = (value - 0.5) * 2.0;
            Rgb([(t * 255.0) as u8, ((1.0 - t) * 255.0) as u8, 0])
        }
    }
} 