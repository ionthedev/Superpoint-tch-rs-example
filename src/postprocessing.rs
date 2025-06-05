use crate::config::KeypointConfig;
use crate::error::SuperPointError;
use crate::keypoint::Keypoint;
use rayon::prelude::*;
use tch::{Device, Tensor};

pub struct KeypointExtractor {
    config: KeypointConfig,
}

impl KeypointExtractor {
    pub fn new(config: KeypointConfig) -> Self {
        Self { config }
    }

    pub fn extract_keypoints(&self, heatmap: &Tensor) -> Result<Vec<Keypoint>, SuperPointError> {
        // 1. Threshold-based filtering
        let mut keypoints = self.extract_candidates(heatmap)?;
        
        // 2. Apply Non-Maximum Suppression if configured
        if let Some(nms_radius) = self.config.nms_radius {
            keypoints = self.apply_nms(keypoints, nms_radius);
        }
        
        // 3. Limit number of keypoints if configured
        if let Some(max_kpts) = self.config.max_keypoints {
            keypoints.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            keypoints.truncate(max_kpts);
        }
        
        Ok(keypoints)
    }
    
    fn extract_candidates(&self, heatmap: &Tensor) -> Result<Vec<Keypoint>, SuperPointError> {
        // Create threshold tensor on same device as heatmap
        let threshold_tensor = Tensor::from(self.config.threshold).to_device(heatmap.device());
        
        // Boolean mask of pixels above threshold
        let mask = heatmap.gt_tensor(&threshold_tensor);
        
        // Get coordinates of non-zero entries
        let nz_coords = mask.nonzero();
        
        // Convert to CPU for processing
        let coords_cpu = nz_coords.to_device(Device::Cpu);
        let heatmap_cpu = heatmap.to_device(Device::Cpu);
        
        // Extract coordinate pairs and scores
        let coords_data: Vec<i64> = Vec::try_from(coords_cpu.contiguous().view((-1,)))
            .map_err(|e| SuperPointError::KeypointExtraction(format!("Failed to extract coordinates: {}", e)))?;
        
        let mut keypoints = Vec::with_capacity(coords_data.len() / 2);
        
        for chunk in coords_data.chunks_exact(2) {
            let row = chunk[0];
            let col = chunk[1];
            
            // Get the score at this position
            let score_tensor = heatmap_cpu.get(row).get(col);
            let score: f32 = f32::try_from(score_tensor)
                .map_err(|e| SuperPointError::KeypointExtraction(format!("Failed to extract score: {}", e)))?;
            
            keypoints.push(Keypoint::new(col as f32, row as f32, score));
        }
        
        Ok(keypoints)
    }
    
    fn apply_nms(&self, mut keypoints: Vec<Keypoint>, radius: f32) -> Vec<Keypoint> {
        // Sort by score (descending)
        keypoints.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        let mut suppressed = vec![false; keypoints.len()];
        let mut result = Vec::new();
        
        for i in 0..keypoints.len() {
            if suppressed[i] {
                continue;
            }
            
            result.push(keypoints[i].clone());
            
            // Suppress nearby keypoints
            for j in (i + 1)..keypoints.len() {
                if !suppressed[j] {
                    let distance = keypoints[i].distance_to(&keypoints[j]);
                    if distance < radius {
                        suppressed[j] = true;
                    }
                }
            }
        }
        
        result
    }
    
    pub fn scale_keypoints_to_original(
        &self,
        keypoints: Vec<Keypoint>,
        original_size: (u32, u32),
        model_size: (i64, i64),
    ) -> Vec<Keypoint> {
        let scale_x = original_size.0 as f32 / model_size.1 as f32;
        let scale_y = original_size.1 as f32 / model_size.0 as f32;
        
        keypoints
            .into_par_iter()
            .map(|mut kp| {
                kp.x *= scale_x;
                kp.y *= scale_y;
                kp
            })
            .collect()
    }
}