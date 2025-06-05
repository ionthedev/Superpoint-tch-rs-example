use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub model: ModelConfig,
    pub image: ImageConfig,
    pub keypoint: KeypointConfig,
    pub visualization: VisualizationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub path: PathBuf,
    pub use_cuda: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageConfig {
    pub width: i64,
    pub height: i64,
    pub normalize: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeypointConfig {
    pub threshold: f64,
    pub max_keypoints: Option<usize>,
    pub nms_radius: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    pub circle_radius: u32,
    pub circle_color: [u8; 3],
    pub line_thickness: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: ModelConfig {
                path: PathBuf::from("./superpoint_v2.pt"),
                use_cuda: true,
            },
            image: ImageConfig {
                width: 320,
                height: 240,
                normalize: true,
            },
            keypoint: KeypointConfig {
                threshold: 0.05,
                max_keypoints: Some(1000),
                nms_radius: Some(4.0),
            },
            visualization: VisualizationConfig {
                circle_radius: 3,
                circle_color: [255, 0, 0],
                line_thickness: 2,
            },
        }
    }
}

impl Config {
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
    
    pub fn to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
} 