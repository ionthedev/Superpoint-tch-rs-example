use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    pub score: f32,
    pub scale: Option<f32>,
    pub angle: Option<f32>,
}

impl Keypoint {
    pub fn new(x: f32, y: f32, score: f32) -> Self {
        Self {
            x,
            y,
            score,
            scale: None,
            angle: None,
        }
    }
    
    pub fn with_scale_angle(x: f32, y: f32, score: f32, scale: f32, angle: f32) -> Self {
        Self {
            x,
            y,
            score,
            scale: Some(scale),
            angle: Some(angle),
        }
    }
    
    pub fn distance_to(&self, other: &Keypoint) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

#[derive(Debug, Clone)]
pub struct KeypointMatch {
    pub keypoint1: Keypoint,
    pub keypoint2: Keypoint,
    pub distance: f32,
}

impl KeypointMatch {
    pub fn new(kp1: Keypoint, kp2: Keypoint) -> Self {
        let distance = kp1.distance_to(&kp2);
        Self {
            keypoint1: kp1,
            keypoint2: kp2,
            distance,
        }
    }
} 