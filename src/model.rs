use crate::error::SuperPointError;
use crate::config::{Config, ModelConfig};
use tch::{CModule, Device, IValue, Kind, Tensor};

pub struct SuperPointModel {
    model: CModule,
    device: Device,
    config: ModelConfig,
}

impl SuperPointModel {
    pub fn new(config: &Config) -> Result<Self, SuperPointError> {
        let device = if config.model.use_cuda && Device::cuda_if_available() != Device::Cpu {
            Device::cuda_if_available()
        } else {
            Device::Cpu
        };

        let model = CModule::load_on_device(&config.model.path, device)
            .map_err(|e| SuperPointError::ModelLoading(format!("{}", e)))?;

        Ok(Self {
            model,
            device,
            config: config.model.clone(),
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn infer(&self, input_tensor: &Tensor) -> Result<Tensor, SuperPointError> {
        // Validate input tensor dimensions
        let input_dims = input_tensor.size();
        if input_dims.len() != 4 || input_dims[1] != 1 {
            return Err(SuperPointError::Inference(format!(
                "Expected input tensor shape [N, 1, H, W], got {:?}",
                input_dims
            )));
        }

        // Run inference
        let output_ival: IValue = self
            .model
            .forward_is(&[IValue::Tensor(input_tensor.shallow_clone())])
            .map_err(|e| SuperPointError::Inference(format!("Forward pass failed: {}", e)))?;

        // Extract the semi-dense heatmap tensor
        let semi: Tensor = match output_ival {
            IValue::Tuple(ref ivals) if !ivals.is_empty() => match &ivals[0] {
                IValue::Tensor(t0) => t0.shallow_clone(),
                other => {
                    return Err(SuperPointError::Inference(format!(
                        "Expected Tensor at tuple index 0, found: {:?}",
                        other
                    )));
                }
            },
            IValue::Tensor(t) => t.shallow_clone(),
            other => {
                return Err(SuperPointError::Inference(format!(
                    "Unexpected IValue from forward: {:?}. Expected Tensor or Tuple(Tensor,…).",
                    other
                )));
            }
        };

        // Ensure proper dimensions and squeeze batch dimension if needed
        let semi = if semi.dim() == 4 && semi.size()[0] == 1 {
            semi.squeeze_dim(0)
        } else if semi.dim() == 3 {
            semi
        } else {
            return Err(SuperPointError::Inference(format!(
                "Unexpected semi-heatmap dimensions: {:?}. Expected [65, Hc, Wc] or [1, 65, Hc, Wc].",
                semi.size()
            )));
        };

        // Apply softmax to get probability distribution
        let prob = semi.softmax(0, Kind::Float);

        // Remove dustbin channel (last channel)
        let prob_cells = prob.narrow(0, 0, 64);

        // Depth-to-space transformation
        let reshaped = prob_cells
            .view((8, 8, prob_cells.size()[1], prob_cells.size()[2]))
            .permute(&[2i64, 0, 3, 1])
            .contiguous()
            .view((prob_cells.size()[1] * 8, prob_cells.size()[2] * 8));

        Ok(reshaped)
    }
} 