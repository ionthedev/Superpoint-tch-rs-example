use anyhow::{Result, anyhow};
use image::{GenericImageView, RgbImage};
use tch::{CModule, Device, IValue, Kind, TchError, Tensor};

const IMAGE_HEIGHT: i64 = 240; // Set Height to 240
const IMAGE_WIDTH: i64 = 320; // Set Width to 320
// ============================
const KEYPOINT_THRESHOLD: f64 = 0.05;

fn main() -> Result<()> {
    // 1. Choose device: CPU or CUDA (if available)
    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    // 2. Paths
    let model_path = "./superpoint_v2.pt";
    let input_image = "./input.png";
    let output_image = "output_keypoints_fullres.png";

    // 3. Load the TorchScript SuperPoint model
    println!("Loading SuperPoint model from {}...", model_path);
    let model = load_superpoint_model(model_path, device)?;
    println!("Model loaded successfully.");

    // 4. Preprocess the input image into a [1, 1, 240, 320] tensor
    println!("Loading and preprocessing image {}...", input_image);
    let input_tensor = load_and_preprocess(input_image, device)?;
    println!(
        "Image preprocessed successfully. Tensor shape: {:?}",
        input_tensor.size()
    );

    // 5. Inference → get a [240, 320] heatmap
    println!("Performing inference to get keypoint heatmap...");
    let heatmap = infer_keypoints(&model, &input_tensor)?;
    println!("Inference complete. Heatmap shape: {:?}", heatmap.size());

    // 6. Extract (row, col) coordinates of keypoints in the 240×320 frame
    println!("Extracting keypoint coordinates...");
    let coords = get_keypoint_coords(&heatmap, KEYPOINT_THRESHOLD)?;
    println!("Found {} keypoints.", coords.len());

    // 7. Load the original full-resolution JPEG so we know its exact dims
    println!("Loading original image for full-res drawing...");
    let dyn_img = image::open(input_image).map_err(|e| {
        anyhow!(
            "Failed to open '{}' for full-res drawing: {}",
            input_image,
            e
        )
    })?;
    let (orig_w, orig_h) = dyn_img.dimensions();

    // 8. Scale each (row, col) from the 240×320 heatmap up to [0..orig_h, 0..orig_w]
    let heat_w = IMAGE_WIDTH as u32; // 320
    let heat_h = IMAGE_HEIGHT as u32; // 240
    let scaled_coords: Vec<(i32, i32)> = coords
        .iter()
        .map(|&(row, col)| {
            let x_full = ((col as f32) * (orig_w as f32) / (heat_w as f32)).round() as i32;
            let y_full = ((row as f32) * (orig_h as f32) / (heat_h as f32)).round() as i32;
            (x_full, y_full)
        })
        .collect();

    // 9. Convert dyn_img into an editable RgbImage so we can draw on it
    let mut fullres_rgb: RgbImage = dyn_img.to_rgb8();

    // 10. Draw red circles (radius = 4) at each scaled coordinate
    for &(x, y) in &scaled_coords {
        if x >= 0 && y >= 0 && (x as u32) < orig_w && (y as u32) < orig_h {
            fullres_rgb.put_pixel(x as u32, y as u32, image::Rgb([255, 0, 0]));
        }
    }

    // 11. Save the full-res output
    fullres_rgb
        .save(output_image)
        .map_err(|e| anyhow!("Failed to save full-res output '{}': {}", output_image, e))?;
    println!(
        "✅ Saved full-resolution keypoint visualization to {}",
        output_image
    );

    Ok(())
}

fn load_superpoint_model(model_path: &str, device: Device) -> Result<CModule> {
    let model = CModule::load_on_device(model_path, device)
        .map_err(|e| anyhow!("Failed to load model '{}': {}", model_path, e))?;
    Ok(model)
}

fn load_and_preprocess(image_path: &str, device: Device) -> Result<Tensor> {
    let img_tensor =
        tch::vision::imagenet::load_image_and_resize(image_path, IMAGE_WIDTH, IMAGE_HEIGHT)
            .map_err(|e| anyhow!("Failed to load and resize image '{}': {}", image_path, e))?;
    let gray_tensor = img_tensor.mean_dim(&[0i64] as &[i64], /*keepdim=*/ false, Kind::Float);
    let input_tensor = gray_tensor.unsqueeze(0).unsqueeze(0).to_device(device);
    Ok(input_tensor)
}

fn infer_keypoints(model: &CModule, input_tensor: &Tensor) -> Result<Tensor> {
    // 1. Run forward_is to get the raw IValue (handles multi-output)
    let output_ival: IValue = model
        .forward_is(&[IValue::Tensor(input_tensor.shallow_clone())])
        .map_err(|e| anyhow!("Model inference failed via forward_is: {}", e))?; // :contentReference[oaicite:9]{index=9}

    // 2. Extract the single “semi” heatmap tensor [1, 65, 30, 40] or [65, 30, 40]
    let semi: Tensor = match output_ival {
        IValue::Tuple(ref ivals) if ivals.len() >= 1 => match &ivals[0] {
            IValue::Tensor(t0) => t0.shallow_clone(),
            other => {
                return Err(anyhow!(
                    "Expected Tensor at tuple index 0, found: {:?}",
                    other
                ));
            }
        },
        IValue::Tensor(t) => t.shallow_clone(),
        other => {
            return Err(anyhow!(
                "Unexpected IValue from forward: {:?}. Expected Tensor or Tuple(Tensor,…).",
                other
            ));
        }
    };

    // 3. Ensure shape is [65, 30, 40]. If it’s [1, 65, 30, 40], squeeze batch dim:
    let semi = if semi.dim() == 4 && semi.size()[0] == 1 {
        semi.squeeze_dim(0)
    } else if semi.dim() == 3 {
        semi
    } else {
        return Err(anyhow!(
            "Unexpected semi-heatmap dims: {:?}. Expected [65, Hc, Wc].",
            semi.size()
        ));
    };

    // 4. Channel-wise softmax over dim=0 (65 channels → probability distribution)
    let prob = semi.softmax(0, Kind::Float); // [65, 30, 40] :contentReference[oaicite:10]{index=10}

    // 5. Slice off dustbin (last channel): keep channels 0..64 → shape [64, 30, 40]
    let prob_cells = prob.narrow(0, 0, 64);

    // 6. Depth-to-space: [64, 30, 40] → [8, 8, 30, 40]
    let reshaped = prob_cells
        .view((8, 8, 30, 40))
        .permute(&[2i64, 0, 3, 1])
        .contiguous()
        .view((30 * 8, 40 * 8));

    Ok(reshaped)
}

fn get_keypoint_coords(heatmap: &Tensor, threshold: f64) -> Result<Vec<(i64, i64)>> {
    // 1. Create a tensor of same device as heatmap with the threshold value
    let threshold_tensor = Tensor::from(threshold).to_device(heatmap.device());
    // 2. Boolean mask of shape [H, W]
    let mask = heatmap.gt_tensor(&threshold_tensor);

    // 3. Get (row, col) indices of non-zero entries → Tensor [N, 2]
    let nz_coords: Tensor = mask.nonzero();

    // 4. Flatten [N, 2] → [2*N]
    let flat_coords = nz_coords.contiguous().view((-1,));
    // 5. Move to CPU and convert to Vec<i64>
    let coords_flat: Vec<i64> =
        Vec::<i64>::try_from(flat_coords.to_device(Device::Cpu)).map_err(|e: TchError| {
            anyhow!("Failed to convert flattened coords tensor to Vec: {}", e)
        })?;

    // 6. Reconstruct Vec<(row, col)>
    let mut coords = Vec::with_capacity(coords_flat.len() / 2);
    for chunk in coords_flat.chunks_exact(2) {
        let row = chunk[0];
        let col = chunk[1];
        coords.push((row, col));
    }
    Ok(coords)
}
