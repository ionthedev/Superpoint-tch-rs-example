use clap::{Arg, ArgAction, Command};
use image::GenericImageView;
use log::info;
use std::path::Path;
use superpoint_rs::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let matches = Command::new("SuperPoint Keypoint Detector")
        .version("0.1.0")
        .author("Your Name")
        .about("Rust implementation of SuperPoint keypoint detection")
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .value_name("FILE")
                .help("Input image path")
                .required(true),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("FILE")
                .help("Output image path")
                .default_value("output_keypoints.png"),
        )
        .arg(
            Arg::new("model")
                .short('m')
                .long("model")
                .value_name("FILE")
                .help("Path to SuperPoint model (.pt file)")
                .default_value("./superpoint_v2.pt"),
        )
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Configuration file (TOML format)"),
        )
        .arg(
            Arg::new("threshold")
                .short('t')
                .long("threshold")
                .value_name("FLOAT")
                .help("Keypoint detection threshold")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("max-keypoints")
                .long("max-keypoints")
                .value_name("INT")
                .help("Maximum number of keypoints to detect")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("no-cuda")
                .long("no-cuda")
                .help("Disable CUDA acceleration")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("save-heatmap")
                .long("save-heatmap")
                .help("Save heatmap visualization")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("save-config")
                .long("save-config")
                .value_name("FILE")
                .help("Save current configuration to file"),
        )
        .get_matches();

    // Load or create configuration
    let mut config = if let Some(config_path) = matches.get_one::<String>("config") {
        println!("Loading configuration from: {}", config_path);
        Config::from_file(config_path)?
    } else if Path::new("config.toml").exists() {
        println!("Auto-detected config.toml, loading configuration...");
        Config::from_file("config.toml")?
    } else {
        println!("Using default configuration");
        Config::default()
    };

    // Print current configuration for debugging
    println!("Configuration:");
    println!("  Threshold: {}", config.keypoint.threshold);
    println!("  Max keypoints: {:?}", config.keypoint.max_keypoints);
    println!("  NMS radius: {:?}", config.keypoint.nms_radius);
    println!("  Circle radius: {}", config.visualization.circle_radius);

    // Override config with command line arguments
    if let Some(model_path) = matches.get_one::<String>("model") {
        config.model.path = model_path.into();
    }

    if let Some(&threshold) = matches.get_one::<f64>("threshold") {
        config.keypoint.threshold = threshold;
    }

    if let Some(&max_kpts) = matches.get_one::<usize>("max-keypoints") {
        config.keypoint.max_keypoints = Some(max_kpts);
    }

    if matches.get_flag("no-cuda") {
        config.model.use_cuda = false;
    }

    // Save configuration if requested
    if let Some(save_path) = matches.get_one::<String>("save-config") {
        config.to_file(save_path)?;
        println!("Configuration saved to {}", save_path);
    }

    let input_path = matches.get_one::<String>("input").unwrap();
    let output_path = matches.get_one::<String>("output").unwrap();

    // Validate input file exists
    if !Path::new(input_path).exists() {
        eprintln!("Error: Input file '{}' does not exist", input_path);
        std::process::exit(1);
    }

    // Validate model file exists
    if !config.model.path.exists() {
        eprintln!("Error: Model file '{:?}' does not exist", config.model.path);
        std::process::exit(1);
    }

    info!("Starting SuperPoint keypoint detection");
    info!("Input: {}", input_path);
    info!("Output: {}", output_path);
    info!("Model: {:?}", config.model.path);

    // Run the detection pipeline
    let result = run_detection(&config, input_path, output_path, matches.get_flag("save-heatmap"));

    match result {
        Ok(num_keypoints) => {
            println!("✅ Successfully detected {} keypoints", num_keypoints);
            println!("   Results saved to: {}", output_path);
        }
        Err(e) => {
            eprintln!("❌ Error: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

fn run_detection(
    config: &Config,
    input_path: &str,
    output_path: &str,
    save_heatmap: bool,
) -> Result<usize, SuperPointError> {
    // 1. Initialize components
    info!("Initializing SuperPoint model...");
    let model = SuperPointModel::new(config)?;
    let device = model.device();
    info!("Using device: {:?}", device);

    let preprocessor = preprocessing::ImagePreprocessor::new(config.image.clone(), device);
    let extractor = postprocessing::KeypointExtractor::new(config.keypoint.clone());
    let visualizer = visualization::Visualizer::new(config.visualization.clone());

    // 2. Load and preprocess image
    info!("Loading and preprocessing image...");
    let (input_tensor, original_image) = preprocessor.load_and_preprocess(input_path)?;
    info!("Image preprocessed. Tensor shape: {:?}", input_tensor.size());

    // 3. Run inference
    info!("Running SuperPoint inference...");
    let heatmap = model.infer(&input_tensor)?;
    info!("Inference complete. Heatmap shape: {:?}", heatmap.size());

    // 4. Extract keypoints
    info!("Extracting keypoints...");
    let keypoints_model_space = extractor.extract_keypoints(&heatmap)?;
    info!("Found {} keypoints in model space", keypoints_model_space.len());

    // 5. Scale keypoints to original image dimensions
    let original_dims = original_image.dimensions();
    let model_dims = (config.image.height, config.image.width);
    let keypoints = extractor.scale_keypoints_to_original(
        keypoints_model_space,
        original_dims,
        model_dims,
    );

    // 6. Create visualization
    info!("Creating visualization...");
    let result_image = visualizer.draw_keypoints_with_scores(&original_image, &keypoints)?;
    result_image.save(output_path)?;

    // 7. Optionally save heatmap visualization
    if save_heatmap {
        let heatmap_path = format!("{}_heatmap.png", output_path.trim_end_matches(".png"));
        info!("Saving heatmap visualization to {}...", heatmap_path);
        let heatmap_vis = visualizer.create_heatmap_visualization(&heatmap)?;
        heatmap_vis.save(&heatmap_path)?;
    }

    Ok(keypoints.len())
}
