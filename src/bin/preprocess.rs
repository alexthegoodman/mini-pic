use image::{imageops::FilterType, GenericImageView, ImageFormat};
use mini_pic::data_paths;
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};

const TARGET_SIZE: u32 = 64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting image preprocessing...");
    let input_dir = data_paths::source_dir();
    let output_dir = data_paths::resized_dir();
    println!("Input: {}", input_dir.display());
    println!("Output: {}", output_dir.display());

    // Create output directory if it doesn't exist
    fs::create_dir_all(&output_dir)?;

    // Collect all image files from input directory
    let image_files: Vec<PathBuf> = fs::read_dir(&input_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| matches!(ext.to_lowercase().as_str(), "png" | "jpg" | "jpeg"))
                .unwrap_or(false)
        })
        .collect();

    println!("Found {} images to process", image_files.len());
    if image_files.is_empty() {
        return Err(format!("No images found in {}", input_dir.display()).into());
    }

    // Process images in parallel
    let results: Vec<_> = image_files
        .par_iter()
        .enumerate()
        .map(|(idx, path)| {
            if idx % 100 == 0 {
                println!("Processing image {}/{}", idx, image_files.len());
            }
            process_image(path, &output_dir)
        })
        .collect();

    // Count successes and failures
    let (successes, failures): (Vec<_>, Vec<_>) = results.iter().partition(|r| r.is_ok());

    println!("\nProcessing complete!");
    println!("Successfully processed: {}", successes.len());
    println!("Failed: {}", failures.len());

    // Print first few failures for debugging
    if !failures.is_empty() {
        println!("\nFirst few failures:");
        for (i, result) in failures.iter().take(5).enumerate() {
            if let Err(e) = result {
                println!("  {}. {}", i + 1, e);
            }
        }
        return Err(format!("{} images failed to preprocess", failures.len()).into());
    }

    Ok(())
}

fn process_image(input_path: &Path, output_dir: &Path) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Load the image
    let img = image::open(input_path)?;

    // Get dimensions
    let (width, height) = img.dimensions();

    // Calculate the crop size (center crop to square)
    let crop_size = width.min(height);
    let x_offset = (width - crop_size) / 2;
    let y_offset = (height - crop_size) / 2;

    // Center crop to square
    let cropped = img.crop_imm(x_offset, y_offset, crop_size, crop_size);

    // Resize to 64x64 using Lanczos3 filter for high quality
    let resized = cropped.resize_exact(TARGET_SIZE, TARGET_SIZE, FilterType::Lanczos3);

    // Construct output path
    let filename = input_path
        .file_name()
        .ok_or("Invalid filename")?;
    let output_path = output_dir.join(filename);

    // Save as PNG
    resized.save_with_format(&output_path, ImageFormat::Png)?;

    Ok(())
}
