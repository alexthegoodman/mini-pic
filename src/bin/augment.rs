use image::{DynamicImage, ImageFormat, Rgba};
use mini_pic::data_paths;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ImageMetadata {
    p: String,  // prompt
    se: u64,    // seed
    c: f64,     // cfg_scale
    st: u32,    // steps
    sa: String, // sampler
}

type PartData = HashMap<String, ImageMetadata>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting image augmentation...");
    let input_dir = data_paths::resized_dir();
    let output_dir = data_paths::augmented_dir();
    let json_dir = data_paths::source_dir();
    println!("Images: {}", input_dir.display());
    println!("Metadata: {}", json_dir.display());
    println!("Output: {}", output_dir.display());

    // Create output directories
    fs::create_dir_all(&output_dir)?;

    // Collect all JSON part files
    let json_files: Vec<PathBuf> = fs::read_dir(&json_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "json")
                .unwrap_or(false)
        })
        .collect();

    println!("Found {} JSON part files", json_files.len());
    if json_files.is_empty() {
        return Err(format!("No JSON part files found in {}", json_dir.display()).into());
    }

    // Process each JSON part file
    let failures = AtomicUsize::new(0);
    json_files
        .par_iter()
        .for_each(|json_path| {
            if let Err(e) = process_part(json_path, &input_dir, &output_dir) {
                eprintln!("Error processing {:?}: {}", json_path, e);
                failures.fetch_add(1, Ordering::Relaxed);
            }
        });

    if failures.load(Ordering::Relaxed) > 0 {
        return Err(format!("{} parts failed augmentation", failures.load(Ordering::Relaxed)).into());
    }

    println!("Complete");
    Ok(())
}

fn process_part(
    json_path: &Path,
    input_dir: &Path,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let part_name = json_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");

    // Load JSON metadata
    let file = File::open(json_path)?;
    let reader = BufReader::new(file);
    let part_data: PartData = serde_json::from_reader(reader)?;

    // Create new augmented metadata
    let mut augmented_data: PartData = HashMap::new();

    let mut total_generated = 0;
    let mut failed_images = 0;

    // Process each image in this part
    for (filename, metadata) in part_data.iter() {
        let input_path = input_dir.join(filename);

        // Skip if image doesn't exist
        if !input_path.exists() {
            failed_images += 1;
            continue;
        }

        // Load image
        let img = match image::open(&input_path) {
            Ok(img) => img,
            Err(_) => {
                failed_images += 1;
                continue;
            }
        };

        // Copy original image to augmented folder
        let output_path = output_dir.join(filename);
        if let Err(_) = fs::copy(&input_path, &output_path) {
            failed_images += 1;
            continue;
        }
        augmented_data.insert(filename.clone(), metadata.clone());

        // Generate augmentations
        let augmentations = generate_augmentations(&img, filename, metadata);

        for (aug_filename, aug_metadata, aug_img) in augmentations {
            // Save augmented image
            let output_path = output_dir.join(&aug_filename);
            match aug_img.save_with_format(&output_path, ImageFormat::Png) {
                Ok(()) => {
                    augmented_data.insert(aug_filename, aug_metadata);
                    total_generated += 1;
                }
                Err(_) => failed_images += 1,
            }
        }
    }

    // Save augmented JSON part file
    let json_filename = json_path
        .file_name()
        .ok_or("Invalid JSON filename")?;
    let output_json_path = output_dir.join(json_filename);
    let output_file = File::create(output_json_path)?;
    serde_json::to_writer(output_file, &augmented_data)?;

    println!("Processed {}: {} variants, {} failures", part_name, total_generated, failed_images);
    if failed_images > 0 {
        return Err(format!("{} images or variants failed in {}", failed_images, part_name).into());
    }

    Ok(())
}

fn generate_augmentations(
    img: &DynamicImage,
    original_filename: &str,
    metadata: &ImageMetadata,
) -> Vec<(String, ImageMetadata, DynamicImage)> {
    let mut augmentations = Vec::new();
    let base_name = original_filename.trim_end_matches(".png");

    // 1. Horizontal flip
    let flipped = img.fliph();
    augmentations.push((
        format!("{}_flip.png", base_name),
        add_prompt_prefix(metadata, "flipped, mirrored, "),
        flipped,
    ));

    // 2. Vertical flip
    let vflipped = img.flipv();
    augmentations.push((
        format!("{}_vflip.png", base_name),
        add_prompt_prefix(metadata, "upside down, flipped vertically, "),
        vflipped,
    ));

    // 3. Rotation 90 degrees
    let rot90 = img.rotate90();
    augmentations.push((
        format!("{}_rot90.png", base_name),
        add_prompt_prefix(metadata, "rotated 90 degrees, sideways, "),
        rot90,
    ));

    // 4. Rotation 180 degrees
    let rot180 = img.rotate180();
    augmentations.push((
        format!("{}_rot180.png", base_name),
        add_prompt_prefix(metadata, "rotated 180 degrees, upside down, "),
        rot180,
    ));

    // 5. Rotation 270 degrees
    let rot270 = img.rotate270();
    augmentations.push((
        format!("{}_rot270.png", base_name),
        add_prompt_prefix(metadata, "rotated 270 degrees, sideways, "),
        rot270,
    ));

    // 6. Blur
    let blurred = img.blur(2.0);
    augmentations.push((
        format!("{}_blur.png", base_name),
        add_prompt_prefix(metadata, "blurry, out of focus, "),
        blurred,
    ));

    // 7. Strong blur
    let strong_blur = img.blur(4.0);
    augmentations.push((
        format!("{}_strongblur.png", base_name),
        add_prompt_prefix(metadata, "very blurry, heavily out of focus, "),
        strong_blur,
    ));

    // 8. Brightness adjustments
    let brightened = adjust_brightness(img, 40);
    augmentations.push((
        format!("{}_bright.png", base_name),
        add_prompt_prefix(metadata, "bright, overexposed, "),
        brightened,
    ));

    let darkened = adjust_brightness(img, -40);
    augmentations.push((
        format!("{}_dark.png", base_name),
        add_prompt_prefix(metadata, "dark, underexposed, "),
        darkened,
    ));

    // 9. Grayscale
    let grayscale = DynamicImage::ImageLuma8(img.to_luma8());
    augmentations.push((
        format!("{}_gray.png", base_name),
        add_prompt_prefix(metadata, "black and white, grayscale, monochrome, "),
        grayscale,
    ));

    // 10. High contrast
    let high_contrast = adjust_contrast(img, 1.5);
    augmentations.push((
        format!("{}_hcontrast.png", base_name),
        add_prompt_prefix(metadata, "high contrast, "),
        high_contrast,
    ));

    // 11. Low contrast
    let low_contrast = adjust_contrast(img, 0.5);
    augmentations.push((
        format!("{}_lcontrast.png", base_name),
        add_prompt_prefix(metadata, "low contrast, washed out, "),
        low_contrast,
    ));

    augmentations
}

fn add_prompt_prefix(metadata: &ImageMetadata, prefix: &str) -> ImageMetadata {
    let mut new_metadata = metadata.clone();
    new_metadata.p = format!("{}{}", prefix, metadata.p);
    new_metadata
}

fn adjust_brightness(img: &DynamicImage, value: i32) -> DynamicImage {
    let mut result = img.to_rgba8();
    for pixel in result.pixels_mut() {
        let r = (pixel[0] as i32 + value).clamp(0, 255) as u8;
        let g = (pixel[1] as i32 + value).clamp(0, 255) as u8;
        let b = (pixel[2] as i32 + value).clamp(0, 255) as u8;
        *pixel = Rgba([r, g, b, pixel[3]]);
    }
    DynamicImage::ImageRgba8(result)
}

fn adjust_contrast(img: &DynamicImage, factor: f32) -> DynamicImage {
    let mut result = img.to_rgba8();
    for pixel in result.pixels_mut() {
        let r = ((pixel[0] as f32 - 128.0) * factor + 128.0).clamp(0.0, 255.0) as u8;
        let g = ((pixel[1] as f32 - 128.0) * factor + 128.0).clamp(0.0, 255.0) as u8;
        let b = ((pixel[2] as f32 - 128.0) * factor + 128.0).clamp(0.0, 255.0) as u8;
        *pixel = Rgba([r, g, b, pixel[3]]);
    }
    DynamicImage::ImageRgba8(result)
}
