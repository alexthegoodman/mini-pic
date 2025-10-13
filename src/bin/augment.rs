use image::{imageops, DynamicImage, GenericImageView, ImageFormat, Rgba};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const INPUT_DIR: &str = "../diffusiondb/unzipped-64/";
const OUTPUT_DIR: &str = "../diffusiondb/unzipped-64-augmented/";
const INPUT_JSON_DIR: &str = "../diffusiondb/unzipped-json/";
const OUTPUT_JSON_DIR: &str = "../diffusiondb/unzipped-json-augmented/";

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

    // Create output directories
    fs::create_dir_all(OUTPUT_DIR)?;
    fs::create_dir_all(OUTPUT_JSON_DIR)?;

    // Collect all JSON part files
    let json_path = Path::new(INPUT_JSON_DIR);
    let json_files: Vec<PathBuf> = fs::read_dir(json_path)?
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

    // Process each JSON part file
    json_files
        .par_iter()
        .for_each(|json_path| {
            if let Err(e) = process_part(json_path) {
                eprintln!("Error processing {:?}: {}", json_path, e);
            }
        });

    println!("Complete");
    Ok(())
}

fn process_part(
    json_path: &Path
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

    // Add original images to augmented dataset
    for (filename, metadata) in part_data.iter() {
        augmented_data.insert(filename.clone(), metadata.clone());
    }

    let mut total_generated = 0;

    // Process each image in this part
    for (filename, metadata) in part_data.iter() {
        let input_path = Path::new(INPUT_DIR).join(filename);

        // Skip if image doesn't exist
        if !input_path.exists() {
            continue;
        }

        // Load image
        let img = match image::open(&input_path) {
            Ok(img) => img,
            Err(_) => continue,
        };

        // Copy original image to augmented folder
        let output_path = Path::new(OUTPUT_DIR).join(filename);
        if let Err(_) = fs::copy(&input_path, &output_path) {
            continue;
        }

        // Generate augmentations
        let augmentations = generate_augmentations(&img, filename, metadata);

        for (aug_filename, aug_metadata, aug_img) in augmentations {
            // Save augmented image
            let output_path = Path::new(OUTPUT_DIR).join(&aug_filename);
            if let Ok(_) = aug_img.save_with_format(&output_path, ImageFormat::Png) {
                augmented_data.insert(aug_filename, aug_metadata);
                total_generated += 1;
            }
        }
    }

    // Save augmented JSON part file
    let json_filename = json_path
        .file_name()
        .ok_or("Invalid JSON filename")?;
    let output_json_path = Path::new(OUTPUT_JSON_DIR).join(json_filename);
    let output_file = File::create(output_json_path)?;
    serde_json::to_writer(output_file, &augmented_data)?;

    println!("Processed {}", part_name);

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
