// #![recursion_limit = "256"] // wgpu/naga auto-trait (Sync) checks overflow at the default

// use burn::backend::wgpu::{Wgpu, WgpuDevice};
// use mini_pic::inference::DiffusionInference;

// /// Usage: infer.exe "<prompt>" <model_dir> [steps] [output_path]
// ///
// /// model_dir must contain the model/config.json pair training::run() saves -
// /// its name is derived from that run's hyperparameters (see
// /// training::artifact_dir_name), so there's no fixed default; copy the
// /// "Artifact dir: ..." path training printed at the start of that run.
// fn main() {
//     let args: Vec<String> = std::env::args().collect();

//     let usage = || {
//         eprintln!("Usage: infer \"<prompt>\" <model_dir> [steps] [output_path]");
//         std::process::exit(1);
//     };

//     let prompt = args.get(1).cloned().unwrap_or_else(usage);
//     let model_dir = args.get(2).cloned().unwrap_or_else(usage);
//     let steps: usize = args
//         .get(3)
//         .and_then(|s| s.parse().ok())
//         .unwrap_or(50);
//     let output_path = args.get(4).cloned().unwrap_or_else(|| "output.png".to_string());

//     let model_path = format!("{model_dir}/model");
//     let tokenizer_path = "tokenizer.json".to_string();

//     let device = WgpuDevice::default();
//     let inference = DiffusionInference::<Wgpu>::new(&model_path, &tokenizer_path, device)
//         .expect("Failed to load model");

//     let image = inference.generate(&prompt, steps);

//     DiffusionInference::<Wgpu>::save_image(image, &output_path).expect("Failed to save image");

//     println!("Done: {}", output_path);
// }


#![recursion_limit = "256"] // wgpu/naga auto-trait (Sync) checks overflow at the default

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;
use mini_pic::inference::DiffusionInference;

/// Usage: infer_grid.exe <model_dir> [steps] [output_path]
///
/// Generates one image per hardcoded prompt in PROMPTS below and tiles them
/// into a single 4x4 grid image, saved once to `output_path`.
///
/// model_dir must contain the model/config.json pair training::run() saves -
/// its name is derived from that run's hyperparameters (see
/// training::artifact_dir_name), so there's no fixed default; copy the
/// "Artifact dir: ..." path training printed at the start of that run.

// Edit these to whatever you want to sample. Order matches the grid
// left-to-right, top-to-bottom (index 0 = row 0 col 0, index 4 = row 1 col 0,
// etc). Must have exactly GRID_ROWS * GRID_COLS entries.
const PROMPTS: [&str; 4] = [
    // "doom eternal, game concept art, veins and worms, muscular, crustacean exoskeleton",
    // "a beautiful very detailed highly detailed building ranch by frank gehry, galactic darkacademia tron dramatic lighting studio ghibli",
    // "a beautiful ultradetailed anime illustration of unknown backroom level nature by bjarke ingels",
    // "hyperrealistic portrait of a philippine baby character in a scenic environment, flowers, by beksinski zdzislaw, buchholz, quint, yoro sean",
    "low contrast, washed out, stargate made of stone that form a circle, cinematic view, epic sky, detailed, concept art, low angle, high detail, warm lighting, volumetric, godrays, vivid, beautiful, trending on artstation, by jordan grimmer",
    "upside down, flipped vertically, a wholesome animation key shot of masculine lynx - headed navigator, navigation deck of nostromo, studio ghibli, pixar and disney animation, sharp, disney concept art watercolor illustration by mandy jurgens and alphonse mucha and alena aenami, pastel color palette, dramatic lighting, highly detailed",
    // "portrait of a cloaked female devil, evil, ominous, luscious, pointy teeth, stunning, detailed, by artgerm, by greg rutkowski, by luis royo, by pixar, by myazaki, gothic, final fantasy, fantasy, medieval",
    // "low contrast, washed out, by maxfield parrish, greg manchess, mucha",
    // "very blurry, heavily out of focus, poignant portrait black and white photo of an old couple smiling at each other, nostalgia, love",
    // "dark, underexposed, a geometrical portrait of a knave, fractal flowering background, digital art, analogous colours, trending on artstation",
    // "bright, overexposed, man in tux with a giant cheeseburger head highly detailed ink drawing by junji ito",
    "realistic corgi, intricate paper quilling, swirls, spirals, white background",
    "a dog",
    // "a cat",
    // "a bird",
    // "a person",
];

// const GRID_ROWS: usize = 4;
// const GRID_COLS: usize = 4;

const GRID_ROWS: usize = 2;
const GRID_COLS: usize = 2;

fn main() {
    const _: () = assert!(PROMPTS.len() == GRID_ROWS * GRID_COLS);

    let args: Vec<String> = std::env::args().collect();

    let usage = || {
        eprintln!("Usage: infer_grid <model_dir> [steps] [output_path]");
        std::process::exit(1);
    };

    let model_dir = args.get(1).cloned().unwrap_or_else(usage);
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50);
    let output_path = args.get(3).cloned().unwrap_or_else(|| "grid.png".to_string());

    let model_path = format!("{model_dir}/model");
    let tokenizer_path = "tokenizer.json".to_string();

    let device = WgpuDevice::default();
    let inference = DiffusionInference::<Wgpu>::new(&model_path, &tokenizer_path, device)
        .expect("Failed to load model");

    // Generate sequentially (safest on a single WgpuDevice - concurrent
    // generate() calls sharing one device/queue are more likely to trip
    // over each other than save any real time) and collect the per-image
    // tensors so we tile them ourselves instead of writing 16 separate files.
    //
    // generate() returns a Tensor<Wgpu, 4> as [1, channels, height, width]
    // (a batch of one), and save_image() below expects that same 4D shape -
    // so we keep everything in 4D throughout rather than squeezing down and
    // unsqueezing back later.
    let mut images: Vec<Tensor<Wgpu, 4>> = Vec::with_capacity(PROMPTS.len());
    for (i, &prompt) in PROMPTS.iter().enumerate() {
        println!("[{:>2}/{}] generating: {}", i + 1, PROMPTS.len(), prompt);
        images.push(inference.generate(prompt, steps));
    }

    let grid = build_grid(images, GRID_ROWS, GRID_COLS);

    DiffusionInference::<Wgpu>::save_grid_image(grid, &output_path).expect("Failed to save image");

    println!("Done: {}", output_path);
    println!("\nPrompt -> grid position (row, col):");
    for (i, prompt) in PROMPTS.iter().enumerate() {
        println!("  ({}, {}) {}", i / GRID_COLS, i % GRID_COLS, prompt);
    }
}

/// Tiles `images` (each [1, channels, height, width], all the same shape)
/// into a single [1, channels, height * rows, width * cols] tensor, filled
/// left-to-right then top-to-bottom - i.e. images[0] ends up top-left.
fn build_grid(images: Vec<Tensor<Wgpu, 4>>, rows: usize, cols: usize) -> Tensor<Wgpu, 4> {
    assert_eq!(
        images.len(),
        rows * cols,
        "expected {} images for a {}x{} grid, got {}",
        rows * cols,
        rows,
        cols,
        images.len()
    );

    let row_tensors: Vec<Tensor<Wgpu, 4>> = images
        .chunks(cols)
        .map(|row| Tensor::cat(row.to_vec(), 3)) // concat left-to-right along width (dim 3)
        .collect();

    Tensor::cat(row_tensors, 2) // stack rows top-to-bottom along height (dim 2)
}