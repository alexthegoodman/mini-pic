use serde_json;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tokenizers::models::bpe::{BpeBuilder, BPE};
use tokenizers::normalizers::{Sequence, NFD, Lowercase, StripAccents};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{AddedToken, TokenizerBuilder};

const JSON_DIR: &str = "../diffusiondb/unzipped-json/";
const OUTPUT_PATH: &str = "tokenizer.json";
const VOCAB_SIZE: usize = 8000; // BPE vocab size

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Training BPE tokenizer on diffusiondb prompts...");

    // Collect all prompts from JSON files
    let mut all_prompts = Vec::new();
    let json_path = Path::new(JSON_DIR);

    let json_files: Vec<_> = fs::read_dir(json_path)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "json")
                .unwrap_or(false)
        })
        .collect();

    println!("Reading {} JSON files...", json_files.len());

    #[derive(serde::Deserialize)]
    struct ImageMetadataJson {
        p: String, // prompt
    }

    for (idx, json_file) in json_files.iter().enumerate() {
        if idx % 2 == 0 {
            println!("Processing file {}/{}", idx + 1, json_files.len());
        }

        let content = fs::read_to_string(json_file)?;
        let data: HashMap<String, ImageMetadataJson> = serde_json::from_str(&content)?;

        for (_image_filename, metadata) in data {
            all_prompts.push(metadata.p);
        }
    }

    println!("Collected {} prompts", all_prompts.len());
    println!("Training BPE tokenizer with vocab size {}...", VOCAB_SIZE);

    // Build BPE tokenizer
    let mut builder = TokenizerBuilder::new();

    // Create BPE model
    let mut bpe_builder = BpeBuilder::new();
    bpe_builder
        .vocab_size(VOCAB_SIZE)
        .min_frequency(2)
        .show_progress(true);

    let bpe = bpe_builder
        .build_from_iterator(all_prompts.iter().map(|s| s.as_str()))?;

    builder = builder.with_model(bpe);

    // Add normalizer (NFD unicode normalization, lowercase, strip accents)
    let normalizer = Sequence::new(vec![
        Box::new(NFD),
        Box::new(Lowercase),
        Box::new(StripAccents),
    ]);
    builder = builder.with_normalizer(Some(normalizer));

    // Add pre-tokenizer (split on whitespace)
    builder = builder.with_pre_tokenizer(Some(Whitespace {}));

    // Build the tokenizer
    let mut tokenizer = builder.build()?;

    // Add special tokens
    let pad_token = AddedToken::from("[PAD]", true);
    let unk_token = AddedToken::from("[UNK]", true);
    let bos_token = AddedToken::from("[BOS]", true);
    let eos_token = AddedToken::from("[EOS]", true);

    tokenizer.add_special_tokens(&[
        pad_token.clone(),
        unk_token.clone(),
        bos_token.clone(),
        eos_token.clone(),
    ]);

    // Set up post-processor to add BOS and EOS tokens
    let post_processor = TemplateProcessing::builder()
        .try_single("[BOS] $A [EOS]")?
        .try_pair("[BOS] $A [EOS] $B:1 [EOS]:1")?
        .special_tokens(vec![
            ("[BOS]", tokenizer.token_to_id("[BOS]").unwrap() as usize),
            ("[EOS]", tokenizer.token_to_id("[EOS]").unwrap() as usize),
        ])
        .build()?;

    tokenizer.with_post_processor(post_processor);

    // Save tokenizer
    tokenizer.save(OUTPUT_PATH, false)?;
    println!("\nTokenizer saved to {}", OUTPUT_PATH);

    // Test the tokenizer
    println!("\n--- Testing tokenizer ---");
    let test_prompts = vec![
        "a beautiful landscape painting",
        "cyberpunk city at night, neon lights, highly detailed",
        "cute cat, digital art, trending on artstation",
    ];

    for prompt in test_prompts {
        let encoding = tokenizer.encode(prompt, false)?;
        println!("\nPrompt: {}", prompt);
        println!("Tokens: {:?}", encoding.get_tokens());
        println!("Token IDs: {:?}", encoding.get_ids());
        println!("Length: {}", encoding.len());
    }

    Ok(())
}
