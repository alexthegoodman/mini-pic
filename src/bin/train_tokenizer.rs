use mini_pic::data_paths;
use std::collections::HashMap;
use std::fs;
use tokenizers::normalizers::{NormalizerWrapper, Sequence, Lowercase, StripAccents};
use tokenizers::normalizers::unicode::{NFD};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::decoders::wordpiece::WordPiece as DecoderWordPiece;
use tokenizers::models::wordpiece::{WordPiece, WordPieceTrainerBuilder};
use tokenizers::{AddedToken, Error, TokenizerBuilder};

const OUTPUT_PATH: &str = "tokenizer.json";
const VOCAB_SIZE: usize = 4096; // WordPiece vocab size

fn main() -> Result<(), Error> {
    println!("Training WordPiece tokenizer on DiffusionDB prompts...");

    // Collect all prompts from JSON files
    let mut all_prompts = Vec::new();
    let source_dir = data_paths::source_dir();
    let json_path = source_dir.as_path();
    println!("Metadata: {}", json_path.display());

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
    if json_files.is_empty() {
        return Err(format!("No JSON part files found in {}", json_path.display()).into());
    }

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
    println!("Training WordPiece tokenizer with vocab size {}...", VOCAB_SIZE);

    // // Build BPE tokenizer
    // let mut builder = TokenizerBuilder::new();

    // // Create BPE model
    // let mut bpe_builder = BpeTrainerBuilder::new();
    // bpe_builder
    //     .vocab_size(VOCAB_SIZE)
    //     .min_frequency(2)
    //     .show_progress(true);

    // // let bpe = bpe_builder
    // //     .build_from_iterator(all_prompts.iter().map(|s| s.as_str()))?;

    // builder = builder.with_model(bpe_builder);

    // // Add normalizer (NFD unicode normalization, lowercase, strip accents)
    // let normalizer = Sequence::new(vec![
    //     Box::new(NFD),
    //     Box::new(Lowercase),
    //     Box::new(StripAccents),
    // ]);
    // builder = builder.with_normalizer(Some(normalizer));

    // // Add pre-tokenizer (split on whitespace)
    // builder = builder.with_pre_tokenizer(Some(Whitespace {}));

    // // Build the tokenizer
    // let mut tokenizer = builder.build()?;

    // // Add special tokens
    // let pad_token = AddedToken::from("[PAD]", true);
    // let unk_token = AddedToken::from("[UNK]", true);
    // let bos_token = AddedToken::from("[BOS]", true);
    // let eos_token = AddedToken::from("[EOS]", true);

    // tokenizer.add_special_tokens(&[
    //     pad_token.clone(),
    //     unk_token.clone(),
    //     bos_token.clone(),
    //     eos_token.clone(),
    // ]);

     // Set up post-processor to add BOS and EOS tokens
    let normalizer = Sequence::new(vec![
        NormalizerWrapper::NFD(NFD),
        // Box::new(Lowercase),
        // Box::new(StripAccents),
        NormalizerWrapper::Lowercase(Lowercase),
        NormalizerWrapper::StripAccents(StripAccents),
    ]);

    let mut trainer = WordPieceTrainerBuilder::new()
        .vocab_size(VOCAB_SIZE) // Keep it small but enough for your patterns
        .special_tokens(vec![
            AddedToken::from(String::from("[PAD]"), true),
            AddedToken::from(String::from("[UNK]"), true),
            AddedToken::from(String::from("[BOS]"), true),
            AddedToken::from(String::from("[EOS]"), true),
        ])
        .build();

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(WordPiece::default())
        .with_normalizer(Some(normalizer))
        // Use Digits pre-tokenizer to handle numbers better
        // .with_pre_tokenizer(Some(Digits::new(true)))
        .with_pre_tokenizer(Some(Whitespace {}))
        .with_post_processor(Some(TemplateProcessing::default()))
        .with_decoder(Some(DecoderWordPiece::default()))
        .build()?;

    // // Save tokenizer
    // tokenizer.save(OUTPUT_PATH, false)?;
    // println!("\nTokenizer saved to {}", OUTPUT_PATH);

    tokenizer.train(&mut trainer, all_prompts.into_iter())?;
    let bos_id = tokenizer.token_to_id("[BOS]").ok_or("BOS token missing")?;
    let eos_id = tokenizer.token_to_id("[EOS]").ok_or("EOS token missing")?;
    let post_processor = TemplateProcessing::builder()
        .try_single("[BOS] $A [EOS]")?
        .try_pair("[BOS] $A [EOS] $B:1 [EOS]:1")?
        .special_tokens(vec![("[BOS]", bos_id), ("[EOS]", eos_id)])
        .build()?;
    tokenizer.with_post_processor(Some(post_processor));
    tokenizer.save(OUTPUT_PATH, false)?;

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
