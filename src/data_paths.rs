use std::path::PathBuf;

/// The downloaded DiffusionDB archives are extracted flat into `images`.
pub fn data_root() -> PathBuf {
    std::env::var_os("MINI_PIC_DATA_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("D:/DiffusionDB"))
}

pub fn source_dir() -> PathBuf {
    data_root().join("images")
}

pub fn resized_dir() -> PathBuf {
    data_root().join("images-64")
}

pub fn augmented_dir() -> PathBuf {
    data_root().join("images-64-augmented")
}
