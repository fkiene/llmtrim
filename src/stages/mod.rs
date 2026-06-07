//! Compression stages.

pub mod cache;
pub mod dedup;
pub mod hygiene;
pub mod image;
pub mod ngram;
pub mod output;
pub mod retrieve;
pub mod serialize;
pub mod skeleton;
pub mod tools;

pub use cache::CacheStage;
pub use dedup::DedupStage;
pub use hygiene::HygieneStage;
pub use image::ImageStage;
pub use ngram::NgramStage;
pub use output::OutputControlStage;
pub use retrieve::RetrieveStage;
pub use serialize::SerializeStage;
pub use skeleton::{MinifyCodeStage, SkeletonStage};
pub use tools::ToolStage;
