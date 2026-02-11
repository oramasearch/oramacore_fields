//! Example to create a test f64 index for CLI testing.
//!
//! Run with: cargo run --example create_f64_test_index

use oramacore_number_index::{IndexedValue, NumberStorage, Threshold};
use std::{f64, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from("/tmp/test_index_f64");

    // Remove old index if exists
    let _ = std::fs::remove_dir_all(&path);

    println!("Creating f64 test index at {}", path.display());

    let index: NumberStorage<f64> = NumberStorage::new(path.clone(), Threshold::default())?;

    // Insert values
    println!("Inserting values...");
    index.insert(&IndexedValue::Plain(-100.5), 1)?;
    index.insert(&IndexedValue::Plain(0.0), 2)?;
    index.insert(&IndexedValue::Plain(f64::consts::PI), 3)?;
    index.insert(&IndexedValue::Plain(f64::consts::E), 4)?;
    index.insert(&IndexedValue::Plain(f64::consts::SQRT_2), 5)?;
    index.insert(&IndexedValue::Plain(-50.25), 6)?;
    index.insert(&IndexedValue::Plain(100.0), 7)?;

    // Compact
    println!("Compacting...");
    index.compact(1)?;

    println!("Done! f64 index created at {}", path.display());
    println!("\nTest with:");
    println!("  cargo run --features cli --bin oramacore-index -- search --path /tmp/test_index_f64 --type f64 --gte 0.0");
    println!("  cargo run --features cli --bin oramacore-index -- search --path /tmp/test_index_f64 --type f64 --between -50.0,50.0");

    Ok(())
}
