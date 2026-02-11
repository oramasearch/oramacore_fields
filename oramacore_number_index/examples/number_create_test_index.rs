//! Example to create a test index for CLI testing.
//!
//! Run with: cargo run --example create_test_index

use oramacore_number_index::{IndexedValue, NumberStorage, Threshold};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from("/tmp/test_index");

    // Remove old index if exists
    let _ = std::fs::remove_dir_all(&path);

    println!("Creating test index at {}", path.display());

    let index: NumberStorage<u64> = NumberStorage::new(path.clone(), Threshold::default())?;

    // Insert values
    println!("Inserting values...");
    index.insert(&IndexedValue::Plain(100), 1)?;
    index.insert(&IndexedValue::Plain(200), 2)?;
    index.insert(&IndexedValue::Plain(300), 3)?;
    index.insert(&IndexedValue::Plain(150), 4)?;
    index.insert(&IndexedValue::Plain(250), 5)?;
    index.insert(&IndexedValue::Plain(100), 6)?; // Duplicate value
    index.insert(&IndexedValue::Plain(175), 7)?;
    index.insert(&IndexedValue::Plain(225), 8)?;
    index.insert(&IndexedValue::Plain(50), 9)?;
    index.insert(&IndexedValue::Plain(350), 10)?;

    // Delete one
    println!("Deleting doc_id 2...");
    index.delete(2);

    // Compact
    println!("Compacting...");
    index.compact(1)?;

    println!("Done! Index created at {}", path.display());
    println!("\nTest with:");
    println!("  cargo run --features cli --bin oramacore-index -- info --path /tmp/test_index");
    println!("  cargo run --features cli --bin oramacore-index -- check --path /tmp/test_index --verbose");
    println!("  cargo run --features cli --bin oramacore-index -- search --path /tmp/test_index --type u64 --gte 100 --limit 5");
    println!("  cargo run --features cli --bin oramacore-index -- search --path /tmp/test_index --type u64 --between 100,250 --format json");

    Ok(())
}
