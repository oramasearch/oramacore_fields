//! Basic usage example for oramacore_number_index.
//!
//! This example demonstrates the core functionality of the library:
//! - Creating a new U64Storage
//! - Inserting values with document IDs
//! - Querying with various FilterOp variants
//! - Deleting documents
//! - Compacting the index
//! - Reopening an existing index

use oramacore_number_index::{FilterOp, IndexedValue, NumberStorage, Threshold, U64Storage};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let index_path = PathBuf::from("/tmp/oramacore_example_index");

    // Clean up any previous run
    let _ = std::fs::remove_dir_all(&index_path);

    println!("Creating a new U64Storage...");
    let index: U64Storage = NumberStorage::new(index_path.clone(), Threshold::default())?;

    // Insert some values (value, doc_id)
    println!("Inserting 5 documents...");
    index.insert(&IndexedValue::Plain(100), 1)?;
    index.insert(&IndexedValue::Plain(200), 2)?;
    index.insert(&IndexedValue::Plain(300), 3)?;
    index.insert(&IndexedValue::Plain(150), 4)?;
    index.insert(&IndexedValue::Plain(250), 5)?;

    // Query with various FilterOp variants
    println!("Running queries...");
    let eq: Vec<u64> = index.filter(FilterOp::Eq(200)).iter().collect();
    let gt: Vec<u64> = index.filter(FilterOp::Gt(200)).iter().collect();
    let gte: Vec<u64> = index.filter(FilterOp::Gte(200)).iter().collect();
    let lt: Vec<u64> = index.filter(FilterOp::Lt(200)).iter().collect();
    let lte: Vec<u64> = index.filter(FilterOp::Lte(200)).iter().collect();
    let between: Vec<u64> = index
        .filter(FilterOp::BetweenInclusive(150, 250))
        .iter()
        .collect();

    println!("  Eq(200): {eq:?}");
    println!("  Gt(200): {gt:?}");
    println!("  Gte(200): {gte:?}");
    println!("  Lt(200): {lt:?}");
    println!("  Lte(200): {lte:?}");
    println!("  BetweenInclusive(150, 250): {between:?}");

    // Delete a document
    println!("Deleting document 2...");
    index.delete(2);
    let after_delete: Vec<u64> = index.filter(FilterOp::Gte(100)).iter().collect();
    println!("  Documents after delete: {after_delete:?}");

    // Compact the index
    println!("Compacting the index...");
    index.compact(1)?;

    drop(index); // Close the index

    // Reopen the existing index to verify persistence
    println!("\nReopening the index to verify data integrity...");
    let index: U64Storage = NumberStorage::new(index_path.clone(), Threshold::default())?;

    let results: Vec<u64> = index.filter(FilterOp::Gte(100)).iter().collect();
    println!("Matching elements: {:?}", results.len());

    // Clean up
    drop(index);
    let _ = std::fs::remove_dir_all(&index_path);

    println!("\nExample completed successfully!");

    Ok(())
}
