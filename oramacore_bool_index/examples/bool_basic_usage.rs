//! Basic usage example for oramacore_bool_index.
//!
//! Demonstrates:
//! - Creating a `BoolStorage` with default `DeletionThreshold`
//! - Using `BoolIndexer` to extract boolean values from JSON
//! - Inserting documents with `insert(indexed_value, doc_id)`
//! - Filtering with `filter(bool)` and iterating results
//! - Deleting documents with `delete(doc_id)`
//!
//! Run with: `cargo run --example basic_usage`

use oramacore_bool_index::{BoolIndexer, BoolStorage, DeletionThreshold};
use serde_json::json;
use std::env;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    // Create a temporary directory for the index
    let index_path: PathBuf = env::temp_dir().join("bool_index_basic_example");

    // Clean up from previous runs
    if index_path.exists() {
        std::fs::remove_dir_all(&index_path)?;
    }

    println!("Creating index at: {}", index_path.display());

    // Create a new BoolStorage with default threshold (0.1)
    let storage = BoolStorage::new(index_path.clone(), DeletionThreshold::default())?;

    // Create a BoolIndexer for plain (non-array) boolean values
    let indexer = BoolIndexer::new(false);

    // Simulate JSON documents with a boolean field
    let documents = vec![
        (1, json!(true)),
        (2, json!(false)),
        (3, json!(true)),
        (4, json!(false)),
        (5, json!(true)),
    ];

    // Index documents using BoolIndexer + BoolStorage::update
    println!("\nIndexing documents...");
    for (doc_id, value) in &documents {
        if let Some(indexed_value) = indexer.index_json(value) {
            storage.insert(&indexed_value, *doc_id);
            println!("  doc_id={doc_id} value={value}");
        }
    }

    // Query documents with value=true
    let true_docs: Vec<u64> = storage.filter(true).iter().collect();
    println!("\nDocuments with value=true: {true_docs:?}");

    // Query documents with value=false
    let false_docs: Vec<u64> = storage.filter(false).iter().collect();
    println!("Documents with value=false: {false_docs:?}");

    // Delete a document
    println!("\nDeleting document 3...");
    storage.delete(3);

    let true_docs_after_delete: Vec<u64> = storage.filter(true).iter().collect();
    println!("Documents with value=true after deletion: {true_docs_after_delete:?}");

    // --- Array indexing ---
    println!("\n--- Array indexing ---");

    let array_indexer = BoolIndexer::new(true);

    // A document with an array of booleans: [true, false, true]
    // This will insert the doc_id into BOTH the true and false sets
    let array_value = json!([true, false, true]);
    if let Some(indexed_value) = array_indexer.index_json(&array_value) {
        storage.insert(&indexed_value, 10);
        println!("  doc_id=10 value={array_value}");
    }

    // A document with all-true array
    let all_true = json!([true, true]);
    if let Some(indexed_value) = array_indexer.index_json(&all_true) {
        storage.insert(&indexed_value, 11);
        println!("  doc_id=11 value={all_true}");
    }

    // Final state
    let final_true: Vec<u64> = storage.filter(true).iter().collect();
    let final_false: Vec<u64> = storage.filter(false).iter().collect();

    println!("\nFinal state:");
    println!("  Documents with value=true: {final_true:?}");
    println!("  Documents with value=false: {final_false:?}");

    // Clean up
    std::fs::remove_dir_all(&index_path)?;
    println!("\nCleaned up index directory.");

    Ok(())
}
