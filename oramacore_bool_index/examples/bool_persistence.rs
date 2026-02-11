//! Persistence example for oramacore_bool_index.
//!
//! Demonstrates:
//! - Creating an index and inserting documents
//! - Calling `compact(offset)` to persist to disk
//! - Dropping the index and reopening it
//! - Verifying data survived the restart
//! - Using `info()` for statistics
//!
//! Run with: `cargo run --example persistence`

use oramacore_bool_index::{BoolStorage, DeletionThreshold, IndexedValue};
use std::env;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    // Create a temporary directory for the index
    let index_path: PathBuf = env::temp_dir().join("bool_index_persistence_example");

    // Clean up from previous runs
    if index_path.exists() {
        std::fs::remove_dir_all(&index_path)?;
    }

    println!("=== Phase 1: Create and populate index ===\n");
    println!("Index path: {}", index_path.display());

    // Create and populate the index
    {
        let index = BoolStorage::new(index_path.clone(), DeletionThreshold::default())?;

        // Insert documents
        println!("Inserting documents...");
        index.insert(&IndexedValue::Plain(true), 1);
        index.insert(&IndexedValue::Plain(true), 5);
        index.insert(&IndexedValue::Plain(true), 10);
        index.insert(&IndexedValue::Plain(false), 2);
        index.insert(&IndexedValue::Plain(false), 6);

        // Data is only in memory at this point
        let info = index.info();
        println!("\nBefore compaction:");
        println!("  Pending ops: {}", info.pending_ops);
        println!("  Compacted true count: {}", info.true_count);
        println!("  Compacted false count: {}", info.false_count);

        // Compact to persist data to disk
        println!("\nCompacting at version 1...");
        index.compact(1)?;

        // Check info after compaction
        let info = index.info();
        println!("\nAfter compaction:");
        println!("  Current version: {}", info.current_version_number);
        println!("  Compacted true count: {}", info.true_count);
        println!("  Compacted false count: {}", info.false_count);
        println!("  Total size: {} bytes", info.total_size_bytes());
        println!("  Version directory: {}", info.version_dir.display());

        // Insert more data and compact again
        println!("\nInserting more documents...");
        index.insert(&IndexedValue::Plain(true), 20);
        index.insert(&IndexedValue::Plain(true), 30);
        index.insert(&IndexedValue::Plain(false), 25);

        println!("Compacting at version 2...");
        index.compact(2)?;

        let info = index.info();
        println!("\nAfter second compaction:");
        println!("  Current version: {}", info.current_version_number);
        println!("  Compacted true count: {}", info.true_count);
        println!("  Compacted false count: {}", info.false_count);

        // Clean up old versions
        println!("\nCleaning up old versions...");
        index.cleanup();

        println!("\nDropping index (simulating application shutdown)...");
    } // index is dropped here

    println!("\n=== Phase 2: Reopen and verify ===\n");

    // Reopen the index
    {
        let index = BoolStorage::new(index_path.clone(), DeletionThreshold::default())?;

        // Check info
        let info = index.info();
        println!("Reopened index:");
        println!("  Current version: {}", info.current_version_number);
        println!("  Format version: {}", info.format_version);
        println!("  True count: {}", info.true_count);
        println!("  False count: {}", info.false_count);

        // Verify data
        let true_docs: Vec<u64> = index.filter(true).iter().collect();
        let false_docs: Vec<u64> = index.filter(false).iter().collect();

        println!("\nData verification:");
        println!("  Documents with value=true: {true_docs:?}");
        println!("  Documents with value=false: {false_docs:?}");

        // Verify expected values
        assert_eq!(
            true_docs,
            vec![1, 5, 10, 20, 30],
            "True documents mismatch!"
        );
        assert_eq!(false_docs, vec![2, 6, 25], "False documents mismatch!");
        println!("\n  Data survived the restart successfully!");
    }

    // Clean up
    std::fs::remove_dir_all(&index_path)?;
    println!("\nCleaned up index directory.");

    Ok(())
}
