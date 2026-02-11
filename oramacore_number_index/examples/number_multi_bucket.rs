//! Example demonstrating multi-bucket configuration.
//!
//! This example shows how to configure the index with custom bucket sizes
//! to create multiple data files (buckets) during compaction.

use oramacore_number_index::{FilterOp, IndexedValue, NumberStorage, Threshold, U64Storage};
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let index_path = PathBuf::from("/tmp/oramacore_multi_bucket_example");

    // Clean up any previous run
    let _ = fs::remove_dir_all(&index_path);

    println!("=== Creating U64Storage with small bucket size ===\n");

    // Each entry in the data file consists of:
    // - DataEntryHeader: 32 bytes (key: 8, next: 8, prev: 8, count: 8)
    // - doc_ids: 8 bytes each
    //
    // So a single (value, doc_id) pair uses 32 + 8 = 40 bytes
    //
    // By setting bucket_target_bytes to 100, we force a new bucket
    // to be created after approximately every 2 entries.

    let index_stride = 1; // Create header entry for each key (for demonstration)
    let bucket_target_bytes = 100; // Very small: ~2 entries per bucket

    let index: U64Storage = NumberStorage::new_with_config(
        index_path.clone(),
        Threshold::default(),
        index_stride,
        bucket_target_bytes,
    )?;

    println!("Configuration:");
    println!("  - index_stride: {index_stride} (header entry every N doc_ids)");
    println!("  - bucket_target_bytes: {bucket_target_bytes} bytes");
    println!("  - Entry size: 40 bytes (32-byte header + 8-byte doc_id)");

    // Insert 10 entries - this should create multiple buckets
    println!("Inserting 10 entries...");
    for i in 1..=10u64 {
        index.insert(&IndexedValue::Plain(i * 100), i)?;
    }

    // Compact to write data to disk
    println!("Compacting the index...");
    index.compact(1)?;

    drop(index); // Close the index to ensure all data is flushed
                 // Reopen the index to verify data integrity
    println!("\nReopening the index to verify data integrity...");
    let index: U64Storage = NumberStorage::new_with_config(
        index_path.clone(),
        Threshold::default(),
        index_stride,
        bucket_target_bytes,
    )?;

    // Verify data integrity across all buckets
    println!("Verifying data integrity...");
    let all_results: Vec<u64> = index.filter(FilterOp::Gte(0)).iter().collect();
    println!("Matching elements: {:?}", all_results.len());

    // Clean up
    drop(index);
    let _ = fs::remove_dir_all(&index_path);

    println!("\nExample completed successfully!");

    Ok(())
}
