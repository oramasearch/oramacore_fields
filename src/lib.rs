//! On-disk field indexes for search engines.
//!
//! Provides five specialized index types with a shared architecture: an in-memory
//! live layer for fast writes, memory-mapped compacted storage for efficient reads,
//! and periodic compaction with concurrent access.
//!
//! # Modules
//!
//! | Module | Type | Query Operations |
//! |--------|------|-----------------|
//! | [`mod@bool`] | Boolean (true/false) | Filter by value |
//! | [`number`] | Numeric (u64, f64) | eq, gt, gte, lt, lte, between |
//! | [`string`] | Full-text (BM25 scoring) | Search by tokens (exact, fuzzy, prefix) |
//! | [`string_filter`] | String (exact match) | Filter by key |
//! | [`geopoint`] | Geographic (lat/lon) | Bounding box, radius |
//!
//! # Usage
//!
//! Each index type provides an **Indexer** that extracts values from JSON and a
//! **Storage** that persists and queries them. The typical flow is:
//!
//! 1. Create a `*Indexer` to parse JSON values
//! 2. Create a `*Storage` for on-disk persistence
//! 3. Use `indexer.index_json(json)` to extract an `IndexedValue`
//! 4. Insert the `IndexedValue` into the storage with a document ID
//! 5. Query the storage with filter operations
//!
//! # Architecture
//!
//! All five modules share a common two-layer design:
//!
//! - **LiveLayer**: In-memory storage for fast inserts and deletes. Thread-safe via `RwLock`.
//! - **CompactedVersion**: Memory-mapped disk storage loaded via `memmap2`. Swapped
//!   atomically using `ArcSwap` for lock-free reads during compaction.
//! - **Compaction**: Periodically merges live data into a new on-disk version, enabling
//!   persistence without blocking reads or writes.

pub mod bool;
pub mod geopoint;
pub mod number;
pub mod string;
pub mod string_filter;

#[doc = include_str!("../README.md")]
#[cfg(doctest)]
struct _ReadmeDocTests;
