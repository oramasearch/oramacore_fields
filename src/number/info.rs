//! Index information and integrity check types.

use std::path::PathBuf;

#[cfg(feature = "cli")]
use serde::Serialize;

/// Metadata and statistics about an index.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "cli", derive(Serialize))]
pub struct IndexInfo {
    /// The file format version.
    pub format_version: u32,
    /// The current version offset (from CURRENT file).
    pub current_offset: u64,
    /// Path to the current version directory.
    pub version_dir: PathBuf,
    /// Number of header entries in header.idx.
    pub header_entry_count: usize,
    /// Number of deleted doc_ids in deleted.bin.
    pub deleted_count: usize,
    /// Number of data files (data_XXXX.dat).
    pub data_file_count: usize,
    /// Size of header.idx in bytes.
    pub header_size_bytes: u64,
    /// Size of deleted.bin in bytes.
    pub deleted_size_bytes: u64,
    /// Total size of all data files in bytes.
    pub data_total_bytes: u64,
    /// Number of pending inserts in the live layer.
    pub pending_inserts: usize,
    /// Number of pending deletes in the live layer.
    pub pending_deletes: usize,
}

/// Result of an integrity check operation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "cli", derive(Serialize))]
pub struct IntegrityCheckResult {
    /// Whether all checks passed.
    pub passed: bool,
    /// Individual check results.
    pub checks: Vec<IntegrityCheck>,
}

/// A single integrity check result.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "cli", derive(Serialize))]
pub struct IntegrityCheck {
    /// Name of the check.
    pub name: String,
    /// Status of the check.
    pub status: CheckStatus,
    /// Optional details about the check result.
    pub details: Option<String>,
}

/// Status of an integrity check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(Serialize))]
pub enum CheckStatus {
    /// Check passed.
    Ok,
    /// Check failed.
    Failed,
    /// Check was skipped.
    Skipped,
}

impl IntegrityCheckResult {
    /// Create a new IntegrityCheckResult with the given checks.
    pub fn new(checks: Vec<IntegrityCheck>) -> Self {
        let passed = checks.iter().all(|c| c.status != CheckStatus::Failed);
        Self { passed, checks }
    }
}

impl IntegrityCheck {
    /// Create a new passing check.
    pub fn ok(name: impl Into<String>, details: Option<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Ok,
            details,
        }
    }

    /// Create a new failing check.
    pub fn failed(name: impl Into<String>, details: Option<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Failed,
            details,
        }
    }

    /// Create a new skipped check.
    pub fn skipped(name: impl Into<String>, details: Option<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Skipped,
            details,
        }
    }
}
