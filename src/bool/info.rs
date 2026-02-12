//! Index information and integrity check types.
//!
//! This module provides structured types for index metadata and integrity
//! checking, allowing the CLI to access this information without directly
//! inspecting the filesystem.

use std::path::PathBuf;

#[cfg(feature = "cli")]
use serde::Serialize;

/// Metadata and statistics about an index.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "cli", derive(Serialize))]
pub struct IndexInfo {
    /// The file format version.
    pub format_version: u32,
    /// The current version number (from CURRENT file).
    pub current_version_number: u64,
    /// Path to the current version directory.
    pub version_dir: PathBuf,
    /// Number of documents with value=true.
    pub true_count: usize,
    /// Number of documents with value=false.
    pub false_count: usize,
    /// Number of deleted documents.
    pub deleted_count: usize,
    /// Size of true.bin in bytes.
    pub true_size_bytes: u64,
    /// Size of false.bin in bytes.
    pub false_size_bytes: u64,
    /// Size of deleted.bin in bytes.
    pub deleted_size_bytes: u64,
    /// Number of pending operations in the live layer.
    pub pending_ops: usize,
}

impl IndexInfo {
    /// Total number of documents (true + false).
    pub fn total_documents(&self) -> usize {
        self.true_count + self.false_count
    }

    /// Total size of all data files in bytes.
    pub fn total_size_bytes(&self) -> u64 {
        self.true_size_bytes + self.false_size_bytes + self.deleted_size_bytes
    }
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
}
