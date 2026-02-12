use std::path::PathBuf;

/// Metadata and size statistics about a string filter index.
#[derive(Debug, Clone)]
pub struct IndexInfo {
    pub format_version: u32,
    pub current_version_number: u64,
    pub version_dir: PathBuf,
    pub unique_keys_count: usize,
    pub total_postings_count: usize,
    pub deleted_count: usize,
    pub fst_size_bytes: u64,
    pub postings_size_bytes: u64,
    pub deleted_size_bytes: u64,
    pub pending_ops: usize,
}

impl IndexInfo {
    pub fn total_size_bytes(&self) -> u64 {
        self.fst_size_bytes + self.postings_size_bytes + self.deleted_size_bytes
    }
}

/// Result of an integrity check operation.
#[derive(Debug, Clone)]
pub struct IntegrityCheckResult {
    pub passed: bool,
    pub checks: Vec<IntegrityCheck>,
}

/// A single integrity check result.
#[derive(Debug, Clone)]
pub struct IntegrityCheck {
    pub name: String,
    pub status: CheckStatus,
    pub details: Option<String>,
}

/// Status of an integrity check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckStatus {
    Ok,
    Failed,
    Skipped,
}

impl IntegrityCheckResult {
    pub fn new(checks: Vec<IntegrityCheck>) -> Self {
        let passed = checks.iter().all(|c| c.status != CheckStatus::Failed);
        Self { passed, checks }
    }
}

impl IntegrityCheck {
    pub fn ok(name: impl Into<String>, details: Option<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Ok,
            details,
        }
    }

    pub fn failed(name: impl Into<String>, details: Option<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Failed,
            details,
        }
    }
}
