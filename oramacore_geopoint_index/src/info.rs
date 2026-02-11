use std::path::PathBuf;

#[derive(Debug)]
#[cfg_attr(feature = "cli", derive(serde::Serialize))]
pub struct IndexInfo {
    pub format_version: u32,
    pub current_version_id: u64,
    pub version_dir: PathBuf,
    pub segment_count: usize,
    pub deleted_count: usize,
    pub deleted_size_bytes: u64,
    pub total_points: usize,
    pub pending_inserts: usize,
    pub pending_deletes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(serde::Serialize))]
pub enum CheckStatus {
    Ok,
    Fail,
    Skip,
}

#[derive(Debug)]
#[cfg_attr(feature = "cli", derive(serde::Serialize))]
pub struct IntegrityCheck {
    pub name: String,
    pub status: CheckStatus,
    pub message: String,
}

#[derive(Debug)]
#[cfg_attr(feature = "cli", derive(serde::Serialize))]
pub struct IntegrityCheckResult {
    pub checks: Vec<IntegrityCheck>,
    pub passed: bool,
}

impl std::fmt::Display for CheckStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckStatus::Ok => write!(f, "OK"),
            CheckStatus::Fail => write!(f, "FAIL"),
            CheckStatus::Skip => write!(f, "SKIP"),
        }
    }
}
