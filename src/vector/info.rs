use super::config::DistanceMetric;
use std::path::PathBuf;

#[cfg(feature = "cli")]
use serde::Serialize;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "cli", derive(Serialize))]
pub struct IndexInfo {
    pub format_version: u32,
    pub current_version_number: u64,
    pub version_dir: PathBuf,
    pub num_vectors: usize,
    pub dimensions: usize,
    pub metric: DistanceMetric,
    pub pending_ops: usize,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "cli", derive(Serialize))]
pub struct IntegrityCheckResult {
    pub passed: bool,
    pub checks: Vec<IntegrityCheck>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "cli", derive(Serialize))]
pub struct IntegrityCheck {
    pub name: String,
    pub status: CheckStatus,
    pub details: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(Serialize))]
pub enum CheckStatus {
    Ok,
    Failed,
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
