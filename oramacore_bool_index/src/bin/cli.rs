//! CLI tool for inspecting and validating bool-index data.

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};

#[cfg(test)]
use oramacore_bool_index::IndexedValue;
use oramacore_bool_index::{BoolStorage, CheckStatus, DeletionThreshold};

/// CLI tool for inspecting and querying oramacore bool indexes.
#[derive(Parser)]
#[command(name = "bool-index-cli")]
#[command(about = "Inspect, validate, and query oramacore bool indexes")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Check index integrity and validate files
    Check {
        /// Path to the index directory
        path: PathBuf,

        /// Show detailed validation information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Display index information and statistics
    Info {
        /// Path to the index directory
        path: PathBuf,

        /// Output format
        #[arg(short, long, value_enum, default_value = "human")]
        format: OutputFormat,
    },

    /// Display index data (for debugging)
    Show {
        /// Path to the index directory
        path: PathBuf,

        /// List all document IDs in each category
        #[arg(long)]
        list_ids: bool,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum OutputFormat {
    Human,
    Json,
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Check { path, verbose } => cmd_check(&path, verbose),
        Commands::Info { path, format } => cmd_info(&path, format),
        Commands::Show { path, list_ids } => cmd_show(&path, list_ids),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn cmd_check(path: &Path, verbose: bool) -> Result<(), String> {
    let index = BoolStorage::new(path.to_path_buf(), DeletionThreshold::default())
        .map_err(|e| format!("Failed to open index: {e}"))?;

    let result = index.integrity_check();

    for check in &result.checks {
        let status_str = match check.status {
            CheckStatus::Ok => "[OK]",
            CheckStatus::Failed => "[FAIL]",
            CheckStatus::Skipped => "[SKIP]",
        };

        if check.status == CheckStatus::Failed || verbose {
            if let Some(ref details) = check.details {
                println!("{status_str} {}: {details}", check.name);
            } else {
                println!("{status_str} {}", check.name);
            }
        }
    }

    if result.passed {
        println!("\nIndex validation PASSED");
        Ok(())
    } else {
        println!("\nIndex validation FAILED");
        std::process::exit(1);
    }
}

fn cmd_info(path: &Path, format: OutputFormat) -> Result<(), String> {
    let index = BoolStorage::new(path.to_path_buf(), DeletionThreshold::default())
        .map_err(|e| format!("Failed to open index: {e}"))?;

    let info = index.info();

    match format {
        OutputFormat::Human => {
            println!("Index Information");
            println!("=================");
            println!("Format version:  {}", info.format_version);
            println!("Current version:  {}", info.current_version_number);
            println!("Version dir:     {}", info.version_dir.display());
            println!();
            println!("True documents:  {}", info.true_count);
            println!("False documents: {}", info.false_count);
            println!("Deleted entries: {}", info.deleted_count);
            println!("Total documents: {}", info.total_documents());
            println!();
            println!("True size:       {} bytes", info.true_size_bytes);
            println!("False size:      {} bytes", info.false_size_bytes);
            println!("Deleted size:    {} bytes", info.deleted_size_bytes);
            println!("Total size:      {} bytes", info.total_size_bytes());
            println!();
            println!("Pending ops:     {}", info.pending_ops);
        }
        OutputFormat::Json => {
            #[cfg(feature = "cli")]
            {
                let json = serde_json::to_string_pretty(&info).map_err(|e| e.to_string())?;
                println!("{json}");
            }
            #[cfg(not(feature = "cli"))]
            {
                return Err("JSON output requires the 'cli' feature".to_string());
            }
        }
    }

    Ok(())
}

fn cmd_show(path: &Path, list_ids: bool) -> Result<(), String> {
    let index = BoolStorage::new(path.to_path_buf(), DeletionThreshold::default())
        .map_err(|e| format!("Failed to open index: {e}"))?;

    let info = index.info();

    println!("Format version: {}", info.format_version);
    println!("Current version: {}", info.current_version_number);
    println!();

    println!("Statistics:");
    println!(
        "  True postings:  {} documents ({} bytes)",
        info.true_count, info.true_size_bytes
    );
    println!(
        "  False postings: {} documents ({} bytes)",
        info.false_count, info.false_size_bytes
    );
    println!(
        "  Deleted:        {} documents ({} bytes)",
        info.deleted_count, info.deleted_size_bytes
    );
    println!("  Total documents: {}", info.total_documents());

    if list_ids {
        let true_ids: Vec<u64> = index.filter(true).iter().collect();
        let false_ids: Vec<u64> = index.filter(false).iter().collect();

        println!();
        println!("True document IDs:");
        for id in &true_ids {
            println!("  {id}");
        }

        println!();
        println!("False document IDs:");
        for id in &false_ids {
            println!("  {id}");
        }

        // Note: deleted IDs aren't directly accessible via filter, but we show the count above
        println!();
        println!("Deleted document IDs: (count: {})", info.deleted_count);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_index() -> (TempDir, PathBuf) {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().to_path_buf();

        let index = BoolStorage::new(path.clone(), DeletionThreshold::default()).unwrap();
        index.insert(&IndexedValue::Plain(true), 1);
        index.insert(&IndexedValue::Plain(true), 5);
        index.insert(&IndexedValue::Plain(true), 10);
        index.insert(&IndexedValue::Plain(false), 2);
        index.insert(&IndexedValue::Plain(false), 6);
        index.compact(1).unwrap();

        (tmp, path)
    }

    #[test]
    fn test_cmd_info() {
        let (_tmp, path) = create_test_index();
        cmd_info(&path, OutputFormat::Human).unwrap();
    }

    #[test]
    fn test_cmd_show() {
        let (_tmp, path) = create_test_index();
        cmd_show(&path, false).unwrap();
    }

    #[test]
    fn test_cmd_show_with_list_ids() {
        let (_tmp, path) = create_test_index();
        cmd_show(&path, true).unwrap();
    }

    #[test]
    fn test_cmd_check() {
        let (_tmp, path) = create_test_index();
        cmd_check(&path, false).unwrap();
    }

    #[test]
    fn test_cmd_check_verbose() {
        let (_tmp, path) = create_test_index();
        cmd_check(&path, true).unwrap();
    }

    #[test]
    fn test_cmd_check_uncompacted_index() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().to_path_buf();
        // Opening an empty directory works, but integrity check fails
        // because CURRENT file doesn't exist until compaction
        // Note: cmd_check calls process::exit on failure, so we test the result is Err
        // or for this test, we check that the index itself validates correctly after compact
        let index = BoolStorage::new(path.clone(), DeletionThreshold::default()).unwrap();
        let result = index.integrity_check();
        assert!(!result.passed); // Should fail - no CURRENT file

        // Insert something so compact actually writes version files
        index.insert(&IndexedValue::Plain(true), 1);
        index.compact(1).unwrap();
        let result = index.integrity_check();
        assert!(result.passed);
    }

    #[test]
    fn test_cmd_check_wrong_format_version() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().to_path_buf();

        // Create a valid index first
        let index = BoolStorage::new(path.clone(), DeletionThreshold::default()).unwrap();
        index.insert(&IndexedValue::Plain(true), 1);
        index.compact(1).unwrap();
        drop(index);

        // Manually write a wrong format version to CURRENT
        let current_path = path.join("CURRENT");
        std::fs::write(&current_path, "999\n1").unwrap();

        // Opening should fail due to format version mismatch
        let result = cmd_check(&path, false);
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("Unsupported format version"));
    }
}
