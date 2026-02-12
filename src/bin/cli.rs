//! Unified CLI tool for inspecting and querying oramacore field indexes.

use std::path::{Path, PathBuf};
use std::process;

use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;

/// CLI tool for inspecting and querying oramacore field indexes.
#[derive(Parser)]
#[command(name = "oramacore_fields")]
#[command(about = "Inspect, validate, and query oramacore field indexes")]
struct Cli {
    #[command(subcommand)]
    command: TopLevel,
}

#[derive(Subcommand)]
enum TopLevel {
    /// Inspect and query bool indexes
    Bool {
        #[command(subcommand)]
        command: BoolCommands,
    },
    /// Inspect and query number indexes
    Number {
        #[command(subcommand)]
        command: NumberCommands,
    },
    /// Inspect and query geopoint indexes
    Geopoint {
        #[command(subcommand)]
        command: GeoCommands,
    },
}

// ── Output format (shared) ──────────────────────────────────────────────────

#[derive(Clone, Copy, ValueEnum)]
enum OutputFormat {
    Human,
    Json,
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Bool subcommands
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Subcommand)]
enum BoolCommands {
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

fn bool_check(path: &Path, verbose: bool) -> Result<(), String> {
    use oramacore_fields::bool::{BoolStorage, CheckStatus, DeletionThreshold};

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
        process::exit(1);
    }
}

fn bool_info(path: &Path, format: OutputFormat) -> Result<(), String> {
    use oramacore_fields::bool::{BoolStorage, DeletionThreshold};

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
            let json = serde_json::to_string_pretty(&info).map_err(|e| e.to_string())?;
            println!("{json}");
        }
    }

    Ok(())
}

fn bool_show(path: &Path, list_ids: bool) -> Result<(), String> {
    use oramacore_fields::bool::{BoolStorage, DeletionThreshold};

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

        println!();
        println!("Deleted document IDs: (count: {})", info.deleted_count);
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Number subcommands
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, ValueEnum)]
enum IndexType {
    U64,
    F64,
}

#[derive(Serialize)]
struct SearchResult {
    count: usize,
    doc_ids: Vec<u64>,
}

#[derive(Subcommand)]
enum NumberCommands {
    /// Check index integrity and validate files
    Check {
        /// Path to the index directory
        #[arg(short, long)]
        path: PathBuf,
        /// Index value type
        #[arg(short = 't', long, value_enum, default_value = "u64")]
        r#type: IndexType,
        /// Show detailed validation information
        #[arg(short, long)]
        verbose: bool,
    },
    /// Display index information and statistics
    Info {
        /// Path to the index directory
        #[arg(short, long)]
        path: PathBuf,
        /// Index value type
        #[arg(short = 't', long, value_enum, default_value = "u64")]
        r#type: IndexType,
        /// Output format
        #[arg(short, long, value_enum, default_value = "human")]
        format: OutputFormat,
    },
    /// Search the index with filter operations
    Search {
        /// Path to the index directory
        #[arg(short, long)]
        path: PathBuf,
        /// Index value type
        #[arg(short = 't', long, value_enum)]
        r#type: IndexType,
        /// Exact equality filter
        #[arg(long, group = "filter", allow_hyphen_values = true)]
        eq: Option<String>,
        /// Greater than filter
        #[arg(long, group = "filter", allow_hyphen_values = true)]
        gt: Option<String>,
        /// Greater than or equal filter
        #[arg(long, group = "filter", allow_hyphen_values = true)]
        gte: Option<String>,
        /// Less than filter
        #[arg(long, group = "filter", allow_hyphen_values = true)]
        lt: Option<String>,
        /// Less than or equal filter
        #[arg(long, group = "filter", allow_hyphen_values = true)]
        lte: Option<String>,
        /// Between filter (inclusive), format: MIN,MAX
        #[arg(long, group = "filter", allow_hyphen_values = true)]
        between: Option<String>,
        /// Maximum number of results to return
        #[arg(short, long)]
        limit: Option<usize>,
        /// Output format
        #[arg(short, long, value_enum, default_value = "human")]
        format: OutputFormat,
    },
}

fn number_check(path: &Path, index_type: IndexType, verbose: bool) -> Result<(), String> {
    use oramacore_fields::number::{CheckStatus, NumberStorage, Threshold};

    let result = match index_type {
        IndexType::U64 => {
            let index: NumberStorage<u64> =
                NumberStorage::new(path.to_path_buf(), Threshold::default())
                    .map_err(|e| format!("Failed to open index: {e}"))?;
            index.integrity_check()
        }
        IndexType::F64 => {
            let index: NumberStorage<f64> =
                NumberStorage::new(path.to_path_buf(), Threshold::default())
                    .map_err(|e| format!("Failed to open index: {e}"))?;
            index.integrity_check()
        }
    };

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
        process::exit(1);
    }
}

fn number_info(path: &Path, index_type: IndexType, format: OutputFormat) -> Result<(), String> {
    use oramacore_fields::number::{IndexInfo, NumberStorage, Threshold};

    let info: IndexInfo = match index_type {
        IndexType::U64 => {
            let index: NumberStorage<u64> =
                NumberStorage::new(path.to_path_buf(), Threshold::default())
                    .map_err(|e| format!("Failed to open index: {e}"))?;
            index.info()
        }
        IndexType::F64 => {
            let index: NumberStorage<f64> =
                NumberStorage::new(path.to_path_buf(), Threshold::default())
                    .map_err(|e| format!("Failed to open index: {e}"))?;
            index.info()
        }
    };

    match format {
        OutputFormat::Human => {
            println!("Index Information");
            println!("=================");
            println!("Format version:  {}", info.format_version);
            println!("Current offset:  {}", info.current_offset);
            println!("Version dir:     {}", info.version_dir.display());
            println!();
            println!("Header entries:  {}", info.header_entry_count);
            println!("Deleted entries: {}", info.deleted_count);
            println!("Data files:      {}", info.data_file_count);
            println!();
            println!("Header size:     {} bytes", info.header_size_bytes);
            println!("Deleted size:    {} bytes", info.deleted_size_bytes);
            println!("Data size:       {} bytes", info.data_total_bytes);
            println!(
                "Total size:      {} bytes",
                info.header_size_bytes + info.deleted_size_bytes + info.data_total_bytes
            );
            println!();
            println!("Pending inserts: {}", info.pending_inserts);
            println!("Pending deletes: {}", info.pending_deletes);
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&info).map_err(|e| e.to_string())?;
            println!("{json}");
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn number_search(
    path: &Path,
    index_type: IndexType,
    eq: Option<String>,
    gt: Option<String>,
    gte: Option<String>,
    lt: Option<String>,
    lte: Option<String>,
    between: Option<String>,
    limit: Option<usize>,
    format: OutputFormat,
) -> Result<(), String> {
    match index_type {
        IndexType::U64 => {
            number_search_typed::<u64>(path, eq, gt, gte, lt, lte, between, limit, format)
        }
        IndexType::F64 => {
            number_search_typed::<f64>(path, eq, gt, gte, lt, lte, between, limit, format)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn number_search_typed<T>(
    path: &Path,
    eq: Option<String>,
    gt: Option<String>,
    gte: Option<String>,
    lt: Option<String>,
    lte: Option<String>,
    between: Option<String>,
    limit: Option<usize>,
    format: OutputFormat,
) -> Result<(), String>
where
    T: oramacore_fields::number::IndexableNumber + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    use oramacore_fields::number::{FilterOp, NumberStorage, Threshold};

    let filter_op: FilterOp<T> = if let Some(v) = eq {
        FilterOp::Eq(parse_value(&v)?)
    } else if let Some(v) = gt {
        FilterOp::Gt(parse_value(&v)?)
    } else if let Some(v) = gte {
        FilterOp::Gte(parse_value(&v)?)
    } else if let Some(v) = lt {
        FilterOp::Lt(parse_value(&v)?)
    } else if let Some(v) = lte {
        FilterOp::Lte(parse_value(&v)?)
    } else if let Some(v) = between {
        let parts: Vec<&str> = v.split(',').collect();
        if parts.len() != 2 {
            return Err("between filter must be in format MIN,MAX".to_string());
        }
        let min: T = parse_value(parts[0])?;
        let max: T = parse_value(parts[1])?;
        FilterOp::BetweenInclusive(min, max)
    } else {
        return Err(
            "No filter specified. Use one of: --eq, --gt, --gte, --lt, --lte, --between"
                .to_string(),
        );
    };

    let index: NumberStorage<T> = NumberStorage::new(path.to_path_buf(), Threshold::default())
        .map_err(|e| format!("Failed to open index: {e}"))?;

    let filter_data = index.filter(filter_op);
    let mut doc_ids: Vec<u64> = filter_data.iter().collect();

    let total_count = doc_ids.len();
    if let Some(limit) = limit {
        doc_ids.truncate(limit);
    }

    let result = SearchResult {
        count: total_count,
        doc_ids,
    };

    match format {
        OutputFormat::Human => {
            println!("Found {} result(s)", result.count);
            if let Some(limit) = limit {
                if result.count > limit {
                    println!("(showing first {limit})");
                }
            }
            println!();
            for doc_id in &result.doc_ids {
                println!("{doc_id}");
            }
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?;
            println!("{json}");
        }
    }

    Ok(())
}

fn parse_value<T: std::str::FromStr>(s: &str) -> Result<T, String>
where
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    s.trim()
        .parse::<T>()
        .map_err(|e| format!("Failed to parse value '{s}': {e}"))
}

// ═══════════════════════════════════════════════════════════════════════════════
//  GeoPoint subcommands
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Subcommand)]
enum GeoCommands {
    /// Check data integrity of a geopoint index
    Check {
        /// Path to the index directory
        path: PathBuf,
        /// Show detailed output for each check
        #[arg(long)]
        verbose: bool,
    },
    /// Display index metadata and statistics
    Info {
        /// Path to the index directory
        path: PathBuf,
        /// Output format
        #[arg(long, default_value = "human")]
        format: OutputFormat,
    },
    /// Search the index with a geo filter
    Search {
        /// Path to the index directory
        path: PathBuf,
        /// Bounding box filter: MIN_LAT,MAX_LAT,MIN_LON,MAX_LON
        #[arg(long, conflicts_with = "radius")]
        bbox: Option<String>,
        /// Radius filter: LAT,LON,METERS
        #[arg(long, conflicts_with = "bbox")]
        radius: Option<String>,
        /// Maximum number of results
        #[arg(long)]
        limit: Option<usize>,
        /// Output format
        #[arg(long, default_value = "human")]
        format: OutputFormat,
    },
}

fn geo_check(path: PathBuf, verbose: bool) {
    use oramacore_fields::geopoint::{CheckStatus, GeoPointStorage, Threshold};

    let storage = match GeoPointStorage::new(path, Threshold::default(), 10) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[FAIL] Failed to open index: {e}");
            process::exit(1);
        }
    };

    let result = storage.integrity_check();

    for check in &result.checks {
        let tag = match check.status {
            CheckStatus::Ok => "[OK]",
            CheckStatus::Fail => "[FAIL]",
            CheckStatus::Skip => "[SKIP]",
        };
        if verbose {
            println!("{tag} {}: {}", check.name, check.message);
        } else {
            println!("{tag} {}", check.name);
        }
    }

    if result.passed {
        println!("\nAll checks passed.");
    } else {
        println!("\nSome checks failed.");
        process::exit(1);
    }
}

fn geo_info(path: PathBuf, format: OutputFormat) {
    use oramacore_fields::geopoint::{GeoPointStorage, Threshold};

    let storage = match GeoPointStorage::new(path, Threshold::default(), 10) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to open index: {e}");
            process::exit(1);
        }
    };

    let info = match storage.info() {
        Ok(i) => i,
        Err(e) => {
            eprintln!("Failed to get index info: {e}");
            process::exit(1);
        }
    };

    match format {
        OutputFormat::Human => {
            println!("Format version:     {}", info.format_version);
            println!("Version ID:         {}", info.current_version_id);
            println!("Version directory:   {}", info.version_dir.display());
            println!("Segments:           {}", info.segment_count);
            println!("Total points:       {}", info.total_points);
            println!("Deleted count:      {}", info.deleted_count);
            println!("deleted.bin size:   {} bytes", info.deleted_size_bytes);
            println!("Pending inserts:    {}", info.pending_inserts);
            println!("Pending deletes:    {}", info.pending_deletes);
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&info).unwrap();
            println!("{json}");
        }
    }
}

fn geo_search(
    path: PathBuf,
    bbox: Option<String>,
    radius: Option<String>,
    limit: Option<usize>,
    format: OutputFormat,
) {
    use oramacore_fields::geopoint::{GeoFilterOp, GeoPoint, GeoPointStorage, Threshold};

    let op = if let Some(bbox_str) = bbox {
        parse_bbox(&bbox_str)
    } else if let Some(radius_str) = radius {
        parse_radius(&radius_str)
    } else {
        eprintln!("Either --bbox or --radius must be specified");
        process::exit(1);
    };

    let storage = match GeoPointStorage::new(path, Threshold::default(), 10) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to open index: {e}");
            process::exit(1);
        }
    };

    let filter = storage.filter(op);
    let results: Vec<u64> = match limit {
        Some(n) => filter.iter().take(n).collect(),
        None => filter.iter().collect(),
    };

    match format {
        OutputFormat::Human => {
            println!("Found {} results:", results.len());
            for doc_id in &results {
                println!("  {doc_id}");
            }
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&results).unwrap();
            println!("{json}");
        }
    }

    fn parse_bbox(s: &str) -> GeoFilterOp {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 4 {
            eprintln!("Invalid --bbox format. Expected: MIN_LAT,MAX_LAT,MIN_LON,MAX_LON");
            process::exit(1);
        }
        let min_lat: f64 = parts[0].trim().parse().unwrap_or_else(|e| {
            eprintln!("Invalid MIN_LAT: {e}");
            process::exit(1);
        });
        let max_lat: f64 = parts[1].trim().parse().unwrap_or_else(|e| {
            eprintln!("Invalid MAX_LAT: {e}");
            process::exit(1);
        });
        let min_lon: f64 = parts[2].trim().parse().unwrap_or_else(|e| {
            eprintln!("Invalid MIN_LON: {e}");
            process::exit(1);
        });
        let max_lon: f64 = parts[3].trim().parse().unwrap_or_else(|e| {
            eprintln!("Invalid MAX_LON: {e}");
            process::exit(1);
        });
        GeoFilterOp::BoundingBox {
            min_lat,
            max_lat,
            min_lon,
            max_lon,
        }
    }

    fn parse_radius(s: &str) -> GeoFilterOp {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 3 {
            eprintln!("Invalid --radius format. Expected: LAT,LON,METERS");
            process::exit(1);
        }
        let lat: f64 = parts[0].trim().parse().unwrap_or_else(|e| {
            eprintln!("Invalid LAT: {e}");
            process::exit(1);
        });
        let lon: f64 = parts[1].trim().parse().unwrap_or_else(|e| {
            eprintln!("Invalid LON: {e}");
            process::exit(1);
        });
        let radius_meters: f64 = parts[2].trim().parse().unwrap_or_else(|e| {
            eprintln!("Invalid METERS: {e}");
            process::exit(1);
        });
        let center = GeoPoint::new(lat, lon).unwrap_or_else(|e| {
            eprintln!("Invalid center point: {e}");
            process::exit(1);
        });
        GeoFilterOp::Radius {
            center,
            radius_meters,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════════

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        TopLevel::Bool { command } => match command {
            BoolCommands::Check { path, verbose } => bool_check(&path, verbose),
            BoolCommands::Info { path, format } => bool_info(&path, format),
            BoolCommands::Show { path, list_ids } => bool_show(&path, list_ids),
        },
        TopLevel::Number { command } => match command {
            NumberCommands::Check {
                path,
                r#type,
                verbose,
            } => number_check(&path, r#type, verbose),
            NumberCommands::Info {
                path,
                r#type,
                format,
            } => number_info(&path, r#type, format),
            NumberCommands::Search {
                path,
                r#type,
                eq,
                gt,
                gte,
                lt,
                lte,
                between,
                limit,
                format,
            } => number_search(&path, r#type, eq, gt, gte, lt, lte, between, limit, format),
        },
        TopLevel::Geopoint { command } => {
            match command {
                GeoCommands::Check { path, verbose } => geo_check(path, verbose),
                GeoCommands::Info { path, format } => geo_info(path, format),
                GeoCommands::Search {
                    path,
                    bbox,
                    radius,
                    limit,
                    format,
                } => geo_search(path, bbox, radius, limit, format),
            }
            return;
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}

#[cfg(test)]
mod tests_cli {
    use super::*;
    use oramacore_fields::bool::{BoolStorage, DeletionThreshold, IndexedValue};
    use tempfile::TempDir;

    fn create_test_bool_index() -> (TempDir, PathBuf) {
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
    fn test_bool_cmd_info() {
        let (_tmp, path) = create_test_bool_index();
        bool_info(&path, OutputFormat::Human).unwrap();
    }

    #[test]
    fn test_bool_cmd_show() {
        let (_tmp, path) = create_test_bool_index();
        bool_show(&path, false).unwrap();
    }

    #[test]
    fn test_bool_cmd_show_with_list_ids() {
        let (_tmp, path) = create_test_bool_index();
        bool_show(&path, true).unwrap();
    }

    #[test]
    fn test_bool_cmd_check() {
        let (_tmp, path) = create_test_bool_index();
        bool_check(&path, false).unwrap();
    }

    #[test]
    fn test_bool_cmd_check_verbose() {
        let (_tmp, path) = create_test_bool_index();
        bool_check(&path, true).unwrap();
    }

    #[test]
    fn test_bool_cmd_check_uncompacted_index() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().to_path_buf();
        let index = BoolStorage::new(path.clone(), DeletionThreshold::default()).unwrap();
        let result = index.integrity_check();
        assert!(!result.passed);

        index.insert(&IndexedValue::Plain(true), 1);
        index.compact(1).unwrap();
        let result = index.integrity_check();
        assert!(result.passed);
    }

    #[test]
    fn test_bool_cmd_check_wrong_format_version() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().to_path_buf();

        let index = BoolStorage::new(path.clone(), DeletionThreshold::default()).unwrap();
        index.insert(&IndexedValue::Plain(true), 1);
        index.compact(1).unwrap();
        drop(index);

        let current_path = path.join("CURRENT");
        std::fs::write(&current_path, "999\n1").unwrap();

        let result = bool_check(&path, false);
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("Unsupported format version"));
    }
}
