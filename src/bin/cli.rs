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
    /// Inspect and search string (full-text) indexes
    String {
        #[command(subcommand)]
        command: StringCommands,
    },
    /// Inspect and search vector (ANN) indexes
    Vector {
        #[command(subcommand)]
        command: VectorCommands,
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
    /// Filter the index for documents matching a boolean value
    Filter {
        /// Path to the index directory
        path: PathBuf,
        /// The boolean value to search for (true or false)
        #[arg(long)]
        value: bool,
        /// Maximum number of results to return
        #[arg(short, long)]
        limit: Option<usize>,
        /// Output format
        #[arg(short, long, value_enum, default_value = "human")]
        format: OutputFormat,
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

fn bool_filter(
    path: &Path,
    value: bool,
    limit: Option<usize>,
    format: OutputFormat,
) -> Result<(), String> {
    use oramacore_fields::bool::{BoolStorage, DeletionThreshold};

    let index = BoolStorage::new(path.to_path_buf(), DeletionThreshold::default())
        .map_err(|e| format!("Failed to open index: {e}"))?;

    let filter_data = index.filter(value);
    let mut doc_ids: Vec<u64> = filter_data.iter().collect();

    let total_count = doc_ids.len();
    if let Some(limit) = limit {
        doc_ids.truncate(limit);
    }

    let result = FilterResult {
        count: total_count,
        doc_ids,
    };

    match format {
        OutputFormat::Human => {
            println!("Found {} result(s) where value = {value}", result.count);
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

// ═══════════════════════════════════════════════════════════════════════════════
//  Number subcommands
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, ValueEnum)]
enum IndexType {
    U64,
    F64,
}

#[derive(Serialize)]
struct FilterResult {
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
    /// Filter the index with filter operations
    Filter {
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
fn number_filter(
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
            number_filter_typed::<u64>(path, eq, gt, gte, lt, lte, between, limit, format)
        }
        IndexType::F64 => {
            number_filter_typed::<f64>(path, eq, gt, gte, lt, lte, between, limit, format)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn number_filter_typed<T>(
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

    let result = FilterResult {
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
    /// Filter the index with a geo filter
    Filter {
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

fn geo_filter(
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
//  String (full-text) subcommands
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Subcommand)]
enum StringCommands {
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
    /// Search the index for matching documents
    Search {
        /// Path to the index directory
        path: PathBuf,
        /// Search query (space-separated tokens)
        #[arg(short, long)]
        query: String,
        /// Tolerance: 0 = exact, omit for prefix, N = Levenshtein distance
        #[arg(short, long)]
        tolerance: Option<u8>,
        /// Use prefix matching (overrides tolerance)
        #[arg(long)]
        prefix: bool,
        /// Only match exact (unstemmed) positions
        #[arg(long)]
        exact_match: bool,
        /// Field boost multiplier
        #[arg(long)]
        boost: Option<f32>,
        /// Phrase boost multiplier for consecutive tokens
        #[arg(long)]
        phrase_boost: Option<f32>,
        /// Maximum number of results to return
        #[arg(short, long)]
        limit: Option<usize>,
        /// Output format
        #[arg(short, long, value_enum, default_value = "human")]
        format: OutputFormat,
    },
}

fn string_check(path: &Path, verbose: bool) -> Result<(), String> {
    use oramacore_fields::string::{CheckStatus, StringStorage, Threshold};

    let index = StringStorage::new(path.to_path_buf(), Threshold::default())
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

fn string_info(path: &Path, format: OutputFormat) -> Result<(), String> {
    use oramacore_fields::string::{StringStorage, Threshold};

    let index = StringStorage::new(path.to_path_buf(), Threshold::default())
        .map_err(|e| format!("Failed to open index: {e}"))?;

    let info = index.info();

    match format {
        OutputFormat::Human => {
            println!("Index Information");
            println!("=================");
            println!("Format version:    {}", info.format_version);
            println!("Current version:   {}", info.current_version_number);
            println!("Version dir:       {}", info.version_dir.display());
            println!();
            println!("Unique terms:      {}", info.unique_terms_count);
            println!("Total postings:    {}", info.total_postings_count);
            println!("Total documents:   {}", info.total_documents);
            println!("Avg field length:  {:.2}", info.avg_field_length);
            println!("Deleted entries:   {}", info.deleted_count);
            println!();
            println!("FST size:          {} bytes", info.fst_size_bytes);
            println!("Postings size:     {} bytes", info.postings_size_bytes);
            println!("Doc lengths size:  {} bytes", info.doc_lengths_size_bytes);
            println!("Deleted size:      {} bytes", info.deleted_size_bytes);
            println!("Global info size:  {} bytes", info.global_info_size_bytes);
            println!("Total size:        {} bytes", info.total_size_bytes());
            println!();
            println!("Pending ops:       {}", info.pending_ops);
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&info).map_err(|e| e.to_string())?;
            println!("{json}");
        }
    }

    Ok(())
}

#[derive(Serialize)]
struct SearchResultOutput {
    count: usize,
    docs: Vec<SearchResultDoc>,
}

#[derive(Serialize)]
struct SearchResultDoc {
    doc_id: u64,
    score: f32,
}

#[allow(clippy::too_many_arguments)]
fn string_search(
    path: &Path,
    query: &str,
    tolerance: Option<u8>,
    prefix: bool,
    exact_match: bool,
    boost: Option<f32>,
    phrase_boost: Option<f32>,
    limit: Option<usize>,
    format: OutputFormat,
) -> Result<(), String> {
    use oramacore_fields::string::{
        BM25Scorer, Bm25Params, NoFilter, SearchParams, StringStorage, Threshold,
    };

    let index = StringStorage::new(path.to_path_buf(), Threshold::default())
        .map_err(|e| format!("Failed to open index: {e}"))?;

    let tokens: Vec<String> = query.split_whitespace().map(|s| s.to_string()).collect();
    if tokens.is_empty() {
        return Err("Query must contain at least one token".to_string());
    }

    let tol = if prefix {
        None
    } else {
        Some(tolerance.unwrap_or(0))
    };

    let params = SearchParams {
        tokens: &tokens,
        exact_match,
        boost: boost.unwrap_or(1.0),
        bm25_params: Bm25Params::default(),
        tolerance: tol,
        phrase_boost,
    };

    let mut scorer = BM25Scorer::new();
    index
        .search::<NoFilter>(&params, None, &mut scorer)
        .map_err(|e| format!("Search failed: {e}"))?;

    let mut search_result = scorer.into_search_result();
    search_result.sort_by_score();

    let total_count = search_result.docs.len();
    let docs: Vec<SearchResultDoc> = search_result
        .docs
        .iter()
        .take(limit.unwrap_or(usize::MAX))
        .map(|d| SearchResultDoc {
            doc_id: d.doc_id,
            score: d.score,
        })
        .collect();

    let result = SearchResultOutput {
        count: total_count,
        docs,
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
            for doc in &result.docs {
                println!("{}\t{:.6}", doc.doc_id, doc.score);
            }
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?;
            println!("{json}");
        }
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Vector subcommands
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, ValueEnum)]
enum MetricArg {
    L2,
    DotProduct,
    Cosine,
}

impl MetricArg {
    fn into_metric(self) -> oramacore_fields::vector::DistanceMetric {
        use oramacore_fields::vector::DistanceMetric;
        match self {
            MetricArg::L2 => DistanceMetric::L2,
            MetricArg::DotProduct => DistanceMetric::DotProduct,
            MetricArg::Cosine => DistanceMetric::Cosine,
        }
    }
}

#[derive(Subcommand)]
enum VectorCommands {
    /// Check index integrity and validate files
    Check {
        /// Path to the index directory
        path: PathBuf,
        /// Vector dimensions
        #[arg(short, long)]
        dimensions: usize,
        /// Distance metric
        #[arg(short, long, value_enum)]
        metric: MetricArg,
        /// Show detailed validation information
        #[arg(short, long)]
        verbose: bool,
    },
    /// Display index information and statistics
    Info {
        /// Path to the index directory
        path: PathBuf,
        /// Vector dimensions
        #[arg(short, long)]
        dimensions: usize,
        /// Distance metric
        #[arg(short = 'M', long, value_enum)]
        metric: MetricArg,
        /// Output format
        #[arg(short, long, value_enum, default_value = "human")]
        format: OutputFormat,
    },
    /// Search for nearest neighbors
    Search {
        /// Path to the index directory
        path: PathBuf,
        /// Vector dimensions
        #[arg(short, long)]
        dimensions: usize,
        /// Distance metric
        #[arg(short = 'M', long, value_enum)]
        metric: MetricArg,
        /// Query vector as comma-separated floats (e.g. "0.1,0.2,0.3")
        #[arg(short, long, allow_hyphen_values = true)]
        query: String,
        /// Number of nearest neighbors to return
        #[arg(short, long)]
        k: usize,
        /// Search expansion factor (ef_search override)
        #[arg(long)]
        ef_search: Option<usize>,
        /// Output format
        #[arg(short, long, value_enum, default_value = "human")]
        format: OutputFormat,
    },
}

fn open_vector_storage(
    path: &Path,
    dimensions: usize,
    metric: MetricArg,
) -> Result<oramacore_fields::vector::VectorStorage, String> {
    use oramacore_fields::vector::{SegmentConfig, VectorConfig, VectorStorage};

    let config = VectorConfig::new(dimensions, metric.into_metric())
        .map_err(|e| format!("Invalid config: {e}"))?;
    VectorStorage::new(path.to_path_buf(), config, SegmentConfig::default())
        .map_err(|e| format!("Failed to open index: {e}"))
}

fn vector_check(
    path: &Path,
    dimensions: usize,
    metric: MetricArg,
    verbose: bool,
) -> Result<(), String> {
    use oramacore_fields::vector::CheckStatus;

    let storage = open_vector_storage(path, dimensions, metric)?;
    let result = storage.integrity_check();

    for check in &result.checks {
        let status_str = match check.status {
            CheckStatus::Ok => "[OK]",
            CheckStatus::Failed => "[FAIL]",
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

fn vector_info(
    path: &Path,
    dimensions: usize,
    metric: MetricArg,
    format: OutputFormat,
) -> Result<(), String> {
    let storage = open_vector_storage(path, dimensions, metric)?;
    let info = storage.info();

    match format {
        OutputFormat::Human => {
            println!("Index Information");
            println!("=================");
            println!("Format version:    {}", info.format_version);
            println!("Current version:   {}", info.current_version_number);
            println!("Version dir:       {}", info.version_dir.display());
            println!();
            println!("Dimensions:        {}", info.dimensions);
            println!("Metric:            {}", info.metric);
            println!("Num vectors:       {}", info.num_vectors);
            println!();
            println!("Pending ops:       {}", info.pending_ops);
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&info).map_err(|e| e.to_string())?;
            println!("{json}");
        }
    }

    Ok(())
}

#[derive(Serialize)]
struct VectorSearchResult {
    count: usize,
    results: Vec<VectorSearchHit>,
}

#[derive(Serialize)]
struct VectorSearchHit {
    doc_id: u64,
    distance: f32,
}

fn vector_search(
    path: &Path,
    dimensions: usize,
    metric: MetricArg,
    query_str: &str,
    k: usize,
    ef_search: Option<usize>,
    format: OutputFormat,
) -> Result<(), String> {
    let storage = open_vector_storage(path, dimensions, metric)?;

    let query: Vec<f32> = query_str
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<f32>()
                .map_err(|e| format!("Failed to parse query component '{s}': {e}"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let results = storage
        .search(&query, k, ef_search)
        .map_err(|e| format!("Search failed: {e}"))?;

    let result = VectorSearchResult {
        count: results.len(),
        results: results
            .iter()
            .map(|(doc_id, distance)| VectorSearchHit {
                doc_id: *doc_id,
                distance: *distance,
            })
            .collect(),
    };

    match format {
        OutputFormat::Human => {
            println!("Found {} result(s)", result.count);
            println!();
            for hit in &result.results {
                println!("{}\t{:.6}", hit.doc_id, hit.distance);
            }
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?;
            println!("{json}");
        }
    }

    Ok(())
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
            BoolCommands::Filter {
                path,
                value,
                limit,
                format,
            } => bool_filter(&path, value, limit, format),
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
            NumberCommands::Filter {
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
            } => number_filter(&path, r#type, eq, gt, gte, lt, lte, between, limit, format),
        },
        TopLevel::Geopoint { command } => {
            match command {
                GeoCommands::Check { path, verbose } => geo_check(path, verbose),
                GeoCommands::Info { path, format } => geo_info(path, format),
                GeoCommands::Filter {
                    path,
                    bbox,
                    radius,
                    limit,
                    format,
                } => geo_filter(path, bbox, radius, limit, format),
            }
            return;
        }
        TopLevel::String { command } => match command {
            StringCommands::Check { path, verbose } => string_check(&path, verbose),
            StringCommands::Info { path, format } => string_info(&path, format),
            StringCommands::Search {
                path,
                query,
                tolerance,
                prefix,
                exact_match,
                boost,
                phrase_boost,
                limit,
                format,
            } => string_search(
                &path,
                &query,
                tolerance,
                prefix,
                exact_match,
                boost,
                phrase_boost,
                limit,
                format,
            ),
        },
        TopLevel::Vector { command } => match command {
            VectorCommands::Check {
                path,
                dimensions,
                metric,
                verbose,
            } => vector_check(&path, dimensions, metric, verbose),
            VectorCommands::Info {
                path,
                dimensions,
                metric,
                format,
            } => vector_info(&path, dimensions, metric, format),
            VectorCommands::Search {
                path,
                dimensions,
                metric,
                query,
                k,
                ef_search,
                format,
            } => vector_search(&path, dimensions, metric, &query, k, ef_search, format),
        },
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
    fn test_bool_cmd_filter_true() {
        let (_tmp, path) = create_test_bool_index();
        bool_filter(&path, true, None, OutputFormat::Human).unwrap();
    }

    #[test]
    fn test_bool_cmd_filter_false() {
        let (_tmp, path) = create_test_bool_index();
        bool_filter(&path, false, None, OutputFormat::Human).unwrap();
    }

    #[test]
    fn test_bool_cmd_filter_with_limit() {
        let (_tmp, path) = create_test_bool_index();
        bool_filter(&path, true, Some(2), OutputFormat::Human).unwrap();
    }

    #[test]
    fn test_bool_cmd_filter_json() {
        let (_tmp, path) = create_test_bool_index();
        bool_filter(&path, true, None, OutputFormat::Json).unwrap();
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

    // ── String CLI tests ─────────────────────────────────────────────────────

    fn create_test_string_index() -> (TempDir, PathBuf) {
        use oramacore_fields::string::{
            IndexedValue as StringIndexedValue, StringStorage, TermData, Threshold,
        };
        use std::collections::HashMap;

        let tmp = TempDir::new().unwrap();
        let path = tmp.path().to_path_buf();

        let index = StringStorage::new(path.clone(), Threshold::default()).unwrap();

        let mut terms1 = HashMap::new();
        terms1.insert(
            "hello".to_string(),
            TermData {
                exact_positions: vec![0],
                stemmed_positions: vec![],
            },
        );
        terms1.insert(
            "world".to_string(),
            TermData {
                exact_positions: vec![1],
                stemmed_positions: vec![],
            },
        );
        index.insert(
            1,
            StringIndexedValue {
                field_length: 2,
                terms: terms1,
            },
        );

        let mut terms2 = HashMap::new();
        terms2.insert(
            "hello".to_string(),
            TermData {
                exact_positions: vec![0],
                stemmed_positions: vec![],
            },
        );
        terms2.insert(
            "rust".to_string(),
            TermData {
                exact_positions: vec![1],
                stemmed_positions: vec![],
            },
        );
        index.insert(
            2,
            StringIndexedValue {
                field_length: 2,
                terms: terms2,
            },
        );

        let mut terms3 = HashMap::new();
        terms3.insert(
            "goodbye".to_string(),
            TermData {
                exact_positions: vec![0],
                stemmed_positions: vec![],
            },
        );
        index.insert(
            3,
            StringIndexedValue {
                field_length: 1,
                terms: terms3,
            },
        );

        index.compact(1).unwrap();

        (tmp, path)
    }

    #[test]
    fn test_string_cmd_check() {
        let (_tmp, path) = create_test_string_index();
        string_check(&path, false).unwrap();
    }

    #[test]
    fn test_string_cmd_check_verbose() {
        let (_tmp, path) = create_test_string_index();
        string_check(&path, true).unwrap();
    }

    #[test]
    fn test_string_cmd_info() {
        let (_tmp, path) = create_test_string_index();
        string_info(&path, OutputFormat::Human).unwrap();
    }

    #[test]
    fn test_string_cmd_info_json() {
        let (_tmp, path) = create_test_string_index();
        string_info(&path, OutputFormat::Json).unwrap();
    }

    #[test]
    fn test_string_cmd_search() {
        let (_tmp, path) = create_test_string_index();
        string_search(
            &path,
            "hello",
            None,
            false,
            false,
            None,
            None,
            None,
            OutputFormat::Human,
        )
        .unwrap();
    }

    #[test]
    fn test_string_cmd_search_json() {
        let (_tmp, path) = create_test_string_index();
        string_search(
            &path,
            "hello",
            None,
            false,
            false,
            None,
            None,
            None,
            OutputFormat::Json,
        )
        .unwrap();
    }

    #[test]
    fn test_string_cmd_search_with_limit() {
        let (_tmp, path) = create_test_string_index();
        string_search(
            &path,
            "hello",
            None,
            false,
            false,
            None,
            None,
            Some(1),
            OutputFormat::Human,
        )
        .unwrap();
    }

    #[test]
    fn test_string_cmd_search_prefix() {
        let (_tmp, path) = create_test_string_index();
        string_search(
            &path,
            "hel",
            None,
            true,
            false,
            None,
            None,
            None,
            OutputFormat::Human,
        )
        .unwrap();
    }

    #[test]
    fn test_string_cmd_search_exact_match() {
        let (_tmp, path) = create_test_string_index();
        string_search(
            &path,
            "hello",
            None,
            false,
            true,
            None,
            None,
            None,
            OutputFormat::Human,
        )
        .unwrap();
    }

    #[test]
    fn test_string_cmd_search_with_boost() {
        let (_tmp, path) = create_test_string_index();
        string_search(
            &path,
            "hello",
            None,
            false,
            false,
            Some(2.0),
            None,
            None,
            OutputFormat::Human,
        )
        .unwrap();
    }

    #[test]
    fn test_string_cmd_search_with_phrase_boost() {
        let (_tmp, path) = create_test_string_index();
        string_search(
            &path,
            "hello world",
            None,
            false,
            false,
            None,
            Some(1.5),
            None,
            OutputFormat::Human,
        )
        .unwrap();
    }

    #[test]
    fn test_string_cmd_check_uncompacted_index() {
        use oramacore_fields::string::{StringStorage, Threshold};

        let tmp = TempDir::new().unwrap();
        let path = tmp.path().to_path_buf();
        let index = StringStorage::new(path.clone(), Threshold::default()).unwrap();
        let result = index.integrity_check();
        assert!(!result.passed);
    }

    #[test]
    fn test_string_cmd_check_wrong_format_version() {
        let (_tmp, path) = create_test_string_index();

        let current_path = path.join("CURRENT");
        std::fs::write(&current_path, "999\n1").unwrap();

        let result = string_check(&path, false);
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("Unsupported format version"));
    }

    // ── Vector CLI tests ────────────────────────────────────────────────────

    fn create_test_vector_index() -> (TempDir, PathBuf) {
        use oramacore_fields::vector::{
            DistanceMetric, SegmentConfig, VectorConfig, VectorStorage,
        };

        let tmp = TempDir::new().unwrap();
        let path = tmp.path().to_path_buf();

        let config = VectorConfig::new(3, DistanceMetric::Cosine).unwrap();
        let index = VectorStorage::new(path.clone(), config, SegmentConfig::default()).unwrap();

        index.insert(1, &[0.1, 0.2, 0.3]).unwrap();
        index.insert(2, &[0.4, 0.5, 0.6]).unwrap();
        index.insert(3, &[0.9, 0.8, 0.7]).unwrap();
        index.compact(1).unwrap();

        (tmp, path)
    }

    #[test]
    fn test_vector_cmd_check() {
        let (_tmp, path) = create_test_vector_index();
        vector_check(&path, 3, MetricArg::Cosine, false).unwrap();
    }

    #[test]
    fn test_vector_cmd_check_verbose() {
        let (_tmp, path) = create_test_vector_index();
        vector_check(&path, 3, MetricArg::Cosine, true).unwrap();
    }

    #[test]
    fn test_vector_cmd_info() {
        let (_tmp, path) = create_test_vector_index();
        vector_info(&path, 3, MetricArg::Cosine, OutputFormat::Human).unwrap();
    }

    #[test]
    fn test_vector_cmd_info_json() {
        let (_tmp, path) = create_test_vector_index();
        vector_info(&path, 3, MetricArg::Cosine, OutputFormat::Json).unwrap();
    }

    #[test]
    fn test_vector_cmd_search() {
        let (_tmp, path) = create_test_vector_index();
        vector_search(
            &path,
            3,
            MetricArg::Cosine,
            "0.1,0.2,0.3",
            2,
            None,
            OutputFormat::Human,
        )
        .unwrap();
    }

    #[test]
    fn test_vector_cmd_search_json() {
        let (_tmp, path) = create_test_vector_index();
        vector_search(
            &path,
            3,
            MetricArg::Cosine,
            "0.1,0.2,0.3",
            2,
            None,
            OutputFormat::Json,
        )
        .unwrap();
    }

    #[test]
    fn test_vector_cmd_search_with_ef() {
        let (_tmp, path) = create_test_vector_index();
        vector_search(
            &path,
            3,
            MetricArg::Cosine,
            "0.1,0.2,0.3",
            2,
            Some(128),
            OutputFormat::Human,
        )
        .unwrap();
    }

    #[test]
    fn test_vector_cmd_check_uncompacted_index() {
        use oramacore_fields::vector::{
            DistanceMetric, SegmentConfig, VectorConfig, VectorStorage,
        };

        let tmp = TempDir::new().unwrap();
        let path = tmp.path().to_path_buf();
        let config = VectorConfig::new(3, DistanceMetric::L2).unwrap();
        let index = VectorStorage::new(path.clone(), config, SegmentConfig::default()).unwrap();
        let result = index.integrity_check();
        assert!(!result.passed);
    }

    #[test]
    fn test_vector_cmd_check_wrong_format_version() {
        let (_tmp, path) = create_test_vector_index();

        let current_path = path.join("CURRENT");
        std::fs::write(&current_path, "999\n1").unwrap();

        let result = vector_check(&path, 3, MetricArg::Cosine, false);
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("format version"));
    }
}
