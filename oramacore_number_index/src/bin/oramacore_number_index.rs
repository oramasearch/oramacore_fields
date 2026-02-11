//! CLI tool for inspecting, validating, and querying oramacore number indexes.

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;

use oramacore_number_index::{CheckStatus, FilterOp, IndexInfo, NumberStorage, Threshold};

/// CLI tool for inspecting and querying oramacore number indexes.
#[derive(Parser)]
#[command(name = "oramacore_number_index")]
#[command(about = "Inspect, validate, and query oramacore number indexes")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
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

        /// Index value type (required for search)
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

#[derive(Clone, Copy, ValueEnum)]
enum IndexType {
    U64,
    F64,
}

#[derive(Clone, Copy, ValueEnum)]
enum OutputFormat {
    Human,
    Json,
}

#[derive(Serialize)]
struct SearchResult {
    count: usize,
    doc_ids: Vec<u64>,
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Check {
            path,
            r#type,
            verbose,
        } => cmd_check(&path, r#type, verbose),
        Commands::Info {
            path,
            r#type,
            format,
        } => cmd_info(&path, r#type, format),
        Commands::Search {
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
        } => cmd_search(&path, r#type, eq, gt, gte, lt, lte, between, limit, format),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn cmd_check(path: &Path, index_type: IndexType, verbose: bool) -> Result<(), String> {
    // Open index with appropriate type to run integrity check
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

    // Display results
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

fn cmd_info(path: &Path, index_type: IndexType, format: OutputFormat) -> Result<(), String> {
    // Open index with appropriate type to get info
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
fn cmd_search(
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
        IndexType::U64 => search_typed::<u64>(path, eq, gt, gte, lt, lte, between, limit, format),
        IndexType::F64 => search_typed::<f64>(path, eq, gt, gte, lt, lte, between, limit, format),
    }
}

#[allow(clippy::too_many_arguments)]
fn search_typed<T>(
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
    T: oramacore_number_index::IndexableNumber + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    // Parse filter
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

    // Open index
    let index: NumberStorage<T> = NumberStorage::new(path.to_path_buf(), Threshold::default())
        .map_err(|e| format!("Failed to open index: {e}"))?;

    // Execute filter
    let filter_data = index.filter(filter_op);
    let mut doc_ids: Vec<u64> = filter_data.iter().collect();

    // Apply limit
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
