use clap::{Parser, Subcommand, ValueEnum};
use oramacore_geopoint_index::{CheckStatus, GeoFilterOp, GeoPoint, GeoPointStorage, Threshold};
use std::path::PathBuf;
use std::process;

#[derive(Parser)]
#[command(name = "geopoint-index-cli", about = "GeoPoint index inspection tool")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
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

#[derive(Clone, ValueEnum)]
enum OutputFormat {
    Human,
    Json,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Check { path, verbose } => cmd_check(path, verbose),
        Command::Info { path, format } => cmd_info(path, format),
        Command::Search {
            path,
            bbox,
            radius,
            limit,
            format,
        } => cmd_search(path, bbox, radius, limit, format),
    }
}

fn cmd_check(path: PathBuf, verbose: bool) {
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

fn cmd_info(path: PathBuf, format: OutputFormat) {
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

fn cmd_search(
    path: PathBuf,
    bbox: Option<String>,
    radius: Option<String>,
    limit: Option<usize>,
    format: OutputFormat,
) {
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
