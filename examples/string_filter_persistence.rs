use oramacore_fields::string_filter::{IndexedValue, StringFilterStorage, Threshold};

fn p(s: &str) -> IndexedValue {
    IndexedValue::Plain(s.to_string())
}

fn main() -> anyhow::Result<()> {
    let tmp = tempfile::TempDir::new()?;
    let base_path = tmp.path().to_path_buf();

    // Phase 1: Create and populate
    println!("=== Phase 1: Create and populate ===");
    {
        let index = StringFilterStorage::new(base_path.clone(), Threshold::default())?;
        index.insert(&p("color:red"), 1);
        index.insert(&p("color:red"), 2);
        index.insert(&p("color:blue"), 3);
        index.insert(&p("color:blue"), 4);
        index.insert(&p("size:large"), 1);
        index.insert(&p("size:large"), 3);
        index.insert(&p("size:small"), 2);
        index.insert(&p("size:small"), 4);

        index.compact(1)?;
        println!("Compacted to version {}", index.current_version_number());

        let red: Vec<u64> = index.filter("color:red").iter().collect();
        println!("Red items: {red:?}");
    }

    // Phase 2: Reopen and verify
    println!("\n=== Phase 2: Reopen and verify ===");
    {
        let index = StringFilterStorage::new(base_path.clone(), Threshold::default())?;
        println!("Loaded version {}", index.current_version_number());

        let red: Vec<u64> = index.filter("color:red").iter().collect();
        let blue: Vec<u64> = index.filter("color:blue").iter().collect();
        let large: Vec<u64> = index.filter("size:large").iter().collect();
        let small: Vec<u64> = index.filter("size:small").iter().collect();

        println!("Red: {red:?}");
        println!("Blue: {blue:?}");
        println!("Large: {large:?}");
        println!("Small: {small:?}");

        assert_eq!(red, vec![1, 2]);
        assert_eq!(blue, vec![3, 4]);
        assert_eq!(large, vec![1, 3]);
        assert_eq!(small, vec![2, 4]);

        // Add more data and compact again
        index.insert(&p("color:green"), 5);
        index.insert(&p("size:medium"), 5);
        index.compact(2)?;
        println!("\nCompacted to version {}", index.current_version_number());
    }

    // Phase 3: Verify second round persisted
    println!("\n=== Phase 3: Verify second round ===");
    {
        let index = StringFilterStorage::new(base_path.clone(), Threshold::default())?;
        println!("Loaded version {}", index.current_version_number());

        let green: Vec<u64> = index.filter("color:green").iter().collect();
        let medium: Vec<u64> = index.filter("size:medium").iter().collect();
        println!("Green: {green:?}");
        println!("Medium: {medium:?}");

        assert_eq!(green, vec![5]);
        assert_eq!(medium, vec![5]);

        // Cleanup old versions
        index.cleanup();
        println!("Cleaned up old versions");

        let info = index.info();
        println!("Unique keys: {}", info.unique_keys_count);
        println!("Total postings: {}", info.total_postings_count);
    }

    println!("\nAll verifications passed!");
    Ok(())
}
