use oramacore_string_filter_index::{IndexedValue, StringFilterStorage, Threshold};

fn p(s: &str) -> IndexedValue {
    IndexedValue::Plain(s.to_string())
}

fn main() -> anyhow::Result<()> {
    let tmp = tempfile::TempDir::new()?;
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default())?;

    // Insert documents with string tags
    println!("Inserting documents...");
    index.insert(&p("electronics"), 1);
    index.insert(&p("electronics"), 2);
    index.insert(&p("electronics"), 5);
    index.insert(&p("clothing"), 3);
    index.insert(&p("clothing"), 4);
    index.insert(&p("books"), 5);
    index.insert(&p("books"), 6);

    // Filter by exact match
    let electronics: Vec<u64> = index.filter("electronics").iter().collect();
    println!("Electronics: {electronics:?}"); // [1, 2, 5]

    let clothing: Vec<u64> = index.filter("clothing").iter().collect();
    println!("Clothing: {clothing:?}"); // [3, 4]

    let books: Vec<u64> = index.filter("books").iter().collect();
    println!("Books: {books:?}"); // [5, 6]

    let missing: Vec<u64> = index.filter("toys").iter().collect();
    println!("Toys (missing): {missing:?}"); // []

    // Delete a document
    println!("\nDeleting doc 5...");
    index.delete(5);

    let electronics: Vec<u64> = index.filter("electronics").iter().collect();
    println!("Electronics after delete: {electronics:?}"); // [1, 2]

    let books: Vec<u64> = index.filter("books").iter().collect();
    println!("Books after delete: {books:?}"); // [6]

    // Compact to persist
    println!("\nCompacting...");
    index.compact(1)?;
    println!("Version: {}", index.current_version_number());

    // Data still accessible after compaction
    let electronics: Vec<u64> = index.filter("electronics").iter().collect();
    println!("Electronics after compact: {electronics:?}"); // [1, 2]

    println!("\nDone!");
    Ok(())
}
