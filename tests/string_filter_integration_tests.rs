use oramacore_fields::string_filter::{IndexedValue, StringFilterStorage, Threshold};
use tempfile::TempDir;

fn p(s: &str) -> IndexedValue {
    IndexedValue::Plain(s.to_string())
}

#[test]
fn test_basic_crud() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    // Insert
    index.insert(&p("apple"), 1);
    index.insert(&p("apple"), 2);
    index.insert(&p("banana"), 3);
    index.insert(&p("cherry"), 4);
    index.insert(&p("apple"), 5);

    // Filter
    let apple: Vec<u64> = index.filter("apple").iter().collect();
    assert_eq!(apple, vec![1, 2, 5]);

    let banana: Vec<u64> = index.filter("banana").iter().collect();
    assert_eq!(banana, vec![3]);

    let cherry: Vec<u64> = index.filter("cherry").iter().collect();
    assert_eq!(cherry, vec![4]);

    let missing: Vec<u64> = index.filter("missing").iter().collect();
    assert!(missing.is_empty());

    // Delete
    index.delete(2);

    let apple: Vec<u64> = index.filter("apple").iter().collect();
    assert_eq!(apple, vec![1, 5]);
}

#[test]
fn test_compaction_and_persistence() {
    let tmp = TempDir::new().unwrap();
    let base_path = tmp.path().to_path_buf();

    // Create, populate, compact
    {
        let index = StringFilterStorage::new(base_path.clone(), Threshold::default()).unwrap();
        index.insert(&p("red"), 1);
        index.insert(&p("red"), 2);
        index.insert(&p("blue"), 3);
        index.insert(&p("green"), 4);
        index.compact(1).unwrap();
    }

    // Reopen and verify
    {
        let index = StringFilterStorage::new(base_path.clone(), Threshold::default()).unwrap();
        assert_eq!(index.current_version_number(), 1);

        let red: Vec<u64> = index.filter("red").iter().collect();
        assert_eq!(red, vec![1, 2]);

        let blue: Vec<u64> = index.filter("blue").iter().collect();
        assert_eq!(blue, vec![3]);

        let green: Vec<u64> = index.filter("green").iter().collect();
        assert_eq!(green, vec![4]);
    }
}

#[test]
fn test_multiple_compaction_rounds() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    // Round 1
    index.insert(&p("cat"), 1);
    index.insert(&p("dog"), 2);
    index.compact(1).unwrap();

    // Round 2
    index.insert(&p("cat"), 3);
    index.insert(&p("bird"), 4);
    index.compact(2).unwrap();

    // Round 3
    index.insert(&p("dog"), 5);
    index.insert(&p("fish"), 6);
    index.compact(3).unwrap();

    let cat: Vec<u64> = index.filter("cat").iter().collect();
    assert_eq!(cat, vec![1, 3]);

    let dog: Vec<u64> = index.filter("dog").iter().collect();
    assert_eq!(dog, vec![2, 5]);

    let bird: Vec<u64> = index.filter("bird").iter().collect();
    assert_eq!(bird, vec![4]);

    let fish: Vec<u64> = index.filter("fish").iter().collect();
    assert_eq!(fish, vec![6]);
}

#[test]
fn test_delete_after_compaction() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&p("hello"), 1);
    index.insert(&p("hello"), 2);
    index.insert(&p("hello"), 3);
    index.compact(1).unwrap();

    // Delete after compaction
    index.delete(2);

    let results: Vec<u64> = index.filter("hello").iter().collect();
    assert_eq!(results, vec![1, 3]);

    // Compact again with the delete
    index.compact(2).unwrap();

    let results: Vec<u64> = index.filter("hello").iter().collect();
    assert_eq!(results, vec![1, 3]);
}

#[test]
fn test_edge_case_empty_string() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&p(""), 1);
    index.insert(&p(""), 2);
    index.insert(&p("notempty"), 3);

    let results: Vec<u64> = index.filter("").iter().collect();
    assert_eq!(results, vec![1, 2]);

    index.compact(1).unwrap();

    let results: Vec<u64> = index.filter("").iter().collect();
    assert_eq!(results, vec![1, 2]);
}

#[test]
fn test_edge_case_unicode() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&p("日本"), 1);
    index.insert(&p("中国"), 2);
    index.insert(&p("한국"), 3);
    index.insert(&p("🇺🇸"), 4);
    index.insert(&p("café"), 5);

    index.compact(1).unwrap();

    assert_eq!(index.filter("日本").iter().collect::<Vec<_>>(), vec![1]);
    assert_eq!(index.filter("中国").iter().collect::<Vec<_>>(), vec![2]);
    assert_eq!(index.filter("한국").iter().collect::<Vec<_>>(), vec![3]);
    assert_eq!(index.filter("🇺🇸").iter().collect::<Vec<_>>(), vec![4]);
    assert_eq!(index.filter("café").iter().collect::<Vec<_>>(), vec![5]);
}

#[test]
fn test_edge_case_doc_id_zero() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&p("key"), 0);
    index.insert(&p("key"), 1);

    let results: Vec<u64> = index.filter("key").iter().collect();
    assert_eq!(results, vec![0, 1]);

    index.compact(1).unwrap();

    let results: Vec<u64> = index.filter("key").iter().collect();
    assert_eq!(results, vec![0, 1]);
}

#[test]
fn test_edge_case_large_doc_ids() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&p("key"), u64::MAX - 1);
    index.insert(&p("key"), u64::MAX);

    let results: Vec<u64> = index.filter("key").iter().collect();
    assert_eq!(results, vec![u64::MAX - 1, u64::MAX]);

    index.compact(1).unwrap();

    let results: Vec<u64> = index.filter("key").iter().collect();
    assert_eq!(results, vec![u64::MAX - 1, u64::MAX]);
}

#[test]
fn test_many_keys() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    for i in 0u64..500 {
        index.insert(&p(&format!("key_{i:04}")), i);
    }

    index.compact(1).unwrap();

    for i in 0u64..500 {
        let results: Vec<u64> = index.filter(&format!("key_{i:04}")).iter().collect();
        assert_eq!(results, vec![i], "Failed for key_{i:04}");
    }
}

#[test]
fn test_deletion_threshold_apply() {
    let tmp = TempDir::new().unwrap();
    // Very low threshold to force deletion application
    let index =
        StringFilterStorage::new(tmp.path().to_path_buf(), 0.01f64.try_into().unwrap()).unwrap();

    index.insert(&p("a"), 1);
    index.insert(&p("a"), 2);
    index.insert(&p("a"), 3);
    index.delete(2);

    index.compact(1).unwrap();

    let results: Vec<u64> = index.filter("a").iter().collect();
    assert_eq!(results, vec![1, 3]);
}

#[test]
fn test_deletion_threshold_carry_forward() {
    let tmp = TempDir::new().unwrap();
    // Very high threshold so deletes are carried forward
    let index =
        StringFilterStorage::new(tmp.path().to_path_buf(), 0.99f64.try_into().unwrap()).unwrap();

    index.insert(&p("a"), 1);
    index.insert(&p("a"), 2);
    index.insert(&p("a"), 3);
    index.delete(2);

    index.compact(1).unwrap();

    // Should still filter correctly even with carried-forward deletes
    let results: Vec<u64> = index.filter("a").iter().collect();
    assert_eq!(results, vec![1, 3]);
}

#[test]
fn test_info() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&p("hello"), 1);
    index.insert(&p("world"), 2);
    index.compact(1).unwrap();

    let info = index.info();
    assert_eq!(info.current_version_number, 1);
    assert_eq!(info.unique_keys_count, 2);
    assert_eq!(info.total_postings_count, 2);
    assert_eq!(info.deleted_count, 0);
}

#[test]
fn test_integrity_check() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&p("hello"), 1);
    index.compact(1).unwrap();

    let result = index.integrity_check();
    assert!(result.passed);
}

#[test]
fn test_same_doc_id_different_keys() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    // Same doc_id under multiple keys (like tags)
    index.insert(&p("tag_a"), 1);
    index.insert(&p("tag_b"), 1);
    index.insert(&p("tag_c"), 1);
    index.insert(&p("tag_a"), 2);

    let a: Vec<u64> = index.filter("tag_a").iter().collect();
    assert_eq!(a, vec![1, 2]);

    let b: Vec<u64> = index.filter("tag_b").iter().collect();
    assert_eq!(b, vec![1]);

    index.compact(1).unwrap();

    let a: Vec<u64> = index.filter("tag_a").iter().collect();
    assert_eq!(a, vec![1, 2]);

    // Delete doc 1 - should remove from all keys
    index.delete(1);

    let a: Vec<u64> = index.filter("tag_a").iter().collect();
    assert_eq!(a, vec![2]);

    let b: Vec<u64> = index.filter("tag_b").iter().collect();
    assert!(b.is_empty());
}

#[test]
fn test_delete_all_docs_for_key() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    // Insert several doc_ids under one key, plus another key to verify isolation
    index.insert(&p("target"), 1);
    index.insert(&p("target"), 2);
    index.insert(&p("target"), 3);
    index.insert(&p("other"), 10);

    // Delete all docs for "target"
    index.delete(1);
    index.delete(2);
    index.delete(3);

    let results: Vec<u64> = index.filter("target").iter().collect();
    assert!(
        results.is_empty(),
        "Expected empty results after deleting all docs for key"
    );

    // "other" key should be unaffected
    let other: Vec<u64> = index.filter("other").iter().collect();
    assert_eq!(other, vec![10]);

    // Compact and verify still empty
    index.compact(1).unwrap();

    let results: Vec<u64> = index.filter("target").iter().collect();
    assert!(
        results.is_empty(),
        "Expected empty results after compaction"
    );

    let other: Vec<u64> = index.filter("other").iter().collect();
    assert_eq!(other, vec![10]);

    // Second compaction round to verify stability
    index.compact(2).unwrap();

    let results: Vec<u64> = index.filter("target").iter().collect();
    assert!(
        results.is_empty(),
        "Expected empty results after second compaction"
    );
}

// ============================================================================
// keys() Tests
// ============================================================================

#[test]
fn test_keys_live_only() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&p("cherry"), 1);
    index.insert(&p("apple"), 2);
    index.insert(&p("banana"), 3);

    let keys = index.keys();
    assert_eq!(keys, vec!["apple", "banana", "cherry"]);
}

#[test]
fn test_keys_compacted_only() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&p("red"), 1);
    index.insert(&p("green"), 2);
    index.insert(&p("blue"), 3);
    index.compact(1).unwrap();

    let keys = index.keys();
    assert_eq!(keys, vec!["blue", "green", "red"]);
}

#[test]
fn test_keys_merged_with_dedup() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&p("apple"), 1);
    index.insert(&p("cherry"), 2);
    index.compact(1).unwrap();

    // Add overlapping + new key in live layer
    index.insert(&p("cherry"), 3);
    index.insert(&p("banana"), 4);

    let keys = index.keys();
    assert_eq!(keys, vec!["apple", "banana", "cherry"]);
}

#[test]
fn test_keys_empty_index() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    let keys = index.keys();
    assert!(keys.is_empty());
}

#[test]
fn test_keys_phantom_after_deletion() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&p("visible"), 1);
    index.insert(&p("phantom"), 2);
    index.compact(1).unwrap();

    // Delete all docs for "phantom" — key stays in FST until next apply-deletes compaction
    index.delete(2);

    let keys = index.keys();
    // "phantom" is still reported (approximate semantics)
    assert!(keys.contains(&"phantom".to_string()));
    assert!(keys.contains(&"visible".to_string()));

    // But filtering returns no docs for the phantom key
    let phantom_docs: Vec<u64> = index.filter("phantom").iter().collect();
    assert!(phantom_docs.is_empty());
}

// ============================================================================
// Owned Iterator Tests (IntoIterator for FilterData)
// ============================================================================

#[test]
fn test_into_iter_live_only() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&p("hello"), 1);
    index.insert(&p("hello"), 5);
    index.insert(&p("hello"), 10);
    index.insert(&p("world"), 2);

    let results: Vec<u64> = index.filter("hello").into_iter().collect();
    assert_eq!(results, vec![1, 5, 10]);
}

#[test]
fn test_into_iter_compacted_plus_live() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&p("color"), 1);
    index.insert(&p("color"), 3);
    index.compact(1).unwrap();

    index.insert(&p("color"), 5);
    index.insert(&p("color"), 7);

    let results: Vec<u64> = index.filter("color").into_iter().collect();
    assert_eq!(results, vec![1, 3, 5, 7]);
}

#[test]
fn test_into_iter_with_deletes() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&p("key"), 1);
    index.insert(&p("key"), 2);
    index.insert(&p("key"), 3);
    index.compact(1).unwrap();

    index.delete(2);

    let results: Vec<u64> = index.filter("key").into_iter().collect();
    assert_eq!(results, vec![1, 3]);
}

#[test]
fn test_into_iter_empty() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    let results: Vec<u64> = index.filter("nonexistent").into_iter().collect();
    assert!(results.is_empty());
}

#[test]
fn test_into_iter_returned_from_function() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&p("fruit"), 1);
    index.insert(&p("fruit"), 2);
    index.insert(&p("fruit"), 3);

    fn get_docs(index: &StringFilterStorage, key: &str) -> impl Iterator<Item = u64> {
        index.filter(key).into_iter()
    }

    let results: Vec<u64> = get_docs(&index, "fruit").collect();
    assert_eq!(results, vec![1, 2, 3]);
}

#[test]
fn test_into_iter_matches_borrowed_iter() {
    let tmp = TempDir::new().unwrap();
    let index = StringFilterStorage::new(tmp.path().to_path_buf(), Threshold::default()).unwrap();

    index.insert(&p("a"), 1);
    index.insert(&p("a"), 3);
    index.insert(&p("a"), 5);
    index.compact(1).unwrap();

    index.insert(&p("a"), 7);
    index.delete(3);

    let borrowed: Vec<u64> = index.filter("a").iter().collect();
    let owned: Vec<u64> = index.filter("a").into_iter().collect();
    assert_eq!(borrowed, owned);
}
