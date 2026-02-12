use super::compacted::CompactedVersion;
use super::live::LiveSnapshot;
use super::merge::{sorted_merge, sorted_subtract, SortedMerge, SortedSubtract};
use std::iter::Copied;
use std::slice::Iter;
use std::sync::Arc;

type SliceIter<'a> = Copied<Iter<'a, u64>>;
type MergeIter<'a> = SortedMerge<SliceIter<'a>, SliceIter<'a>>;
type SubtractIter<'a> = SortedSubtract<MergeIter<'a>, MergeIter<'a>>;

/// Iterator that yields doc_ids matching an exact string filter.
/// Merges compacted postings with live inserts, then subtracts deletes.
pub struct FilterIterator<'a> {
    iter: SubtractIter<'a>,
}

impl<'a> FilterIterator<'a> {
    pub fn new(
        compacted_postings: &'a [u64],
        live_inserts: &'a [u64],
        compacted_deletes: &'a [u64],
        live_deletes: &'a [u64],
    ) -> Self {
        let merged_postings = sorted_merge(
            compacted_postings.iter().copied(),
            live_inserts.iter().copied(),
        );
        let merged_deletes = sorted_merge(
            compacted_deletes.iter().copied(),
            live_deletes.iter().copied(),
        );
        let iter = sorted_subtract(merged_postings, merged_deletes);

        Self { iter }
    }
}

impl Iterator for FilterIterator<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

/// Zero-allocation filter data that produces iterators.
///
/// Holds Arc references to the compacted version and live snapshot,
/// plus a borrowed key. No heap allocations are performed.
pub struct FilterData<'k> {
    version: Arc<CompactedVersion>,
    snapshot: Arc<LiveSnapshot>,
    key: &'k str,
}

impl<'k> FilterData<'k> {
    pub(crate) fn new(
        version: Arc<CompactedVersion>,
        snapshot: Arc<LiveSnapshot>,
        key: &'k str,
    ) -> Self {
        Self {
            version,
            snapshot,
            key,
        }
    }

    /// Create an iterator over matching doc_ids (ascending order).
    pub fn iter(&self) -> FilterIterator<'_> {
        let compacted_postings = self.version.lookup(self.key).unwrap_or(&[]);
        let live_doc_ids = self.snapshot.doc_ids_for_key(self.key);

        FilterIterator::new(
            compacted_postings,
            live_doc_ids,
            self.version.deletes_slice(),
            &self.snapshot.deletes_sorted,
        )
    }
}

impl<'a, 'k> IntoIterator for &'a FilterData<'k> {
    type Item = u64;
    type IntoIter = FilterIterator<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_empty() {
        let result: Vec<u64> = FilterIterator::new(&[], &[], &[], &[]).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_filter_live_only() {
        let live = [1u64, 5, 10];
        let result: Vec<u64> = FilterIterator::new(&[], &live, &[], &[]).collect();
        assert_eq!(result, vec![1, 5, 10]);
    }

    #[test]
    fn test_filter_compacted_only() {
        let compacted = [2u64, 4, 6];
        let result: Vec<u64> = FilterIterator::new(&compacted, &[], &[], &[]).collect();
        assert_eq!(result, vec![2, 4, 6]);
    }

    #[test]
    fn test_filter_merged() {
        let compacted = [1u64, 5, 10];
        let live = [3u64, 7, 12];
        let result: Vec<u64> = FilterIterator::new(&compacted, &live, &[], &[]).collect();
        assert_eq!(result, vec![1, 3, 5, 7, 10, 12]);
    }

    #[test]
    fn test_filter_with_deletes() {
        let compacted = [1u64, 5, 10];
        let live = [3u64, 7];
        let deletes = [5u64, 7];
        let result: Vec<u64> = FilterIterator::new(&compacted, &live, &deletes, &[]).collect();
        assert_eq!(result, vec![1, 3, 10]);
    }

    #[test]
    fn test_filter_with_live_deletes() {
        let compacted = [1u64, 5, 10];
        let live = [3u64];
        let live_deletes = [5u64];
        let result: Vec<u64> = FilterIterator::new(&compacted, &live, &[], &live_deletes).collect();
        assert_eq!(result, vec![1, 3, 10]);
    }

    #[test]
    fn test_filter_data_into_iter() {
        use super::super::live::LiveLayer;

        let version = Arc::new(CompactedVersion::empty());

        let mut layer = LiveLayer::new();
        layer.insert("hello", 1);
        layer.insert("hello", 5);
        layer.insert("world", 2);
        layer.refresh_snapshot();
        let snapshot = layer.get_snapshot();

        let filter_data = FilterData::new(version, snapshot, "hello");

        let mut results = Vec::new();
        for doc_id in &filter_data {
            results.push(doc_id);
        }

        assert_eq!(results, vec![1, 5]);
    }
}
