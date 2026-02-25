//! Iterators for streaming boolean filter results.

use super::live::LiveSnapshot;
use super::merge::{
    sorted_merge, sorted_merge_desc, sorted_subtract, sorted_subtract_desc, SortedMerge,
    SortedMergeDesc, SortedSubtract, SortedSubtractDesc,
};
use super::version::CompactedVersion;
use std::iter::{Copied, Rev};
use std::slice::Iter;
use std::sync::Arc;

type SliceIter<'a> = Copied<Iter<'a, u64>>;
type MergeIter<'a> = SortedMerge<SliceIter<'a>, SliceIter<'a>>;
type SubtractIter<'a> = SortedSubtract<MergeIter<'a>, MergeIter<'a>>;

type RevSliceIter<'a> = Copied<Rev<Iter<'a, u64>>>;
type MergeIterDesc<'a> = SortedMergeDesc<RevSliceIter<'a>, RevSliceIter<'a>>;
type SubtractIterDesc<'a> = SortedSubtractDesc<MergeIterDesc<'a>, MergeIterDesc<'a>>;

/// Sort order for iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortOrder {
    #[default]
    Ascending,
    Descending,
}

/// Iterator that yields doc_ids matching a boolean filter in ascending order.
pub struct FilterIterator<'a> {
    iter: SubtractIter<'a>,
}

impl<'a> FilterIterator<'a> {
    /// Create a new filter iterator from 4 sorted slices.
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

/// Iterator that yields doc_ids matching a boolean filter in descending order.
pub struct DescendingIterator<'a> {
    iter: SubtractIterDesc<'a>,
}

impl<'a> DescendingIterator<'a> {
    /// Create a new descending filter iterator from 4 sorted slices.
    pub fn new(
        compacted_postings: &'a [u64],
        live_inserts: &'a [u64],
        compacted_deletes: &'a [u64],
        live_deletes: &'a [u64],
    ) -> Self {
        let merged_postings = sorted_merge_desc(
            compacted_postings.iter().rev().copied(),
            live_inserts.iter().rev().copied(),
        );
        let merged_deletes = sorted_merge_desc(
            compacted_deletes.iter().rev().copied(),
            live_deletes.iter().rev().copied(),
        );
        let iter = sorted_subtract_desc(merged_postings, merged_deletes);

        Self { iter }
    }
}

impl Iterator for DescendingIterator<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

/// Iterator that yields doc_ids in either ascending or descending order.
pub enum SortedIterator<'a> {
    Ascending(FilterIterator<'a>),
    Descending(DescendingIterator<'a>),
}

impl Iterator for SortedIterator<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SortedIterator::Ascending(iter) => iter.next(),
            SortedIterator::Descending(iter) => iter.next(),
        }
    }
}

/// Holds a snapshot of the index state and produces iterators over matching doc_ids.
pub struct FilterData {
    version: Arc<CompactedVersion>,
    snapshot: Arc<LiveSnapshot>,
    value: bool,
}

impl FilterData {
    pub(crate) fn new(
        version: Arc<CompactedVersion>,
        snapshot: Arc<LiveSnapshot>,
        value: bool,
    ) -> Self {
        Self {
            version,
            snapshot,
            value,
        }
    }

    /// Create an iterator over matching doc_ids in ascending order.
    pub fn iter(&self) -> FilterIterator<'_> {
        FilterIterator::new(
            self.version.postings_slice(self.value),
            self.snapshot.inserts(self.value),
            self.version.deletes_slice(),
            &self.snapshot.deletes,
        )
    }

    /// Create an iterator that yields doc_ids in the specified order.
    pub fn sorted(&self, order: SortOrder) -> SortedIterator<'_> {
        match order {
            SortOrder::Ascending => SortedIterator::Ascending(self.iter()),
            SortOrder::Descending => SortedIterator::Descending(DescendingIterator::new(
                self.version.postings_slice(self.value),
                self.snapshot.inserts(self.value),
                self.version.deletes_slice(),
                &self.snapshot.deletes,
            )),
        }
    }
}

impl<'a> IntoIterator for &'a FilterData {
    type Item = u64;
    type IntoIter = FilterIterator<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl IntoIterator for FilterData {
    type Item = u64;
    type IntoIter = OwnedFilterIterator;

    fn into_iter(self) -> OwnedFilterIterator {
        owned_ascending(self.version, self.snapshot, self.value)
    }
}

impl FilterData {
    /// Consume this `FilterData` and return an owned iterator in the specified order.
    pub fn into_sorted(self, order: SortOrder) -> OwnedSortedIterator {
        match order {
            SortOrder::Ascending => {
                OwnedSortedIterator::Ascending(owned_ascending(self.version, self.snapshot, self.value))
            }
            SortOrder::Descending => {
                OwnedSortedIterator::Descending(owned_descending(self.version, self.snapshot, self.value))
            }
        }
    }
}

/// Extract raw pointers to the four slices from the Arc-owned data.
///
/// # Safety
///
/// The returned pointers are only valid as long as the `Arc`s they came from
/// remain alive. Callers must store those `Arc`s alongside any references
/// derived from these pointers.
fn extract_slice_ptrs(
    version: &CompactedVersion,
    snapshot: &LiveSnapshot,
    value: bool,
) -> (*const [u64], *const [u64], *const [u64], *const [u64]) {
    let cp: *const [u64] = version.postings_slice(value);
    let li: *const [u64] = snapshot.inserts(value);
    let cd: *const [u64] = version.deletes_slice();
    let ld: *const [u64] = snapshot.deletes.as_slice();
    (cp, li, cd, ld)
}

fn owned_ascending(
    version: Arc<CompactedVersion>,
    snapshot: Arc<LiveSnapshot>,
    value: bool,
) -> OwnedFilterIterator {
    let (cp, li, cd, ld) = extract_slice_ptrs(&version, &snapshot, value);

    // SAFETY: The pointers reference heap data inside `version` (Mmap) and
    // `snapshot` (Vec). Both `Arc`s are moved into `OwnedFilterIterator`,
    // keeping refcounts > 0 for the iterator's entire lifetime. Rust drops
    // fields in declaration order, so `iter` drops before `_version`/`_snapshot`.
    let iter = unsafe {
        FilterIterator::new(&*cp, &*li, &*cd, &*ld)
    };
    let iter = unsafe {
        std::mem::transmute::<FilterIterator<'_>, FilterIterator<'static>>(iter)
    };

    OwnedFilterIterator { iter, _version: version, _snapshot: snapshot }
}

fn owned_descending(
    version: Arc<CompactedVersion>,
    snapshot: Arc<LiveSnapshot>,
    value: bool,
) -> OwnedDescendingIterator {
    let (cp, li, cd, ld) = extract_slice_ptrs(&version, &snapshot, value);

    // SAFETY: Same reasoning as `owned_ascending`.
    let iter = unsafe {
        DescendingIterator::new(&*cp, &*li, &*cd, &*ld)
    };
    let iter = unsafe {
        std::mem::transmute::<DescendingIterator<'_>, DescendingIterator<'static>>(iter)
    };

    OwnedDescendingIterator { iter, _version: version, _snapshot: snapshot }
}

/// Owned ascending iterator that keeps the underlying data alive via `Arc`s.
///
/// Created by calling [`FilterData::into_iter()`] or
/// [`FilterData::into_sorted(SortOrder::Ascending)`].
pub struct OwnedFilterIterator {
    // IMPORTANT: `iter` must be declared before the Arc fields so it drops first.
    iter: FilterIterator<'static>,
    _version: Arc<CompactedVersion>,
    _snapshot: Arc<LiveSnapshot>,
}

impl Iterator for OwnedFilterIterator {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
        self.iter.next()
    }
}

/// Owned descending iterator that keeps the underlying data alive via `Arc`s.
///
/// Created by calling [`FilterData::into_sorted(SortOrder::Descending)`].
pub struct OwnedDescendingIterator {
    // IMPORTANT: `iter` must be declared before the Arc fields so it drops first.
    iter: DescendingIterator<'static>,
    _version: Arc<CompactedVersion>,
    _snapshot: Arc<LiveSnapshot>,
}

impl Iterator for OwnedDescendingIterator {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
        self.iter.next()
    }
}

/// Owned iterator that yields doc_ids in either ascending or descending order.
///
/// Created by calling [`FilterData::into_sorted()`].
pub enum OwnedSortedIterator {
    Ascending(OwnedFilterIterator),
    Descending(OwnedDescendingIterator),
}

impl Iterator for OwnedSortedIterator {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
        match self {
            OwnedSortedIterator::Ascending(iter) => iter.next(),
            OwnedSortedIterator::Descending(iter) => iter.next(),
        }
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
        let live_true = [1u64, 5, 10];
        let live_false = [2u64];

        let true_result: Vec<u64> = FilterIterator::new(&[], &live_true, &[], &[]).collect();
        let false_result: Vec<u64> = FilterIterator::new(&[], &live_false, &[], &[]).collect();

        assert_eq!(true_result, vec![1, 5, 10]);
        assert_eq!(false_result, vec![2]);
    }

    #[test]
    fn test_filter_with_deletes() {
        let live_true = [1u64, 5, 10];
        let live_false = [5u64];
        let deletes = [5u64];

        let true_result: Vec<u64> = FilterIterator::new(&[], &live_true, &[], &deletes).collect();
        let false_result: Vec<u64> = FilterIterator::new(&[], &live_false, &[], &deletes).collect();

        assert_eq!(true_result, vec![1, 10]);
        assert!(false_result.is_empty());
    }

    #[test]
    fn test_filter_data_into_iter() {
        // Test that &FilterData implements IntoIterator via for loop
        let version = Arc::new(CompactedVersion::empty());
        let snapshot = Arc::new(LiveSnapshot {
            true_inserts: vec![1, 5, 10],
            false_inserts: vec![2, 6],
            deletes: vec![],
            ops_len: 0,
        });

        let filter_data = FilterData::new(version, snapshot, true);

        // Use for loop to test IntoIterator implementation
        let mut results = Vec::new();
        for doc_id in &filter_data {
            results.push(doc_id);
        }

        assert_eq!(results, vec![1, 5, 10]);
    }

    #[test]
    fn test_sorted_ascending() {
        let version = Arc::new(CompactedVersion::empty());
        let snapshot = Arc::new(LiveSnapshot {
            true_inserts: vec![1, 5, 10],
            false_inserts: vec![2, 6],
            deletes: vec![],
            ops_len: 0,
        });

        let filter_data = FilterData::new(version, snapshot, true);
        let results: Vec<u64> = filter_data.sorted(SortOrder::Ascending).collect();

        assert_eq!(results, vec![1, 5, 10]);
    }

    #[test]
    fn test_sorted_descending() {
        let version = Arc::new(CompactedVersion::empty());
        let snapshot = Arc::new(LiveSnapshot {
            true_inserts: vec![1, 5, 10],
            false_inserts: vec![2, 6],
            deletes: vec![],
            ops_len: 0,
        });

        let filter_data = FilterData::new(version, snapshot, true);
        let results: Vec<u64> = filter_data.sorted(SortOrder::Descending).collect();

        assert_eq!(results, vec![10, 5, 1]);
    }

    #[test]
    fn test_sorted_empty() {
        let version = Arc::new(CompactedVersion::empty());
        let snapshot = Arc::new(LiveSnapshot {
            true_inserts: vec![],
            false_inserts: vec![],
            deletes: vec![],
            ops_len: 0,
        });

        let filter_data = FilterData::new(version, snapshot, true);

        let asc: Vec<u64> = filter_data.sorted(SortOrder::Ascending).collect();
        let desc: Vec<u64> = filter_data.sorted(SortOrder::Descending).collect();

        assert!(asc.is_empty());
        assert!(desc.is_empty());
    }

    #[test]
    fn test_sorted_with_deletes() {
        let version = Arc::new(CompactedVersion::empty());
        let snapshot = Arc::new(LiveSnapshot {
            true_inserts: vec![1, 5, 10, 15],
            false_inserts: vec![],
            deletes: vec![5, 15],
            ops_len: 0,
        });

        let filter_data = FilterData::new(version, snapshot, true);

        let asc: Vec<u64> = filter_data.sorted(SortOrder::Ascending).collect();
        let desc: Vec<u64> = filter_data.sorted(SortOrder::Descending).collect();

        assert_eq!(asc, vec![1, 10]);
        assert_eq!(desc, vec![10, 1]);
    }

    #[test]
    fn test_sorted_default_order() {
        // SortOrder::Ascending is the default
        assert_eq!(SortOrder::default(), SortOrder::Ascending);
    }

    #[test]
    fn test_owned_into_iter_ascending() {
        let version = Arc::new(CompactedVersion::empty());
        let snapshot = Arc::new(LiveSnapshot {
            true_inserts: vec![1, 5, 10],
            false_inserts: vec![2, 6],
            deletes: vec![],
            ops_len: 0,
        });

        let filter_data = FilterData::new(version, snapshot, true);
        let results: Vec<u64> = filter_data.into_iter().collect();

        assert_eq!(results, vec![1, 5, 10]);
    }

    #[test]
    fn test_owned_into_iter_for_loop() {
        let version = Arc::new(CompactedVersion::empty());
        let snapshot = Arc::new(LiveSnapshot {
            true_inserts: vec![3, 7],
            false_inserts: vec![],
            deletes: vec![],
            ops_len: 0,
        });

        let filter_data = FilterData::new(version, snapshot, true);

        // FilterData consumed by for loop (IntoIterator)
        let mut results = Vec::new();
        for doc_id in filter_data {
            results.push(doc_id);
        }

        assert_eq!(results, vec![3, 7]);
    }

    #[test]
    fn test_owned_into_sorted_ascending() {
        let version = Arc::new(CompactedVersion::empty());
        let snapshot = Arc::new(LiveSnapshot {
            true_inserts: vec![1, 5, 10],
            false_inserts: vec![],
            deletes: vec![],
            ops_len: 0,
        });

        let filter_data = FilterData::new(version, snapshot, true);
        let results: Vec<u64> = filter_data.into_sorted(SortOrder::Ascending).collect();

        assert_eq!(results, vec![1, 5, 10]);
    }

    #[test]
    fn test_owned_into_sorted_descending() {
        let version = Arc::new(CompactedVersion::empty());
        let snapshot = Arc::new(LiveSnapshot {
            true_inserts: vec![1, 5, 10],
            false_inserts: vec![],
            deletes: vec![],
            ops_len: 0,
        });

        let filter_data = FilterData::new(version, snapshot, true);
        let results: Vec<u64> = filter_data.into_sorted(SortOrder::Descending).collect();

        assert_eq!(results, vec![10, 5, 1]);
    }

    #[test]
    fn test_owned_into_iter_with_deletes() {
        let version = Arc::new(CompactedVersion::empty());
        let snapshot = Arc::new(LiveSnapshot {
            true_inserts: vec![1, 5, 10, 15],
            false_inserts: vec![],
            deletes: vec![5, 15],
            ops_len: 0,
        });

        let filter_data = FilterData::new(version, snapshot, true);
        let results: Vec<u64> = filter_data.into_iter().collect();

        assert_eq!(results, vec![1, 10]);
    }

    #[test]
    fn test_owned_into_sorted_with_deletes() {
        let version = Arc::new(CompactedVersion::empty());
        let snapshot = Arc::new(LiveSnapshot {
            true_inserts: vec![1, 5, 10, 15],
            false_inserts: vec![],
            deletes: vec![5, 15],
            ops_len: 0,
        });

        let filter_data = FilterData::new(version, snapshot, true);
        let results: Vec<u64> = filter_data.into_sorted(SortOrder::Descending).collect();

        assert_eq!(results, vec![10, 1]);
    }

    #[test]
    fn test_owned_into_iter_empty() {
        let version = Arc::new(CompactedVersion::empty());
        let snapshot = Arc::new(LiveSnapshot {
            true_inserts: vec![],
            false_inserts: vec![],
            deletes: vec![],
            ops_len: 0,
        });

        let filter_data = FilterData::new(version, snapshot, true);
        let results: Vec<u64> = filter_data.into_iter().collect();

        assert!(results.is_empty());
    }

    #[test]
    fn test_owned_into_sorted_empty() {
        let version = Arc::new(CompactedVersion::empty());
        let snapshot = Arc::new(LiveSnapshot {
            true_inserts: vec![],
            false_inserts: vec![],
            deletes: vec![],
            ops_len: 0,
        });

        let filter_data = FilterData::new(version, snapshot, true);

        let asc: Vec<u64> = filter_data.into_sorted(SortOrder::Ascending).collect();
        assert!(asc.is_empty());
    }

    #[test]
    fn test_owned_false_value() {
        let version = Arc::new(CompactedVersion::empty());
        let snapshot = Arc::new(LiveSnapshot {
            true_inserts: vec![1, 5],
            false_inserts: vec![2, 6, 11],
            deletes: vec![],
            ops_len: 0,
        });

        let filter_data = FilterData::new(version, snapshot, false);
        let results: Vec<u64> = filter_data.into_iter().collect();

        assert_eq!(results, vec![2, 6, 11]);
    }
}
