//! Query iterators for NumberStorage.

use super::compacted::CompactedVersion;
use super::key::IndexableNumber;
use super::live::LiveSnapshot;
use super::merge::{sorted_merge, sorted_merge_descending};
use std::sync::Arc;

/// Sort direction for ordered iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SortOrder {
    /// Ascending order: smallest values first (default)
    #[default]
    Ascending,
    /// Descending order: largest values first
    Descending,
}

/// Filter operations for range queries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FilterOp<T> {
    /// Exact equality: value == target
    Eq(T),
    /// Greater than: value > target
    Gt(T),
    /// Greater than or equal: value >= target
    Gte(T),
    /// Less than: value < target
    Lt(T),
    /// Less than or equal: value <= target
    Lte(T),
    /// Between inclusive: min <= value <= max
    BetweenInclusive(T, T),
}

impl<T: IndexableNumber> FilterOp<T> {
    /// Get the minimum value for the range (inclusive).
    pub fn min_value(&self) -> Option<T> {
        match self {
            FilterOp::Eq(v) => Some(*v),
            FilterOp::Gt(v) => Some(*v), // Will be excluded in filter
            FilterOp::Gte(v) => Some(*v),
            FilterOp::Lt(_) => None,
            FilterOp::Lte(_) => None,
            FilterOp::BetweenInclusive(min, _) => Some(*min),
        }
    }

    /// Get the maximum value for the range (inclusive).
    pub fn max_value(&self) -> Option<T> {
        match self {
            FilterOp::Eq(v) => Some(*v),
            FilterOp::Gt(_) => None,
            FilterOp::Gte(_) => None,
            FilterOp::Lt(v) => Some(*v), // Will be excluded in filter
            FilterOp::Lte(v) => Some(*v),
            FilterOp::BetweenInclusive(_, max) => Some(*max),
        }
    }

    /// Check if a value matches this filter operation.
    pub fn matches(&self, value: T) -> bool {
        match self {
            FilterOp::Eq(target) => T::compare(value, *target) == std::cmp::Ordering::Equal,
            FilterOp::Gt(target) => T::compare(value, *target) == std::cmp::Ordering::Greater,
            FilterOp::Gte(target) => T::compare(value, *target) != std::cmp::Ordering::Less,
            FilterOp::Lt(target) => T::compare(value, *target) == std::cmp::Ordering::Less,
            FilterOp::Lte(target) => T::compare(value, *target) != std::cmp::Ordering::Greater,
            FilterOp::BetweenInclusive(min, max) => {
                T::compare(value, *min) != std::cmp::Ordering::Less
                    && T::compare(value, *max) != std::cmp::Ordering::Greater
            }
        }
    }
}

/// Handle for a filter query result that can be iterated.
pub struct FilterHandle<T: IndexableNumber> {
    version: Arc<CompactedVersion<T>>,
    snapshot: Arc<LiveSnapshot<T>>,
    filter_op: FilterOp<T>,
}

impl<T: IndexableNumber> FilterHandle<T> {
    /// Create a new FilterHandle.
    pub fn new(
        version: Arc<CompactedVersion<T>>,
        snapshot: Arc<LiveSnapshot<T>>,
        filter_op: FilterOp<T>,
    ) -> Self {
        Self {
            version,
            snapshot,
            filter_op,
        }
    }

    /// Create a FilterIterator for this query.
    pub fn iter(&self) -> FilterIterator<'_, T> {
        FilterIterator::new(self)
    }
}

impl<'a, T: IndexableNumber> IntoIterator for &'a FilterHandle<T> {
    type Item = u64;
    type IntoIter = FilterIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: IndexableNumber> IntoIterator for FilterHandle<T> {
    type Item = u64;
    type IntoIter = OwnedFilterIterator<T>;

    fn into_iter(self) -> OwnedFilterIterator<T> {
        let FilterHandle {
            version,
            snapshot,
            filter_op,
        } = self;

        let inner = {
            let min = filter_op.min_value();
            let max = filter_op.max_value();

            let version_ref: &CompactedVersion<T> = &version;
            let snapshot_ref: &LiveSnapshot<T> = &snapshot;

            let live_deletes = Arc::clone(&snapshot_ref.deletes);
            let compacted_deletes = version_ref.deleted_set();
            let compacted_filtered = version_ref.iter_range(min, max).filter(move |(_, doc_id)| {
                !live_deletes.contains(doc_id) && !compacted_deletes.contains(doc_id)
            });

            let live_inserts = snapshot_ref.inserts.iter().copied();
            let merged = sorted_merge(compacted_filtered, live_inserts);

            let result = merged
                .filter(move |(value, _)| filter_op.matches(*value))
                .map(|(_, doc_id)| doc_id);

            let boxed: Box<dyn Iterator<Item = u64> + '_> = Box::new(result);
            // SAFETY: The boxed iterator borrows from version/snapshot via the Arcs.
            // Those Arcs are moved into OwnedFilterIterator, keeping data alive.
            // `inner` is declared before the Arc fields, so it drops first.
            unsafe {
                std::mem::transmute::<
                    Box<dyn Iterator<Item = u64> + '_>,
                    Box<dyn Iterator<Item = u64> + 'static>,
                >(boxed)
            }
        };

        OwnedFilterIterator {
            inner,
            _version: version,
            _snapshot: snapshot,
        }
    }
}

/// Iterator over matching doc_ids from a filter query.
pub struct FilterIterator<'a, T: IndexableNumber> {
    inner: Box<dyn Iterator<Item = u64> + 'a>,
    _marker: std::marker::PhantomData<&'a T>,
}

impl<'a, T: IndexableNumber> FilterIterator<'a, T> {
    fn new(data: &'a FilterHandle<T>) -> Self {
        let filter_op = &data.filter_op;

        // Get range bounds from filter op
        let min = filter_op.min_value();
        let max = filter_op.max_value();

        // Filter compacted entries: remove deleted doc_ids BEFORE merge.
        // This ensures re-inserted doc_ids (in live layer) are not filtered out.
        let live_deletes = Arc::clone(&data.snapshot.deletes);
        let compacted_deletes = data.version.deleted_set();
        let compacted_filtered = data
            .version
            .iter_range(min, max)
            .filter(move |(_, doc_id)| {
                !live_deletes.contains(doc_id) && !compacted_deletes.contains(doc_id)
            });

        // Get live inserts (already correct — refresh_snapshot ensures this)
        let live_inserts = data.snapshot.inserts.iter().copied();

        // Merge compacted (filtered) and live inserts
        let merged = sorted_merge(compacted_filtered, live_inserts);

        // Apply filter operation and extract doc_ids
        let filter_clone = *filter_op;
        let result = merged
            .filter(move |(value, _)| filter_clone.matches(*value))
            .map(|(_, doc_id)| doc_id);

        Self {
            inner: Box::new(result),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: IndexableNumber> Iterator for FilterIterator<'_, T> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

/// Handle for a sort query result that can be iterated.
pub struct SortHandle<T: IndexableNumber> {
    version: Arc<CompactedVersion<T>>,
    snapshot: Arc<LiveSnapshot<T>>,
    order: SortOrder,
}

impl<T: IndexableNumber> SortHandle<T> {
    /// Create a new SortHandle.
    pub fn new(
        version: Arc<CompactedVersion<T>>,
        snapshot: Arc<LiveSnapshot<T>>,
        order: SortOrder,
    ) -> Self {
        Self {
            version,
            snapshot,
            order,
        }
    }

    /// Create a SortIterator for this query.
    pub fn iter(&self) -> SortIterator<'_, T> {
        SortIterator::new(self)
    }
}

impl<'a, T: IndexableNumber> IntoIterator for &'a SortHandle<T> {
    type Item = u64;
    type IntoIter = SortIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: IndexableNumber> IntoIterator for SortHandle<T> {
    type Item = u64;
    type IntoIter = OwnedSortIterator<T>;

    fn into_iter(self) -> OwnedSortIterator<T> {
        let SortHandle {
            version,
            snapshot,
            order,
        } = self;

        let inner = {
            let version_ref: &CompactedVersion<T> = &version;
            let snapshot_ref: &LiveSnapshot<T> = &snapshot;

            match order {
                SortOrder::Ascending => {
                    let live_deletes = Arc::clone(&snapshot_ref.deletes);
                    let compacted_deletes = version_ref.deleted_set();
                    let compacted_filtered = version_ref.iter().filter(move |(_, doc_id)| {
                        !live_deletes.contains(doc_id) && !compacted_deletes.contains(doc_id)
                    });

                    let live_inserts = snapshot_ref.inserts.iter().copied();
                    let merged = sorted_merge(compacted_filtered, live_inserts);
                    let result = merged.map(|(_, doc_id)| doc_id);

                    let boxed: Box<dyn Iterator<Item = u64> + '_> = Box::new(result);
                    // SAFETY: Same as OwnedFilterIterator — Arcs keep data alive.
                    unsafe {
                        std::mem::transmute::<
                            Box<dyn Iterator<Item = u64> + '_>,
                            Box<dyn Iterator<Item = u64> + 'static>,
                        >(boxed)
                    }
                }
                SortOrder::Descending => {
                    let live_deletes = Arc::clone(&snapshot_ref.deletes);
                    let compacted_deletes = version_ref.deleted_set();
                    let compacted_filtered =
                        version_ref.iter_descending().filter(move |(_, doc_id)| {
                            !live_deletes.contains(doc_id) && !compacted_deletes.contains(doc_id)
                        });

                    let live_inserts = snapshot_ref.inserts.iter().copied().rev();
                    let merged = sorted_merge_descending(compacted_filtered, live_inserts);
                    let result = merged.map(|(_, doc_id)| doc_id);

                    let boxed: Box<dyn Iterator<Item = u64> + '_> = Box::new(result);
                    // SAFETY: Same as OwnedFilterIterator — Arcs keep data alive.
                    unsafe {
                        std::mem::transmute::<
                            Box<dyn Iterator<Item = u64> + '_>,
                            Box<dyn Iterator<Item = u64> + 'static>,
                        >(boxed)
                    }
                }
            }
        };

        OwnedSortIterator {
            inner,
            _version: version,
            _snapshot: snapshot,
        }
    }
}

/// Iterator over doc_ids sorted by their associated values.
pub struct SortIterator<'a, T: IndexableNumber> {
    inner: Box<dyn Iterator<Item = u64> + 'a>,
    _marker: std::marker::PhantomData<&'a T>,
}

impl<'a, T: IndexableNumber> SortIterator<'a, T> {
    fn new(data: &'a SortHandle<T>) -> Self {
        match data.order {
            SortOrder::Ascending => Self::new_ascending(data),
            SortOrder::Descending => Self::new_descending(data),
        }
    }

    fn new_ascending(data: &'a SortHandle<T>) -> Self {
        // Filter compacted entries before merge
        let live_deletes = Arc::clone(&data.snapshot.deletes);
        let compacted_deletes = data.version.deleted_set();
        let compacted_filtered = data.version.iter().filter(move |(_, doc_id)| {
            !live_deletes.contains(doc_id) && !compacted_deletes.contains(doc_id)
        });

        // Get live inserts (already sorted ascending)
        let live_inserts = data.snapshot.inserts.iter().copied();

        // Merge and extract doc_ids
        let merged = sorted_merge(compacted_filtered, live_inserts);
        let result = merged.map(|(_, doc_id)| doc_id);

        Self {
            inner: Box::new(result),
            _marker: std::marker::PhantomData,
        }
    }

    fn new_descending(data: &'a SortHandle<T>) -> Self {
        // Filter compacted entries before merge
        let live_deletes = Arc::clone(&data.snapshot.deletes);
        let compacted_deletes = data.version.deleted_set();
        let compacted_filtered = data.version.iter_descending().filter(move |(_, doc_id)| {
            !live_deletes.contains(doc_id) && !compacted_deletes.contains(doc_id)
        });

        // Get live inserts in descending order (reverse the sorted Vec)
        let live_inserts = data.snapshot.inserts.iter().copied().rev();

        // Merge descending and extract doc_ids
        let merged = sorted_merge_descending(compacted_filtered, live_inserts);
        let result = merged.map(|(_, doc_id)| doc_id);

        Self {
            inner: Box::new(result),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: IndexableNumber> Iterator for SortIterator<'_, T> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

/// Owned filter iterator that keeps the underlying data alive via `Arc`s.
///
/// Created by calling [`FilterHandle::into_iter()`].
pub struct OwnedFilterIterator<T: IndexableNumber> {
    // IMPORTANT: `inner` must be declared before the Arc fields so it drops first.
    inner: Box<dyn Iterator<Item = u64> + 'static>,
    _version: Arc<CompactedVersion<T>>,
    _snapshot: Arc<LiveSnapshot<T>>,
}

impl<T: IndexableNumber> Iterator for OwnedFilterIterator<T> {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
        self.inner.next()
    }
}

/// Owned sort iterator that keeps the underlying data alive via `Arc`s.
///
/// Created by calling [`SortHandle::into_iter()`].
pub struct OwnedSortIterator<T: IndexableNumber> {
    // IMPORTANT: `inner` must be declared before the Arc fields so it drops first.
    inner: Box<dyn Iterator<Item = u64> + 'static>,
    _version: Arc<CompactedVersion<T>>,
    _snapshot: Arc<LiveSnapshot<T>>,
}

impl<T: IndexableNumber> Iterator for OwnedSortIterator<T> {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
        self.inner.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_op_matches() {
        // Eq
        assert!(FilterOp::Eq(10u64).matches(10));
        assert!(!FilterOp::Eq(10u64).matches(11));

        // Gt
        assert!(FilterOp::Gt(10u64).matches(11));
        assert!(!FilterOp::Gt(10u64).matches(10));
        assert!(!FilterOp::Gt(10u64).matches(9));

        // Gte
        assert!(FilterOp::Gte(10u64).matches(11));
        assert!(FilterOp::Gte(10u64).matches(10));
        assert!(!FilterOp::Gte(10u64).matches(9));

        // Lt
        assert!(FilterOp::Lt(10u64).matches(9));
        assert!(!FilterOp::Lt(10u64).matches(10));
        assert!(!FilterOp::Lt(10u64).matches(11));

        // Lte
        assert!(FilterOp::Lte(10u64).matches(9));
        assert!(FilterOp::Lte(10u64).matches(10));
        assert!(!FilterOp::Lte(10u64).matches(11));

        // BetweenInclusive
        assert!(FilterOp::BetweenInclusive(10u64, 20).matches(10));
        assert!(FilterOp::BetweenInclusive(10u64, 20).matches(15));
        assert!(FilterOp::BetweenInclusive(10u64, 20).matches(20));
        assert!(!FilterOp::BetweenInclusive(10u64, 20).matches(9));
        assert!(!FilterOp::BetweenInclusive(10u64, 20).matches(21));
    }

    #[test]
    fn test_filter_op_f64() {
        assert!(FilterOp::Eq(1.5f64).matches(1.5));
        assert!(!FilterOp::Eq(1.5f64).matches(1.6));

        assert!(FilterOp::Gt(1.5f64).matches(1.6));
        assert!(!FilterOp::Gt(1.5f64).matches(1.5));

        assert!(FilterOp::BetweenInclusive(-1.0f64, 1.0).matches(0.0));
        assert!(FilterOp::BetweenInclusive(-1.0f64, 1.0).matches(-1.0));
        assert!(FilterOp::BetweenInclusive(-1.0f64, 1.0).matches(1.0));
        assert!(!FilterOp::BetweenInclusive(-1.0f64, 1.0).matches(1.1));
    }

    #[test]
    fn test_filter_op_bounds() {
        let op = FilterOp::BetweenInclusive(10u64, 20);
        assert_eq!(op.min_value(), Some(10));
        assert_eq!(op.max_value(), Some(20));

        let op = FilterOp::Gt(10u64);
        assert_eq!(op.min_value(), Some(10));
        assert_eq!(op.max_value(), None);

        let op = FilterOp::Lt(10u64);
        assert_eq!(op.min_value(), None);
        assert_eq!(op.max_value(), Some(10));
    }
}
