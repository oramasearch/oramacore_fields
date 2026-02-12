//! Sorted merge algorithms.
//!
//! These algorithms are used to efficiently combine live layer data
//! with compacted version data during queries and compaction.

use super::key::IndexableNumber;
use std::cmp::Ordering;
use std::iter::Peekable;

/// A two-pointer merge iterator that combines two sorted iterators.
///
/// Yields items from both iterators in sorted order, with deduplication.
pub struct SortedMerge<L, R, T>
where
    L: Iterator<Item = (T, u64)>,
    R: Iterator<Item = (T, u64)>,
    T: IndexableNumber,
{
    left: Peekable<L>,
    right: Peekable<R>,
    last: Option<(T, u64)>,
}

impl<L, R, T> SortedMerge<L, R, T>
where
    L: Iterator<Item = (T, u64)>,
    R: Iterator<Item = (T, u64)>,
    T: IndexableNumber,
{
    /// Create a new SortedMerge from two sorted iterators.
    pub fn new(left: L, right: R) -> Self {
        Self {
            left: left.peekable(),
            right: right.peekable(),
            last: None,
        }
    }
}

impl<L, R, T> Iterator for SortedMerge<L, R, T>
where
    L: Iterator<Item = (T, u64)>,
    R: Iterator<Item = (T, u64)>,
    T: IndexableNumber,
{
    type Item = (T, u64);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let next_val = match (self.left.peek(), self.right.peek()) {
                (None, None) => return None,
                (Some(_), None) => self.left.next(),
                (None, Some(_)) => self.right.next(),
                (Some(&(l_val, l_doc)), Some(&(r_val, r_doc))) => {
                    let val_cmp = T::compare(l_val, r_val);
                    match val_cmp {
                        Ordering::Less => self.left.next(),
                        Ordering::Greater => self.right.next(),
                        Ordering::Equal => {
                            // Same value, compare by doc_id
                            match l_doc.cmp(&r_doc) {
                                Ordering::Less => self.left.next(),
                                Ordering::Greater => self.right.next(),
                                Ordering::Equal => {
                                    // Same (value, doc_id) - consume both, yield once
                                    self.left.next();
                                    self.right.next()
                                }
                            }
                        }
                    }
                }
            };

            // Deduplicate consecutive equal items
            if let Some(val) = next_val {
                if let Some(last) = self.last {
                    if T::compare(val.0, last.0) == Ordering::Equal && val.1 == last.1 {
                        continue; // Skip duplicate
                    }
                }
                self.last = Some(val);
                return Some(val);
            }
        }
    }
}

/// Create a SortedMerge iterator from two sorted iterators.
pub fn sorted_merge<L, R, T>(left: L, right: R) -> SortedMerge<L, R, T>
where
    L: Iterator<Item = (T, u64)>,
    R: Iterator<Item = (T, u64)>,
    T: IndexableNumber,
{
    SortedMerge::new(left, right)
}

/// A two-pointer merge iterator for descending order.
///
/// Yields items from both iterators in descending order, with deduplication.
pub struct SortedMergeDescending<L, R, T>
where
    L: Iterator<Item = (T, u64)>,
    R: Iterator<Item = (T, u64)>,
    T: IndexableNumber,
{
    left: Peekable<L>,
    right: Peekable<R>,
    last: Option<(T, u64)>,
}

impl<L, R, T> SortedMergeDescending<L, R, T>
where
    L: Iterator<Item = (T, u64)>,
    R: Iterator<Item = (T, u64)>,
    T: IndexableNumber,
{
    /// Create a new SortedMergeDescending from two descending-sorted iterators.
    pub fn new(left: L, right: R) -> Self {
        Self {
            left: left.peekable(),
            right: right.peekable(),
            last: None,
        }
    }
}

impl<L, R, T> Iterator for SortedMergeDescending<L, R, T>
where
    L: Iterator<Item = (T, u64)>,
    R: Iterator<Item = (T, u64)>,
    T: IndexableNumber,
{
    type Item = (T, u64);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let next_val = match (self.left.peek(), self.right.peek()) {
                (None, None) => return None,
                (Some(_), None) => self.left.next(),
                (None, Some(_)) => self.right.next(),
                (Some(&(l_val, l_doc)), Some(&(r_val, r_doc))) => {
                    let val_cmp = T::compare(l_val, r_val);
                    match val_cmp {
                        // For descending: take GREATER value first
                        Ordering::Greater => self.left.next(),
                        Ordering::Less => self.right.next(),
                        Ordering::Equal => {
                            // Same value, compare by doc_id (descending)
                            match l_doc.cmp(&r_doc) {
                                Ordering::Greater => self.left.next(),
                                Ordering::Less => self.right.next(),
                                Ordering::Equal => {
                                    // Same (value, doc_id) - consume both, yield once
                                    self.left.next();
                                    self.right.next()
                                }
                            }
                        }
                    }
                }
            };

            // Deduplicate consecutive equal items
            if let Some(val) = next_val {
                if let Some(last) = self.last {
                    if T::compare(val.0, last.0) == Ordering::Equal && val.1 == last.1 {
                        continue; // Skip duplicate
                    }
                }
                self.last = Some(val);
                return Some(val);
            }
        }
    }
}

/// Create a SortedMergeDescending iterator from two descending-sorted iterators.
pub fn sorted_merge_descending<L, R, T>(left: L, right: R) -> SortedMergeDescending<L, R, T>
where
    L: Iterator<Item = (T, u64)>,
    R: Iterator<Item = (T, u64)>,
    T: IndexableNumber,
{
    SortedMergeDescending::new(left, right)
}

/// Merge two sorted doc_id iterators (for deleted lists).
pub struct SortedMergeDocIds<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    left: Peekable<L>,
    right: Peekable<R>,
    last: Option<u64>,
}

impl<L, R> SortedMergeDocIds<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    pub fn new(left: L, right: R) -> Self {
        Self {
            left: left.peekable(),
            right: right.peekable(),
            last: None,
        }
    }
}

impl<L, R> Iterator for SortedMergeDocIds<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let next_val = match (self.left.peek(), self.right.peek()) {
                (None, None) => return None,
                (Some(_), None) => self.left.next(),
                (None, Some(_)) => self.right.next(),
                (Some(&l), Some(&r)) => match l.cmp(&r) {
                    Ordering::Less => self.left.next(),
                    Ordering::Greater => self.right.next(),
                    Ordering::Equal => {
                        self.left.next();
                        self.right.next()
                    }
                },
            };

            // Deduplicate
            if let Some(val) = next_val {
                if self.last != Some(val) {
                    self.last = Some(val);
                    return Some(val);
                }
            }
        }
    }
}

/// Create a SortedMergeDocIds iterator.
pub fn sorted_merge_doc_ids<L, R>(left: L, right: R) -> SortedMergeDocIds<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    SortedMergeDocIds::new(left, right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sorted_merge_empty() {
        let left: Vec<(u64, u64)> = vec![];
        let right: Vec<(u64, u64)> = vec![];
        let result: Vec<_> = sorted_merge(left.into_iter(), right.into_iter()).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_sorted_merge_left_only() {
        let left = vec![(10u64, 1), (20, 2)];
        let right: Vec<(u64, u64)> = vec![];
        let result: Vec<_> = sorted_merge(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![(10, 1), (20, 2)]);
    }

    #[test]
    fn test_sorted_merge_right_only() {
        let left: Vec<(u64, u64)> = vec![];
        let right = vec![(10u64, 1), (20, 2)];
        let result: Vec<_> = sorted_merge(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![(10, 1), (20, 2)]);
    }

    #[test]
    fn test_sorted_merge_interleaved() {
        let left = vec![(10u64, 1), (30, 3)];
        let right = vec![(20u64, 2), (40, 4)];
        let result: Vec<_> = sorted_merge(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![(10, 1), (20, 2), (30, 3), (40, 4)]);
    }

    #[test]
    fn test_sorted_merge_duplicates() {
        let left = vec![(10u64, 1), (20, 2)];
        let right = vec![(10u64, 1), (30, 3)]; // (10, 1) is duplicate
        let result: Vec<_> = sorted_merge(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![(10, 1), (20, 2), (30, 3)]);
    }

    #[test]
    fn test_sorted_merge_same_value_different_doc() {
        let left = vec![(10u64, 1), (10, 3)];
        let right = vec![(10u64, 2)];
        let result: Vec<_> = sorted_merge(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![(10, 1), (10, 2), (10, 3)]);
    }

    #[test]
    fn test_sorted_merge_doc_ids() {
        let left = vec![1u64, 3, 5];
        let right = vec![2u64, 3, 6]; // 3 is duplicate
        let result: Vec<_> = sorted_merge_doc_ids(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![1, 2, 3, 5, 6]);
    }

    #[test]
    fn test_f64_merge() {
        let left = vec![(1.0f64, 1), (3.0, 3)];
        let right = vec![(2.0f64, 2), (4.0, 4)];
        let result: Vec<_> = sorted_merge(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result.len(), 4);
        assert!((result[0].0 - 1.0).abs() < f64::EPSILON);
        assert_eq!(result[0].1, 1);
        assert!((result[1].0 - 2.0).abs() < f64::EPSILON);
        assert_eq!(result[1].1, 2);
        assert!((result[2].0 - 3.0).abs() < f64::EPSILON);
        assert_eq!(result[2].1, 3);
        assert!((result[3].0 - 4.0).abs() < f64::EPSILON);
        assert_eq!(result[3].1, 4);
    }

    // --- SortedMerge: same value, left doc_id > right doc_id ---
    #[test]
    fn test_sorted_merge_same_value_left_doc_greater() {
        let left = vec![(10u64, 5)];
        let right = vec![(10u64, 2)];
        let result: Vec<_> = sorted_merge(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![(10, 2), (10, 5)]);
    }

    // --- SortedMerge: consecutive dedup across iterators ---
    #[test]
    fn test_sorted_merge_consecutive_dedup() {
        // Both iterators have (10, 1) followed by (10, 1) again from the other side
        // The Equal+Equal branch deduplicates the pair, but we also need the
        // consecutive dedup path (line 78) to fire: produce same item twice in a row.
        // This happens when e.g. left yields (10,1) then right yields (10,1) in sequence
        // but they aren't peeked simultaneously.
        // Easiest way: have duplicates within a single iterator side.
        let left = vec![(10u64, 1), (10, 1)];
        let right: Vec<(u64, u64)> = vec![];
        let result: Vec<_> = sorted_merge(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![(10, 1)]);
    }

    // --- SortedMergeDescending: all paths ---

    #[test]
    fn test_sorted_merge_desc_empty() {
        let left: Vec<(u64, u64)> = vec![];
        let right: Vec<(u64, u64)> = vec![];
        let result: Vec<_> = sorted_merge_descending(left.into_iter(), right.into_iter()).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_sorted_merge_desc_left_only() {
        let left = vec![(20u64, 2), (10, 1)];
        let right: Vec<(u64, u64)> = vec![];
        let result: Vec<_> = sorted_merge_descending(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![(20, 2), (10, 1)]);
    }

    #[test]
    fn test_sorted_merge_desc_right_only() {
        let left: Vec<(u64, u64)> = vec![];
        let right = vec![(20u64, 2), (10, 1)];
        let result: Vec<_> = sorted_merge_descending(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![(20, 2), (10, 1)]);
    }

    #[test]
    fn test_sorted_merge_desc_interleaved() {
        let left = vec![(40u64, 4), (20, 2)];
        let right = vec![(30u64, 3), (10, 1)];
        let result: Vec<_> = sorted_merge_descending(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![(40, 4), (30, 3), (20, 2), (10, 1)]);
    }

    #[test]
    fn test_sorted_merge_desc_same_value_left_doc_greater() {
        // Equal value, left doc > right doc → take left first (descending doc order)
        let left = vec![(10u64, 5)];
        let right = vec![(10u64, 2)];
        let result: Vec<_> = sorted_merge_descending(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![(10, 5), (10, 2)]);
    }

    #[test]
    fn test_sorted_merge_desc_same_value_left_doc_less() {
        // Equal value, left doc < right doc → take right first (descending doc order)
        let left = vec![(10u64, 2)];
        let right = vec![(10u64, 5)];
        let result: Vec<_> = sorted_merge_descending(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![(10, 5), (10, 2)]);
    }

    #[test]
    fn test_sorted_merge_desc_duplicates() {
        // Same (value, doc_id) in both → consume both, yield once
        let left = vec![(10u64, 1)];
        let right = vec![(10u64, 1)];
        let result: Vec<_> = sorted_merge_descending(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![(10, 1)]);
    }

    #[test]
    fn test_sorted_merge_desc_consecutive_dedup() {
        // Consecutive duplicates within a single iterator side
        let left = vec![(10u64, 1), (10, 1)];
        let right: Vec<(u64, u64)> = vec![];
        let result: Vec<_> = sorted_merge_descending(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![(10, 1)]);
    }

    #[test]
    fn test_sorted_merge_desc_f64() {
        let left = vec![(4.0f64, 4), (2.0, 2)];
        let right = vec![(3.0f64, 3), (1.0, 1)];
        let result: Vec<_> = sorted_merge_descending(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result.len(), 4);
        assert!((result[0].0 - 4.0).abs() < f64::EPSILON);
        assert!((result[1].0 - 3.0).abs() < f64::EPSILON);
        assert!((result[2].0 - 2.0).abs() < f64::EPSILON);
        assert!((result[3].0 - 1.0).abs() < f64::EPSILON);
    }

    // --- SortedMergeDocIds: missing paths ---

    #[test]
    fn test_sorted_merge_doc_ids_empty() {
        let left: Vec<u64> = vec![];
        let right: Vec<u64> = vec![];
        let result: Vec<_> = sorted_merge_doc_ids(left.into_iter(), right.into_iter()).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_sorted_merge_doc_ids_left_only() {
        let left = vec![1u64, 3, 5];
        let right: Vec<u64> = vec![];
        let result: Vec<_> = sorted_merge_doc_ids(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![1, 3, 5]);
    }

    #[test]
    fn test_sorted_merge_doc_ids_right_only() {
        let left: Vec<u64> = vec![];
        let right = vec![2u64, 4, 6];
        let result: Vec<_> = sorted_merge_doc_ids(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![2, 4, 6]);
    }

    #[test]
    fn test_sorted_merge_doc_ids_consecutive_dedup() {
        // Consecutive duplicates within a single side
        let left = vec![1u64, 1, 3];
        let right: Vec<u64> = vec![];
        let result: Vec<_> = sorted_merge_doc_ids(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![1, 3]);
    }
}
