//! Two-pointer merge and subtract iterators for sorted u64 sequences.
//!
//! These iterators implement classic merge algorithms that operate in O(n + m) time
//! and O(1) extra space (streaming, not materializing). They're used during compaction
//! to combine postings from the compacted version with new inserts from the live layer.
//!
//! # Preconditions
//!
//! Both iterators require their inputs to be **sorted in ascending order**. Violating
//! this precondition results in incorrect output (not detected at runtime for performance).
//!
//! # Algorithm
//!
//! Both use the two-pointer technique:
//! - Maintain a "current" position in each input
//! - Compare current elements, advance the smaller one
//! - For merge: output the smaller, skip duplicates
//! - For subtract: output left values not found in right
//!
//! This is the same approach used in merge sort's merge step and set operations.

use std::cmp::Ordering;

/// Iterator that merges two sorted iterators into a single sorted sequence,
/// removing duplicates.
///
/// # Preconditions
///
/// Both input iterators must yield values in **ascending sorted order**.
/// Unsorted input produces incorrect output.
///
/// # Complexity
///
/// - **Time**: O(n + m) where n and m are the lengths of the input iterators.
///   Each input element is visited exactly once.
/// - **Space**: O(1) extra space. Only stores peekable wrappers and last-seen value.
///
/// # Duplicates
///
/// Duplicates are removed from the output, including:
/// - Duplicates within the same input (e.g., `[1, 1, 2]` → `[1, 2]`)
/// - Values appearing in both inputs (union semantics)
pub struct SortedMerge<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    left: std::iter::Peekable<L>,
    right: std::iter::Peekable<R>,
    /// Tracks the last emitted value for deduplication.
    last_emitted: Option<u64>,
}

impl<L, R> SortedMerge<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    pub fn new(left: L, right: R) -> Self {
        Self {
            left: left.peekable(),
            right: right.peekable(),
            last_emitted: None,
        }
    }
}

impl<L, R> Iterator for SortedMerge<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let next_val = match (self.left.peek(), self.right.peek()) {
                (Some(&l), Some(&r)) => match l.cmp(&r) {
                    Ordering::Less => self.left.next(),
                    Ordering::Greater => self.right.next(),
                    Ordering::Equal => {
                        self.left.next();
                        self.right.next()
                    }
                },
                (Some(_), None) => self.left.next(),
                (None, Some(_)) => self.right.next(),
                (None, None) => return None,
            };

            // Skip duplicates
            if let Some(val) = next_val {
                if self.last_emitted != Some(val) {
                    self.last_emitted = Some(val);
                    return Some(val);
                }
            }
        }
    }
}

/// Iterator that yields elements from the left iterator that are not in the right iterator.
///
/// Implements set difference: `left - right`. Values in `left` that also appear in
/// `right` are skipped.
///
/// # Preconditions
///
/// Both input iterators must yield values in **ascending sorted order**.
/// Unsorted input produces incorrect output.
///
/// # Complexity
///
/// - **Time**: O(n + m) where n = len(left), m = len(right).
///   Each input element is visited at most once.
/// - **Space**: O(1) extra space. Only stores peekable wrappers.
///
/// # Duplicates in Inputs
///
/// - Duplicates in `left` that aren't in `right` are passed through (not deduplicated)
/// - Duplicates in `right` are handled correctly (multiple removals of same value)
pub struct SortedSubtract<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    source: std::iter::Peekable<L>,
    to_remove: std::iter::Peekable<R>,
}

impl<L, R> SortedSubtract<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    pub fn new(source: L, to_remove: R) -> Self {
        Self {
            source: source.peekable(),
            to_remove: to_remove.peekable(),
        }
    }
}

impl<L, R> Iterator for SortedSubtract<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let left_val = *self.source.peek()?;

            // Advance right until it catches up with or passes left
            while let Some(&right_val) = self.to_remove.peek() {
                if right_val < left_val {
                    self.to_remove.next();
                } else {
                    break;
                }
            }

            // Check if left value should be excluded
            if let Some(&right_val) = self.to_remove.peek() {
                if right_val == left_val {
                    // Skip this value - it's in the subtract set.
                    // Only advance source; keep to_remove at the same position
                    // so subsequent duplicates in source are also removed.
                    self.source.next();
                    continue;
                }
            }

            // Left value is not in right, yield it.
            // If we moved past a removed value, advance to_remove now.
            return self.source.next();
        }
    }
}

/// Helper function to create a merge iterator.
///
/// Convenience wrapper around `SortedMerge::new()`.
///
/// # Example
///
/// ```ignore
/// let left = [1, 3, 5].iter().copied();
/// let right = [2, 3, 4].iter().copied();
/// let merged: Vec<u64> = sorted_merge(left, right).collect();
/// assert_eq!(merged, vec![1, 2, 3, 4, 5]);
/// ```
pub fn sorted_merge<L, R>(left: L, right: R) -> SortedMerge<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    SortedMerge::new(left, right)
}

/// Helper function to create a subtract iterator.
///
/// Convenience wrapper around `SortedSubtract::new()`.
///
/// # Example
///
/// ```ignore
/// let values = [1, 2, 3, 4, 5].iter().copied();
/// let to_remove = [2, 4].iter().copied();
/// let result: Vec<u64> = sorted_subtract(values, to_remove).collect();
/// assert_eq!(result, vec![1, 3, 5]);
/// ```
pub fn sorted_subtract<L, R>(left: L, right: R) -> SortedSubtract<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    SortedSubtract::new(left, right)
}

/// Iterator that merges two sorted-descending iterators into a single
/// descending sequence, removing duplicates.
///
/// Mirror of [`SortedMerge`] for inputs sorted in **descending** order.
/// Picks the **larger** value at each step.
///
/// # Preconditions
///
/// Both input iterators must yield values in **descending sorted order**.
pub struct SortedMergeDesc<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    left: std::iter::Peekable<L>,
    right: std::iter::Peekable<R>,
    last_emitted: Option<u64>,
}

impl<L, R> SortedMergeDesc<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    pub fn new(left: L, right: R) -> Self {
        Self {
            left: left.peekable(),
            right: right.peekable(),
            last_emitted: None,
        }
    }
}

impl<L, R> Iterator for SortedMergeDesc<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let next_val = match (self.left.peek(), self.right.peek()) {
                (Some(&l), Some(&r)) => match l.cmp(&r) {
                    Ordering::Greater => self.left.next(),
                    Ordering::Less => self.right.next(),
                    Ordering::Equal => {
                        self.left.next();
                        self.right.next()
                    }
                },
                (Some(_), None) => self.left.next(),
                (None, Some(_)) => self.right.next(),
                (None, None) => return None,
            };

            if let Some(val) = next_val {
                if self.last_emitted != Some(val) {
                    self.last_emitted = Some(val);
                    return Some(val);
                }
            }
        }
    }
}

/// Iterator that yields elements from the left descending iterator that are
/// not in the right descending iterator.
///
/// Mirror of [`SortedSubtract`] for inputs sorted in **descending** order.
///
/// # Preconditions
///
/// Both input iterators must yield values in **descending sorted order**.
pub struct SortedSubtractDesc<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    source: std::iter::Peekable<L>,
    to_remove: std::iter::Peekable<R>,
}

impl<L, R> SortedSubtractDesc<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    pub fn new(source: L, to_remove: R) -> Self {
        Self {
            source: source.peekable(),
            to_remove: to_remove.peekable(),
        }
    }
}

impl<L, R> Iterator for SortedSubtractDesc<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let left_val = *self.source.peek()?;

            // Advance right until it catches up with or passes left (descending)
            while let Some(&right_val) = self.to_remove.peek() {
                if right_val > left_val {
                    self.to_remove.next();
                } else {
                    break;
                }
            }

            if let Some(&right_val) = self.to_remove.peek() {
                if right_val == left_val {
                    // Only advance source; keep to_remove at the same position
                    // so subsequent duplicates in source are also removed.
                    self.source.next();
                    continue;
                }
            }

            return self.source.next();
        }
    }
}

/// Helper function to create a descending merge iterator.
pub fn sorted_merge_desc<L, R>(left: L, right: R) -> SortedMergeDesc<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    SortedMergeDesc::new(left, right)
}

/// Helper function to create a descending subtract iterator.
pub fn sorted_subtract_desc<L, R>(left: L, right: R) -> SortedSubtractDesc<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    SortedSubtractDesc::new(left, right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_empty_inputs() {
        let result: Vec<u64> = sorted_merge(std::iter::empty(), std::iter::empty()).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_merge_left_empty() {
        let right = vec![1, 2, 3];
        let result: Vec<u64> = sorted_merge(std::iter::empty(), right.into_iter()).collect();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_merge_right_empty() {
        let left = vec![1, 2, 3];
        let result: Vec<u64> = sorted_merge(left.into_iter(), std::iter::empty()).collect();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_merge_disjoint() {
        let left = vec![1, 3, 5];
        let right = vec![2, 4, 6];
        let result: Vec<u64> = sorted_merge(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_merge_with_duplicates() {
        let left = vec![1, 2, 3];
        let right = vec![2, 3, 4];
        let result: Vec<u64> = sorted_merge(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_merge_identical() {
        let left = vec![1, 2, 3];
        let right = vec![1, 2, 3];
        let result: Vec<u64> = sorted_merge(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_subtract_empty_inputs() {
        let result: Vec<u64> = sorted_subtract(std::iter::empty(), std::iter::empty()).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_subtract_empty_right() {
        let left = vec![1, 2, 3];
        let result: Vec<u64> = sorted_subtract(left.into_iter(), std::iter::empty()).collect();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_subtract_empty_left() {
        let right = vec![1, 2, 3];
        let result: Vec<u64> = sorted_subtract(std::iter::empty(), right.into_iter()).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_subtract_no_overlap() {
        let left = vec![1, 3, 5];
        let right = vec![2, 4, 6];
        let result: Vec<u64> = sorted_subtract(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![1, 3, 5]);
    }

    #[test]
    fn test_subtract_partial_overlap() {
        let left = vec![1, 2, 3, 4, 5];
        let right = vec![2, 4];
        let result: Vec<u64> = sorted_subtract(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![1, 3, 5]);
    }

    #[test]
    fn test_subtract_all_removed() {
        let left = vec![1, 2, 3];
        let right = vec![1, 2, 3];
        let result: Vec<u64> = sorted_subtract(left.into_iter(), right.into_iter()).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_subtract_subset() {
        let left = vec![1, 2, 3];
        let right = vec![0, 1, 2, 3, 4, 5];
        let result: Vec<u64> = sorted_subtract(left.into_iter(), right.into_iter()).collect();
        assert!(result.is_empty());
    }

    // Descending merge tests

    #[test]
    fn test_merge_desc_empty_inputs() {
        let result: Vec<u64> = sorted_merge_desc(std::iter::empty(), std::iter::empty()).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_merge_desc_left_empty() {
        let right = vec![3, 2, 1];
        let result: Vec<u64> = sorted_merge_desc(std::iter::empty(), right.into_iter()).collect();
        assert_eq!(result, vec![3, 2, 1]);
    }

    #[test]
    fn test_merge_desc_right_empty() {
        let left = vec![3, 2, 1];
        let result: Vec<u64> = sorted_merge_desc(left.into_iter(), std::iter::empty()).collect();
        assert_eq!(result, vec![3, 2, 1]);
    }

    #[test]
    fn test_merge_desc_disjoint() {
        let left = vec![5, 3, 1];
        let right = vec![6, 4, 2];
        let result: Vec<u64> = sorted_merge_desc(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_merge_desc_with_duplicates() {
        let left = vec![3, 2, 1];
        let right = vec![4, 3, 2];
        let result: Vec<u64> = sorted_merge_desc(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![4, 3, 2, 1]);
    }

    #[test]
    fn test_merge_desc_identical() {
        let left = vec![3, 2, 1];
        let right = vec![3, 2, 1];
        let result: Vec<u64> = sorted_merge_desc(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![3, 2, 1]);
    }

    // Descending subtract tests

    #[test]
    fn test_subtract_desc_empty_inputs() {
        let result: Vec<u64> =
            sorted_subtract_desc(std::iter::empty(), std::iter::empty()).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_subtract_desc_empty_right() {
        let left = vec![3, 2, 1];
        let result: Vec<u64> = sorted_subtract_desc(left.into_iter(), std::iter::empty()).collect();
        assert_eq!(result, vec![3, 2, 1]);
    }

    #[test]
    fn test_subtract_desc_empty_left() {
        let right = vec![3, 2, 1];
        let result: Vec<u64> =
            sorted_subtract_desc(std::iter::empty(), right.into_iter()).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_subtract_desc_no_overlap() {
        let left = vec![5, 3, 1];
        let right = vec![6, 4, 2];
        let result: Vec<u64> = sorted_subtract_desc(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![5, 3, 1]);
    }

    #[test]
    fn test_subtract_desc_partial_overlap() {
        let left = vec![5, 4, 3, 2, 1];
        let right = vec![4, 2];
        let result: Vec<u64> = sorted_subtract_desc(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![5, 3, 1]);
    }

    #[test]
    fn test_subtract_desc_all_removed() {
        let left = vec![3, 2, 1];
        let right = vec![3, 2, 1];
        let result: Vec<u64> = sorted_subtract_desc(left.into_iter(), right.into_iter()).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_subtract_desc_subset() {
        let left = vec![3, 2, 1];
        let right = vec![5, 4, 3, 2, 1, 0];
        let result: Vec<u64> = sorted_subtract_desc(left.into_iter(), right.into_iter()).collect();
        assert!(result.is_empty());
    }
}
