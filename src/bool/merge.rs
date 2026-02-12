//! Streaming merge and subtract iterators for sorted u64 sequences.
//!
//! Both inputs must be sorted in ascending order (or descending for the `Desc` variants).

use std::cmp::Ordering;

/// Merges two ascending sorted iterators into one sorted sequence, removing duplicates.
pub struct SortedMerge<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    left: std::iter::Peekable<L>,
    right: std::iter::Peekable<R>,
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

/// Yields elements from the left ascending iterator that are not in the right.
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

/// Create a merge iterator from two ascending sorted iterators.
///
/// # Example
///
/// ```rust,ignore
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

/// Create a subtract iterator from two ascending sorted iterators.
///
/// # Example
///
/// ```rust,ignore
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

/// Merges two descending sorted iterators into one descending sequence, removing duplicates.
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

/// Yields elements from the left descending iterator that are not in the right.
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

/// Create a merge iterator from two descending sorted iterators.
pub fn sorted_merge_desc<L, R>(left: L, right: R) -> SortedMergeDesc<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    SortedMergeDesc::new(left, right)
}

/// Create a subtract iterator from two descending sorted iterators.
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
