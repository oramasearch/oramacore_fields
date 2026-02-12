use std::cmp::Ordering;

/// Iterator that merges two sorted iterators into a single sorted sequence,
/// removing duplicates.
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

            while let Some(&right_val) = self.to_remove.peek() {
                if right_val < left_val {
                    self.to_remove.next();
                } else {
                    break;
                }
            }

            if let Some(&right_val) = self.to_remove.peek() {
                if right_val == left_val {
                    self.source.next();
                    self.to_remove.next();
                    continue;
                }
            }

            return self.source.next();
        }
    }
}

pub fn sorted_merge<L, R>(left: L, right: R) -> SortedMerge<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    SortedMerge::new(left, right)
}

/// Merge two sorted `&[u64]` slices into a reusable buffer with dedup.
/// Clears `out` before writing. After the call, `out` contains the sorted union.
pub fn merge_sorted_u64_into(a: &[u64], b: &[u64], out: &mut Vec<u64>) {
    out.clear();
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        let va = a[i];
        let vb = b[j];
        match va.cmp(&vb) {
            std::cmp::Ordering::Less => {
                if out.last() != Some(&va) {
                    out.push(va);
                }
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                if out.last() != Some(&vb) {
                    out.push(vb);
                }
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                if out.last() != Some(&va) {
                    out.push(va);
                }
                i += 1;
                j += 1;
            }
        }
    }
    while i < a.len() {
        let va = a[i];
        if out.last() != Some(&va) {
            out.push(va);
        }
        i += 1;
    }
    while j < b.len() {
        let vb = b[j];
        if out.last() != Some(&vb) {
            out.push(vb);
        }
        j += 1;
    }
}

pub fn sorted_subtract<L, R>(left: L, right: R) -> SortedSubtract<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    SortedSubtract::new(left, right)
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
    fn test_merge_into_empty_inputs() {
        let mut out = Vec::new();
        merge_sorted_u64_into(&[], &[], &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_merge_into_one_empty() {
        let mut out = Vec::new();
        merge_sorted_u64_into(&[1, 3, 5], &[], &mut out);
        assert_eq!(out, vec![1, 3, 5]);

        merge_sorted_u64_into(&[], &[2, 4, 6], &mut out);
        assert_eq!(out, vec![2, 4, 6]);
    }

    #[test]
    fn test_merge_into_disjoint() {
        let mut out = Vec::new();
        merge_sorted_u64_into(&[1, 3, 5], &[2, 4, 6], &mut out);
        assert_eq!(out, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_merge_into_overlapping() {
        let mut out = Vec::new();
        merge_sorted_u64_into(&[1, 2, 3], &[2, 3, 4], &mut out);
        assert_eq!(out, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_merge_into_identical() {
        let mut out = Vec::new();
        merge_sorted_u64_into(&[1, 2, 3], &[1, 2, 3], &mut out);
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn test_merge_into_reuses_buffer() {
        let mut out = Vec::with_capacity(100);
        merge_sorted_u64_into(&[1, 2], &[3, 4], &mut out);
        assert_eq!(out, vec![1, 2, 3, 4]);
        assert!(out.capacity() >= 100); // capacity preserved

        merge_sorted_u64_into(&[10], &[20], &mut out);
        assert_eq!(out, vec![10, 20]);
    }

    #[test]
    fn test_merge_large_sequences() {
        let left: Vec<u64> = (0..1000).step_by(2).collect();
        let right: Vec<u64> = (1..1000).step_by(2).collect();
        let result: Vec<u64> = sorted_merge(left.into_iter(), right.into_iter()).collect();
        let expected: Vec<u64> = (0..1000).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtract_right_larger_than_left() {
        let left = vec![5u64, 10];
        let right = vec![1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let result: Vec<u64> = sorted_subtract(left.into_iter(), right.into_iter()).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_merge_single_element_each() {
        let result: Vec<u64> =
            sorted_merge(vec![1u64].into_iter(), vec![2u64].into_iter()).collect();
        assert_eq!(result, vec![1, 2]);
    }

    #[test]
    fn test_merge_single_element_same() {
        let result: Vec<u64> =
            sorted_merge(vec![1u64].into_iter(), vec![1u64].into_iter()).collect();
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn test_subtract_interleaved() {
        let left = vec![1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let right = vec![2u64, 4, 6, 8, 10];
        let result: Vec<u64> = sorted_subtract(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![1, 3, 5, 7, 9]);
    }

    #[test]
    fn test_subtract_single_element() {
        let result: Vec<u64> =
            sorted_subtract(vec![5u64].into_iter(), vec![5u64].into_iter()).collect();
        assert!(result.is_empty());

        let result: Vec<u64> =
            sorted_subtract(vec![5u64].into_iter(), vec![3u64].into_iter()).collect();
        assert_eq!(result, vec![5]);
    }

    #[test]
    fn test_merge_into_large_sequences() {
        let a: Vec<u64> = (0..500).step_by(2).collect();
        let b: Vec<u64> = (1..500).step_by(2).collect();
        let mut out = Vec::new();
        merge_sorted_u64_into(&a, &b, &mut out);
        let expected: Vec<u64> = (0..500).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn test_merge_with_duplicates_within_same_side() {
        // Both sides have same values
        let left = vec![1u64, 1, 2, 3, 3];
        let right = vec![2u64, 3, 4, 4];
        let result: Vec<u64> = sorted_merge(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_merge_into_with_duplicates_within_same_side() {
        let mut out = Vec::new();
        merge_sorted_u64_into(&[1, 1, 2, 3, 3], &[2, 3, 4, 4], &mut out);
        assert_eq!(out, vec![1, 2, 3, 4]);
    }
}
