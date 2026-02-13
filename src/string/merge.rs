use std::cmp::Ordering;

/// Iterator that produces the sorted, deduplicated union of two sorted iterators.
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

/// Iterator that yields elements from the first sorted iterator that are absent from the second.
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

#[allow(dead_code)]
pub fn sorted_subtract<L, R>(left: L, right: R) -> SortedSubtract<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    SortedSubtract::new(left, right)
}

/// Merge two sorted slices into `out`, producing a sorted, deduplicated union.
/// Clears `out` before writing.
#[allow(dead_code)]
pub fn merge_sorted_u64_into(a: &[u64], b: &[u64], out: &mut Vec<u64>) {
    out.clear();
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        let va = a[i];
        let vb = b[j];
        match va.cmp(&vb) {
            Ordering::Less => {
                if out.last() != Some(&va) {
                    out.push(va);
                }
                i += 1;
            }
            Ordering::Greater => {
                if out.last() != Some(&vb) {
                    out.push(vb);
                }
                j += 1;
            }
            Ordering::Equal => {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_empty_inputs() {
        let result: Vec<u64> = sorted_merge(std::iter::empty(), std::iter::empty()).collect();
        assert!(result.is_empty());
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
    fn test_subtract_partial_overlap() {
        let left = vec![1, 2, 3, 4, 5];
        let right = vec![2, 4];
        let result: Vec<u64> = sorted_subtract(left.into_iter(), right.into_iter()).collect();
        assert_eq!(result, vec![1, 3, 5]);
    }

    #[test]
    fn test_merge_into_overlapping() {
        let mut out = Vec::new();
        merge_sorted_u64_into(&[1, 2, 3], &[2, 3, 4], &mut out);
        assert_eq!(out, vec![1, 2, 3, 4]);
    }
}
