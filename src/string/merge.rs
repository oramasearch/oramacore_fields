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

pub fn sorted_merge<L, R>(left: L, right: R) -> SortedMerge<L, R>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
{
    SortedMerge::new(left, right)
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
}
