use anyhow::{Context, Result};
use memmap2::MmapMut;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

const TMP_FILENAME: &str = ".tmp_points";

pub struct MmapVecWriter<T: Copy> {
    writer: BufWriter<File>,
    path: PathBuf,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy> MmapVecWriter<T> {
    pub fn new(dir: &Path) -> Result<Self> {
        let path = dir.join(TMP_FILENAME);
        let file = File::create(&path)
            .with_context(|| format!("Failed to create temp file: {path:?}"))?;
        Ok(Self {
            writer: BufWriter::new(file),
            path,
            len: 0,
            _marker: PhantomData,
        })
    }

    pub fn push(&mut self, entry: &T) -> Result<()> {
        let bytes = unsafe {
            std::slice::from_raw_parts(entry as *const T as *const u8, std::mem::size_of::<T>())
        };
        self.writer.write_all(bytes)?;
        self.len += 1;
        Ok(())
    }

    pub fn finish(self) -> Result<MmapVec<T>> {
        let file = self.writer.into_inner().map_err(|e| e.into_error())
            .with_context(|| format!("Failed to flush temp file: {:?}", self.path))?;
        file.sync_all()
            .with_context(|| format!("Failed to sync temp file: {:?}", self.path))?;
        drop(file);

        if self.len == 0 {
            return Ok(MmapVec {
                mmap: None,
                path: self.path,
                len: 0,
                _marker: PhantomData,
            });
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)
            .with_context(|| format!("Failed to reopen temp file: {:?}", self.path))?;
        let mmap = unsafe { MmapMut::map_mut(&file) }
            .with_context(|| format!("Failed to mmap temp file: {:?}", self.path))?;

        Ok(MmapVec {
            mmap: Some(mmap),
            path: self.path,
            len: self.len,
            _marker: PhantomData,
        })
    }
}

pub struct MmapVec<T: Copy> {
    mmap: Option<MmapMut>,
    path: PathBuf,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy> MmapVec<T> {
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self.mmap {
            Some(ref mut mmap) => unsafe {
                std::slice::from_raw_parts_mut(mmap.as_mut_ptr() as *mut T, self.len)
            },
            None => &mut [],
        }
    }

    pub fn set_len(&mut self, len: usize) {
        debug_assert!(len <= self.len);
        self.len = len;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T: Copy> Drop for MmapVec<T> {
    fn drop(&mut self) {
        // Drop the mmap before removing the file
        self.mmap.take();
        let _ = std::fs::remove_file(&self.path);
    }
}

pub fn dedup_sorted<T: PartialEq>(slice: &mut [T]) -> usize {
    if slice.is_empty() {
        return 0;
    }
    let mut write = 1;
    for read in 1..slice.len() {
        if slice[read] != slice[write - 1] {
            if write != read {
                slice.swap(write, read);
            }
            write += 1;
        }
    }
    write
}

pub fn retain_in_place<T>(slice: &mut [T], f: impl Fn(&T) -> bool) -> usize {
    let mut write = 0;
    for read in 0..slice.len() {
        if f(&slice[read]) {
            if write != read {
                slice.swap(write, read);
            }
            write += 1;
        }
    }
    write
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dedup_sorted_empty() {
        let mut data: Vec<i32> = vec![];
        assert_eq!(dedup_sorted(&mut data), 0);
    }

    #[test]
    fn test_dedup_sorted_no_dups() {
        let mut data = vec![1, 2, 3, 4];
        let len = dedup_sorted(&mut data);
        assert_eq!(len, 4);
        assert_eq!(&data[..len], &[1, 2, 3, 4]);
    }

    #[test]
    fn test_dedup_sorted_with_dups() {
        let mut data = vec![1, 1, 2, 3, 3, 3, 4];
        let len = dedup_sorted(&mut data);
        assert_eq!(len, 4);
        assert_eq!(&data[..len], &[1, 2, 3, 4]);
    }

    #[test]
    fn test_dedup_sorted_all_same() {
        let mut data = vec![5, 5, 5];
        let len = dedup_sorted(&mut data);
        assert_eq!(len, 1);
        assert_eq!(&data[..len], &[5]);
    }

    #[test]
    fn test_retain_in_place_keep_all() {
        let mut data = vec![1, 2, 3];
        let len = retain_in_place(&mut data, |_| true);
        assert_eq!(len, 3);
        assert_eq!(&data[..len], &[1, 2, 3]);
    }

    #[test]
    fn test_retain_in_place_keep_none() {
        let mut data = vec![1, 2, 3];
        let len = retain_in_place(&mut data, |_| false);
        assert_eq!(len, 0);
    }

    #[test]
    fn test_retain_in_place_keep_even() {
        let mut data = vec![1, 2, 3, 4, 5, 6];
        let len = retain_in_place(&mut data, |x| x % 2 == 0);
        assert_eq!(len, 3);
        assert_eq!(&data[..len], &[2, 4, 6]);
    }

    #[test]
    fn test_mmap_vec_writer_roundtrip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut writer: MmapVecWriter<u64> = MmapVecWriter::new(tmp.path()).unwrap();
        for i in 0u64..100 {
            writer.push(&i).unwrap();
        }
        let mut vec = writer.finish().unwrap();
        assert_eq!(vec.len(), 100);
        assert!(!vec.is_empty());

        let slice = vec.as_mut_slice();
        for (i, val) in slice.iter().enumerate() {
            assert_eq!(*val, i as u64);
        }

        // Test set_len
        vec.set_len(50);
        assert_eq!(vec.len(), 50);
        let slice = vec.as_mut_slice();
        assert_eq!(slice.len(), 50);
    }

    #[test]
    fn test_mmap_vec_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let writer: MmapVecWriter<u64> = MmapVecWriter::new(tmp.path()).unwrap();
        let mut vec = writer.finish().unwrap();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
        assert_eq!(vec.as_mut_slice().len(), 0);
    }

    #[test]
    fn test_mmap_vec_drop_cleans_up() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join(TMP_FILENAME);
        {
            let mut writer: MmapVecWriter<u64> = MmapVecWriter::new(tmp.path()).unwrap();
            writer.push(&42u64).unwrap();
            let _vec = writer.finish().unwrap();
            assert!(path.exists());
        }
        assert!(!path.exists());
    }
}
