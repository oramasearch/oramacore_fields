#[cfg(unix)]
use memmap2::Mmap;

#[cfg(unix)]
pub fn advise_sequential(mmap: &Mmap) {
    use libc::{MADV_SEQUENTIAL, c_void, madvise};
    unsafe {
        madvise(mmap.as_ptr() as *mut c_void, mmap.len(), MADV_SEQUENTIAL);
    }
}

#[cfg(not(unix))]
pub fn advise_sequential(_mmap: &memmap2::Mmap) {}

#[cfg(unix)]
pub fn advise_random(mmap: &Mmap) {
    use libc::{MADV_RANDOM, c_void, madvise};
    unsafe {
        madvise(mmap.as_ptr() as *mut c_void, mmap.len(), MADV_RANDOM);
    }
}

#[cfg(not(unix))]
pub fn advise_random(_mmap: &memmap2::Mmap) {}

#[cfg(unix)]
pub fn advise_willneed(mmap: &Mmap) {
    use libc::{MADV_WILLNEED, c_void, madvise};
    unsafe {
        madvise(mmap.as_ptr() as *mut c_void, mmap.len(), MADV_WILLNEED);
    }
}

#[cfg(not(unix))]
pub fn advise_willneed(_mmap: &memmap2::Mmap) {}
