#[cfg(unix)]
use memmap2::Mmap;

#[cfg(unix)]
pub fn advise_sequential(mmap: &Mmap) {
    use libc::{c_void, madvise, MADV_SEQUENTIAL};
    unsafe {
        madvise(mmap.as_ptr() as *mut c_void, mmap.len(), MADV_SEQUENTIAL);
    }
}

#[cfg(not(unix))]
pub fn advise_sequential(_mmap: &memmap2::Mmap) {
    // No-op on non-Unix platforms
}

#[cfg(unix)]
pub fn advise_random(mmap: &Mmap) {
    use libc::{c_void, madvise, MADV_RANDOM};
    unsafe {
        madvise(mmap.as_ptr() as *mut c_void, mmap.len(), MADV_RANDOM);
    }
}

#[cfg(not(unix))]
pub fn advise_random(_mmap: &memmap2::Mmap) {
    // No-op on non-Unix platforms
}
