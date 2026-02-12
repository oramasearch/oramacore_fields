//! Platform-specific I/O hints for memory-mapped files.

use memmap2::Mmap;

/// Hint that the region will be accessed sequentially.
#[cfg(unix)]
pub fn advise_sequential(mmap: &Mmap) {
    unsafe {
        let ret = libc::madvise(
            mmap.as_ptr() as *mut libc::c_void,
            mmap.len(),
            libc::MADV_SEQUENTIAL,
        );
        if ret != 0 {
            tracing::warn!(
                "madvise(MADV_SEQUENTIAL) failed: {}",
                std::io::Error::last_os_error()
            );
        }
    }
}

/// No-op on non-Unix platforms.
#[cfg(not(unix))]
pub fn advise_sequential(_mmap: &Mmap) {
    // No-op
}

/// Hint that the region will be accessed randomly.
#[cfg(unix)]
pub fn advise_random(mmap: &Mmap) {
    unsafe {
        let ret = libc::madvise(
            mmap.as_ptr() as *mut libc::c_void,
            mmap.len(),
            libc::MADV_RANDOM,
        );
        if ret != 0 {
            tracing::warn!(
                "madvise(MADV_RANDOM) failed: {}",
                std::io::Error::last_os_error()
            );
        }
    }
}

/// No-op on non-Unix platforms.
#[cfg(not(unix))]
pub fn advise_random(_mmap: &Mmap) {
    // No-op
}
