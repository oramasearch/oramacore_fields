//! Platform-specific memory access hints for improved read performance.

#[cfg(unix)]
use memmap2::Mmap;

/// Hint to the OS that this memory region will be read sequentially.
///
/// # Safety
///
/// The `unsafe` block is safe because:
/// - `mmap.as_ptr()` returns a valid pointer to mapped memory
/// - `mmap.len()` is the exact size of the mapping
/// - `MADV_SEQUENTIAL` is a hint that doesn't modify memory or cause UB on failure
#[cfg(unix)]
pub fn advise_sequential(mmap: &Mmap) {
    use libc::{c_void, madvise, MADV_SEQUENTIAL};
    // SAFETY: mmap provides valid pointer and length. madvise is advisory
    // and won't cause UB even if it fails.
    unsafe {
        madvise(mmap.as_ptr() as *mut c_void, mmap.len(), MADV_SEQUENTIAL);
    }
}

/// No-op on non-Unix platforms.
#[cfg(not(unix))]
pub fn advise_sequential(_mmap: &memmap2::Mmap) {
    // No-op on non-Unix platforms
}
