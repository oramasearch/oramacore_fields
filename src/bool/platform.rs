//! Platform-specific memory advisory functions.
//!
//! This module provides hints to the operating system about memory access patterns,
//! which can improve performance by optimizing kernel read-ahead and caching behavior.
//!
//! # Why Sequential Access Hints Matter
//!
//! When iterating through a memory-mapped postings file, the access pattern is strictly
//! sequential (doc_ids are sorted). Without hints, the kernel might:
//! - Use small read-ahead windows (inefficient for sequential scans)
//! - Keep recently accessed pages in cache (wasteful for one-pass iteration)
//!
//! With `MADV_SEQUENTIAL`:
//! - Kernel increases read-ahead aggressively
//! - Kernel may free pages after they're read (no need to cache)
//!
//! # Platform Behavior
//!
//! - **Unix** (Linux, macOS, FreeBSD, etc.): Uses `madvise(MADV_SEQUENTIAL)`.
//!   Can provide 10-30% speedup on large files.
//! - **Other platforms**: No-op. Windows has `PrefetchVirtualMemory` but it's
//!   not exposed here.
//!
//! The no-op fallback ensures portability - code works everywhere, just faster on Unix.

#[cfg(unix)]
use memmap2::Mmap;

/// Advise the kernel about sequential read access pattern for the mmap region.
///
/// Tells the kernel that this memory region will be accessed sequentially from
/// start to end. This is optimal for iteration over sorted postings lists.
///
/// # Platform Support
///
/// - **Unix**: Calls `madvise(MADV_SEQUENTIAL)`. Increases read-ahead and may
///   allow the kernel to free pages after reading. Error return is ignored
///   (advisory call, failure is not fatal).
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
///
/// Provided for API compatibility. The index works correctly without memory
/// advisories, just potentially with less optimal kernel caching behavior.
#[cfg(not(unix))]
pub fn advise_sequential(_mmap: &memmap2::Mmap) {
    // No-op on non-Unix platforms
}
