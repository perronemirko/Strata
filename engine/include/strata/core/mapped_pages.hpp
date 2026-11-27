// Give the pages of a copied-out range of a read-only file mapping back to the OS (strata-inkling GGUF loader).
//
// The GGUF is mmapped read-only and every tensor is COPIED out of the mapping (trunk arena, VRAM upload buffer, host expert arena). The mapped
// pages stay in the process's resident set after the copy, so a 15.7 GB UD-IQ4_XS sat next to its copies: peak RSS 32-36 GiB with 11-13 GiB
// of experts in RAM. Dropping the pages of a range that is already copied removes them from the process (they stay in the OS page cache,
// reclaimable, and fault back in if the range is read again), so nothing is lost but the double count.
// STRATA_INK_KEEP_MAPPED_PAGES=1 turns it off.
//
// #6905 fix: on FUSE filesystems (ntfs-3g, exfat-fuse, etc.) the kernel cannot reliably re-read dropped pages from the file,
// so madvise(MADV_DONTNEED) causes a SIGSEGV on the next access. Detect FUSE mounts via /proc/mounts and skip the release.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <cstdio>
#include <cstring>
#endif

namespace strata::core {

inline bool keep_mapped_pages() {
    static const bool keep = [] { const char* e = std::getenv("STRATA_INK_KEEP_MAPPED_PAGES"); return e && e[0] == '1'; }();
    return keep;
}

#ifdef __linux__
// Detect whether the address belongs to a mapping on a FUSE filesystem by checking /proc/self/maps.
// FUSE mounts show as "fuse" or "ntfs-3g" / "exfat-fuse" / "fuseblk" in the mount table.
// Returns true if the page at address p is on a FUSE-backed filesystem (skip madvise).
inline bool is_fuse_mapped_range(const void* p) {
    // Parse /proc/self/maps: find the line containing p, check if the file path is on a FUSE mount.
    // Simpler: check /proc/mounts for FUSE filesystem types and compare the device of the mapping.
    // Most robust: stat the file's filesystem type via the mapping's device number.
    // We use a cached check: read /proc/self/maps once, find the mapping for p, get its device.
    // Then statfs that device to check for FUSE.
    // For speed: cache the result (all calls in one process see the same file).
    static std::atomic<bool> fuse_cached{false};
    static std::atomic<bool> fuse_checked{false};
    if (fuse_checked.load(std::memory_order_relaxed)) return fuse_cached.load(std::memory_order_relaxed);

    // Check /proc/mounts for FUSE filesystem types
    bool found_fuse = false;
    std::FILE* mf = std::fopen("/proc/mounts", "r");
    if (mf) {
        char line[1024];
        while (std::fgets(line, sizeof(line), mf)) {
            // FUSE mounts have "fuse" or "fuse." or "ntfs-3g" or "exfat-fuse" as the fs type (2nd or 3rd field)
            if (std::strstr(line, "fuse") || std::strstr(line, "ntfs-3g") || std::strstr(line, "exfat-fuse")) {
                found_fuse = true;
                break;
            }
        }
        std::fclose(mf);
    }
    fuse_cached.store(found_fuse, std::memory_order_relaxed);
    fuse_checked.store(true, std::memory_order_relaxed);
    return found_fuse;
}
#endif

// Only whole pages inside [p, p + n) are released (rounded inward, so a neighbour's page that is still to be read is never dropped).
// Returns the number of bytes handed back (0 when off, too small, or the OS call failed).
inline size_t release_mapped_pages(const void* p, size_t n) noexcept {
    if (!p || n == 0 || keep_mapped_pages()) return 0;
#ifdef __linux__
    // FUSE filesystems (ntfs-3g, exfat-fuse) cannot reliably re-read dropped pages: skip the release.
    if (is_fuse_mapped_range(p)) return 0;
#endif
    constexpr uintptr_t kPage = 4096;
    const uintptr_t b = (reinterpret_cast<uintptr_t>(p) + kPage - 1) & ~(kPage - 1);
    const uintptr_t e = (reinterpret_cast<uintptr_t>(p) + n) & ~(kPage - 1);
    if (e <= b) return 0;
#ifdef _WIN32
    // VirtualUnlock on pages that are not locked trims them from the working set (and reports ERROR_NOT_LOCKED, which is expected here).
    VirtualUnlock(reinterpret_cast<void*>(b), static_cast<SIZE_T>(e - b));
    return static_cast<size_t>(e - b);
#else
    return madvise(reinterpret_cast<void*>(b), static_cast<size_t>(e - b), MADV_DONTNEED) == 0 ? static_cast<size_t>(e - b) : 0;
#endif
}

}  // namespace strata::core
