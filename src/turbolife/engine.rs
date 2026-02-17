use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(target_arch = "x86_64")]
use std::time::Instant;

use super::activity::{
    append_expand_candidates_unchecked, finalize_prune_and_expand, rebuild_active_set,
};
use super::arena::TileArena;
#[cfg(target_arch = "x86_64")]
use super::kernel::advance_core;
#[cfg(target_arch = "aarch64")]
use super::kernel::advance_tile_fused_neon_assume_changed_no_track_fast;
use super::kernel::{
    KernelBackend, advance_tile_fused_scalar_no_track, advance_tile_fused_scalar_track,
};
#[cfg(target_arch = "x86_64")]
use super::kernel::{advance_tile_fused_avx2_no_track, advance_tile_fused_avx2_track};
#[cfg(target_arch = "aarch64")]
use super::kernel::{advance_tile_fused_neon_no_track_fast, advance_tile_fused_neon_track};
use super::tile::{self, Direction, NO_NEIGHBOR, POPULATION_UNKNOWN, TileIdx};
use super::tile_cache::{
    TileCache, advance_tile_cached_scalar_no_track, advance_tile_cached_scalar_track,
};
#[cfg(target_arch = "x86_64")]
use super::tile_cache::{advance_tile_cached_avx2_no_track, advance_tile_cached_avx2_track};

struct SendPtr<T> {
    inner: *mut T,
}
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> Copy for SendPtr<T> {}
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> SendPtr<T> {
    #[inline(always)]
    fn new(ptr: *mut T) -> Self {
        Self { inner: ptr }
    }
    #[inline(always)]
    fn get(&self) -> *mut T {
        self.inner
    }
}

struct SendConstPtr<T> {
    inner: *const T,
}
unsafe impl<T> Send for SendConstPtr<T> {}
unsafe impl<T> Sync for SendConstPtr<T> {}
impl<T> Copy for SendConstPtr<T> {}
impl<T> Clone for SendConstPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> SendConstPtr<T> {
    #[inline(always)]
    fn new(ptr: *const T) -> Self {
        Self { inner: ptr }
    }
    #[inline(always)]
    fn get(&self) -> *const T {
        self.inner
    }
}

const TILE_SIZE_I64: i64 = 64;
const PARALLEL_KERNEL_MIN_ACTIVE: usize = 128;
const PARALLEL_KERNEL_TILES_PER_THREAD: usize = 16;
const PARALLEL_KERNEL_MIN_CHUNKS: usize = 2;
const KERNEL_CHUNK_MIN: usize = 32;
const SERIAL_CACHE_MAX_ACTIVE: usize = 128;
// Tuned on the main.rs harness (Apple M4): switch to static slices earlier
// to cut dynamic scheduler overhead on medium-large frontiers.
const PARALLEL_STATIC_SCHEDULE_THRESHOLD: usize = 8_192;
// Dynamic scheduler chunking target per worker.
// Lower target (vs 4) reduces atomic cursor traffic while preserving balance.
const PARALLEL_DYNAMIC_TARGET_CHUNKS_PER_WORKER: usize = 2;
const PARALLEL_DYNAMIC_CHUNK_MIN: usize = 8;
const PARALLEL_DYNAMIC_CHUNK_MAX: usize = 2_048;
#[cfg(target_arch = "x86_64")]
const PREFETCH_NEIGHBOR_BORDERS_MIN_ACTIVE: usize = 1_024;
#[cfg(target_arch = "aarch64")]
const PREFETCH_NEIGHBOR_BORDERS_MIN_ACTIVE: usize = 32_768;
// AArch64 PRFM hints regressed end-to-end throughput on the main harness;
// keep them compiled but disabled by default for this target.
#[cfg(target_arch = "aarch64")]
#[cfg(feature = "aggressive-prefetch-aarch64")]
const PREFETCH_TILE_DATA_AARCH64: bool = true;
#[cfg(target_arch = "aarch64")]
#[cfg(not(feature = "aggressive-prefetch-aarch64"))]
const PREFETCH_TILE_DATA_AARCH64: bool = false;
// Tuned for Apple perf cores: i+3 gives L2 enough lookahead while i+1 keeps
// the immediate tile in L1 under heavy frontier churn.
#[cfg(target_arch = "aarch64")]
const PREFETCH_TILE_FAR_AHEAD_AARCH64: usize = 3;
#[cfg(target_arch = "aarch64")]
const PREFETCH_TILE_NEAR_AHEAD_AARCH64: usize = 1;
#[cfg(target_arch = "aarch64")]
const _: [(); 1] = [(); (PREFETCH_TILE_NEAR_AHEAD_AARCH64 >= 1) as usize];
#[cfg(target_arch = "aarch64")]
const _: [(); 1] =
    [(); (PREFETCH_TILE_FAR_AHEAD_AARCH64 > PREFETCH_TILE_NEAR_AHEAD_AARCH64) as usize];
#[cfg(target_arch = "aarch64")]
// Enable `assume_changed` once the active frontier is large enough to
// amortize the extra follow-up bookkeeping in the engine.
const ASSUME_CHANGED_NEON_MIN_ACTIVE: usize = 2_048;

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn prefetch_l1_read<T>(ptr: *const T) {
    unsafe {
        std::arch::asm!(
            "prfm pldl1keep, [{addr}]",
            addr = in(reg) ptr,
            options(nostack, readonly)
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn prefetch_l2_read<T>(ptr: *const T) {
    unsafe {
        std::arch::asm!(
            "prfm pldl2keep, [{addr}]",
            addr = in(reg) ptr,
            options(nostack, readonly)
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn prefetch_l1_write<T>(ptr: *const T) {
    unsafe {
        std::arch::asm!(
            "prfm pstl1keep, [{addr}]",
            addr = in(reg) ptr,
            options(nostack)
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn prefetch_l2_write<T>(ptr: *const T) {
    unsafe {
        std::arch::asm!(
            "prfm pstl2keep, [{addr}]",
            addr = in(reg) ptr,
            options(nostack)
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn prefetch_neighbor_border_lines(
    neighbors_ptr: *const [u32; 8],
    borders_north_read_ptr: *const u64,
    borders_south_read_ptr: *const u64,
    borders_west_read_ptr: *const u64,
    borders_east_read_ptr: *const u64,
    tile_index: usize,
) {
    use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

    let nb = unsafe { &*neighbors_ptr.add(tile_index) };
    unsafe {
        _mm_prefetch(
            borders_south_read_ptr.add(nb[Direction::North.index()] as usize) as *const i8,
            _MM_HINT_T0,
        );
        _mm_prefetch(
            borders_north_read_ptr.add(nb[Direction::South.index()] as usize) as *const i8,
            _MM_HINT_T0,
        );
        _mm_prefetch(
            borders_east_read_ptr.add(nb[Direction::West.index()] as usize) as *const i8,
            _MM_HINT_T0,
        );
        _mm_prefetch(
            borders_west_read_ptr.add(nb[Direction::East.index()] as usize) as *const i8,
            _MM_HINT_T0,
        );
        _mm_prefetch(
            borders_south_read_ptr.add(nb[Direction::NW.index()] as usize) as *const i8,
            _MM_HINT_T0,
        );
        _mm_prefetch(
            borders_south_read_ptr.add(nb[Direction::NE.index()] as usize) as *const i8,
            _MM_HINT_T0,
        );
        _mm_prefetch(
            borders_north_read_ptr.add(nb[Direction::SW.index()] as usize) as *const i8,
            _MM_HINT_T0,
        );
        _mm_prefetch(
            borders_north_read_ptr.add(nb[Direction::SE.index()] as usize) as *const i8,
            _MM_HINT_T0,
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn prefetch_neighbor_border_lines(
    neighbors_ptr: *const [u32; 8],
    borders_north_read_ptr: *const u64,
    borders_south_read_ptr: *const u64,
    borders_west_read_ptr: *const u64,
    borders_east_read_ptr: *const u64,
    tile_index: usize,
) {
    let nb = unsafe { &*neighbors_ptr.add(tile_index) };
    unsafe {
        prefetch_l1_read(borders_south_read_ptr.add(nb[Direction::North.index()] as usize));
        prefetch_l1_read(borders_north_read_ptr.add(nb[Direction::South.index()] as usize));
        prefetch_l1_read(borders_east_read_ptr.add(nb[Direction::West.index()] as usize));
        prefetch_l1_read(borders_west_read_ptr.add(nb[Direction::East.index()] as usize));
        prefetch_l1_read(borders_south_read_ptr.add(nb[Direction::NW.index()] as usize));
        prefetch_l1_read(borders_south_read_ptr.add(nb[Direction::NE.index()] as usize));
        prefetch_l1_read(borders_north_read_ptr.add(nb[Direction::SW.index()] as usize));
        prefetch_l1_read(borders_north_read_ptr.add(nb[Direction::SE.index()] as usize));
    }
}

static PHYSICAL_CORES: OnceLock<usize> = OnceLock::new();
static AUTO_KERNEL_BACKEND: OnceLock<KernelBackend> = OnceLock::new();

#[cfg(target_os = "macos")]
#[inline]
fn apple_perf_core_count() -> Option<usize> {
    use core::ffi::{c_char, c_int, c_void};

    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *const c_void,
            newlen: usize,
        ) -> c_int;
    }

    const PERF_CORE_SYSCTL: &[u8] = b"hw.perflevel0.physicalcpu\0";
    let mut value = 0u32;
    let mut len = std::mem::size_of_val(&value);
    let rc = unsafe {
        // SAFETY:
        // - `PERF_CORE_SYSCTL` is NUL-terminated and lives for the call.
        // - `value` and `len` are valid mutable out-pointers.
        // - `newp` is null with `newlen = 0`, so no write-input buffer is used.
        sysctlbyname(
            PERF_CORE_SYSCTL.as_ptr().cast(),
            (&mut value as *mut u32).cast(),
            &mut len,
            std::ptr::null(),
            0,
        )
    };
    if rc == 0 && len == std::mem::size_of_val(&value) && value > 0 {
        Some(value as usize)
    } else {
        None
    }
}

#[derive(Default)]
#[repr(align(64))]
struct WorkerScratch {
    changed: Vec<TileIdx>,
    changed_influence: Vec<u8>,
    expand: Vec<u32>,
    prune: Vec<TileIdx>,
}

impl WorkerScratch {
    #[inline]
    fn reserve_total_capacity<T>(buf: &mut Vec<T>, target: usize) {
        if buf.capacity() < target {
            buf.reserve(target.saturating_sub(buf.len()));
        }
    }

    #[inline(always)]
    fn reserve_additional_capacity<T>(buf: &mut Vec<T>, additional: usize) {
        if additional == 0 {
            return;
        }
        let spare = buf.capacity().saturating_sub(buf.len());
        if spare < additional {
            // `Vec::reserve` takes an amount *in addition to the current length*.
            // Requesting `additional` ensures `capacity >= len + additional`.
            buf.reserve(additional);
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.changed.clear();
        self.changed_influence.clear();
        self.expand.clear();
        self.prune.clear();
    }

    #[inline]
    fn reserve_for_chunk<const EMIT_CHANGED: bool, const TRACK_NEIGHBOR_INFLUENCE: bool>(
        &mut self,
        chunk_len: usize,
    ) {
        if EMIT_CHANGED {
            Self::reserve_total_capacity(&mut self.changed, chunk_len);
        }
        if EMIT_CHANGED && TRACK_NEIGHBOR_INFLUENCE {
            Self::reserve_total_capacity(&mut self.changed_influence, chunk_len);
        }
        Self::reserve_total_capacity(&mut self.prune, chunk_len);
        let expand_target = chunk_len.saturating_mul(8);
        Self::reserve_total_capacity(&mut self.expand, expand_target);
    }

    #[inline(always)]
    fn reserve_for_additional_work<
        const EMIT_CHANGED: bool,
        const TRACK_NEIGHBOR_INFLUENCE: bool,
    >(
        &mut self,
        work_items: usize,
    ) {
        if EMIT_CHANGED {
            Self::reserve_additional_capacity(&mut self.changed, work_items);
        }
        if EMIT_CHANGED && TRACK_NEIGHBOR_INFLUENCE {
            Self::reserve_additional_capacity(&mut self.changed_influence, work_items);
        }
        Self::reserve_additional_capacity(&mut self.prune, work_items);
        let expand_additional = work_items.saturating_mul(8);
        Self::reserve_additional_capacity(&mut self.expand, expand_additional);
    }
}

#[inline]
fn detect_kernel_backend() -> KernelBackend {
    let auto_enabled = std::env::var("TURBOLIFE_AUTO_KERNEL")
        .ok()
        .and_then(|v| {
            let v = v.trim();
            if v.is_empty() {
                None
            } else if v == "1" || v.eq_ignore_ascii_case("true") {
                Some(true)
            } else if v == "0" || v.eq_ignore_ascii_case("false") {
                Some(false)
            } else {
                None
            }
        })
        .unwrap_or(true);

    if auto_enabled {
        *AUTO_KERNEL_BACKEND.get_or_init(calibrate_auto_backend)
    } else {
        KernelBackend::Scalar
    }
}

#[inline]
fn neon_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn next_seed(seed: &mut u64) -> u64 {
    let mut x = *seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *seed = x;
    x
}

#[cfg(target_arch = "x86_64")]
fn benchmark_backend(backend: KernelBackend) -> u128 {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut current = [0u64; tile::TILE_SIZE];
    let mut next = [0u64; tile::TILE_SIZE];
    for row in &mut current {
        *row = next_seed(&mut seed);
    }

    let ghost = tile::GhostZone {
        north: next_seed(&mut seed),
        south: next_seed(&mut seed),
        west: next_seed(&mut seed),
        east: next_seed(&mut seed),
        nw: (next_seed(&mut seed) & 1) != 0,
        ne: (next_seed(&mut seed) & 1) != 0,
        sw: (next_seed(&mut seed) & 1) != 0,
        se: (next_seed(&mut seed) & 1) != 0,
    };

    let rounds = 1536usize;
    let start = Instant::now();
    let mut checksum = 0u64;
    for round in 0..rounds {
        let (changed, border, has_live) = advance_core(&current, &mut next, &ghost, backend);
        checksum ^= border.north
            ^ border.south
            ^ border.west
            ^ border.east
            ^ border.live_mask() as u64
            ^ changed as u64
            ^ has_live as u64;

        if round & 1 == 0 {
            current = next;
            next = [0u64; tile::TILE_SIZE];
        }
    }
    std::hint::black_box(checksum);
    start.elapsed().as_nanos()
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn benchmark_backend_best(backend: KernelBackend, trials: usize) -> u128 {
    let mut best = u128::MAX;
    for _ in 0..trials.max(1) {
        best = best.min(benchmark_backend(backend));
    }
    best
}

#[cfg(target_arch = "x86_64")]
fn calibrate_auto_backend() -> KernelBackend {
    if !std::is_x86_feature_detected!("avx2") {
        return KernelBackend::Scalar;
    }

    let scalar_ns = benchmark_backend_best(KernelBackend::Scalar, 3);
    let avx2_ns = benchmark_backend_best(KernelBackend::Avx2, 3);

    // Require a large win to avoid picking AVX2 on calibration noise.
    if avx2_ns.saturating_mul(100) <= scalar_ns.saturating_mul(85) {
        KernelBackend::Avx2
    } else {
        KernelBackend::Scalar
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn calibrate_auto_backend() -> KernelBackend {
    if neon_available() {
        KernelBackend::Neon
    } else {
        KernelBackend::Scalar
    }
}

#[inline]
fn physical_core_count() -> usize {
    *PHYSICAL_CORES.get_or_init(|| {
        let physical = num_cpus::get_physical().max(1);
        #[cfg(target_os = "macos")]
        {
            let perf_only = std::env::var("TURBOLIFE_MACOS_PERF_ONLY")
                .ok()
                .and_then(|v| {
                    let v = v.trim();
                    if v.is_empty() {
                        None
                    } else if v == "1" || v.eq_ignore_ascii_case("true") {
                        Some(true)
                    } else if v == "0" || v.eq_ignore_ascii_case("false") {
                        Some(false)
                    } else {
                        None
                    }
                })
                .unwrap_or(false);
            if perf_only && let Some(perf) = apple_perf_core_count() {
                return perf.min(physical).max(1);
            }
        }
        physical
    })
}

#[inline]
fn effective_parallel_threads(active_len: usize, thread_count: usize) -> usize {
    if thread_count <= 1 || active_len < PARALLEL_KERNEL_MIN_ACTIVE {
        return 1;
    }
    let by_work = active_len / PARALLEL_KERNEL_TILES_PER_THREAD;
    let by_chunks = active_len / KERNEL_CHUNK_MIN;
    let effective = by_work.min(by_chunks).min(thread_count).max(1);
    if effective < PARALLEL_KERNEL_MIN_CHUNKS {
        1
    } else {
        effective
    }
}

#[inline]
fn memory_parallel_cap(thread_count: usize) -> usize {
    if thread_count <= 8 {
        thread_count
    } else {
        thread_count.div_ceil(2).max(8)
    }
}

#[inline]
fn auto_pool_thread_count_for_physical(physical: usize) -> usize {
    let physical = physical.max(1);
    if physical <= 8 {
        physical
    } else {
        physical.div_ceil(2).max(6)
    }
}

#[inline]
fn auto_pool_thread_count() -> usize {
    auto_pool_thread_count_for_physical(physical_core_count())
}

#[inline]
fn tuned_parallel_threads(active_len: usize, thread_count: usize) -> usize {
    if thread_count <= 1 {
        return 1;
    }
    let mut effective = effective_parallel_threads(active_len, thread_count);
    if effective <= 1 {
        return 1;
    }
    let bw_cap = memory_parallel_cap(thread_count);
    let tuned_cap = if active_len < 256 {
        bw_cap.min(2)
    } else if active_len < 1_024 {
        bw_cap.min(4)
    } else if active_len < 8_192 {
        bw_cap.min(6)
    } else if active_len < 32_768 {
        bw_cap.min(8)
    } else {
        bw_cap
    };
    effective = effective.min(tuned_cap).min(thread_count);
    effective.max(1)
}

#[cfg(test)]
#[inline]
fn churn_at_most_percent(changed_len: usize, active_len: usize, max_churn_pct: usize) -> bool {
    (changed_len as u128) * 100 <= (active_len as u128) * (max_churn_pct as u128)
}

#[cfg(test)]
#[inline]
fn churn_below_percent(changed_len: usize, active_len: usize, churn_pct_threshold: usize) -> bool {
    (changed_len as u128) * 100 < (active_len as u128) * (churn_pct_threshold as u128)
}

#[inline]
fn dynamic_target_chunks_per_worker(_active_len: usize, _changed_len: usize) -> usize {
    PARALLEL_DYNAMIC_TARGET_CHUNKS_PER_WORKER
}

#[inline]
fn dynamic_parallel_chunk_size(
    active_len: usize,
    changed_len: usize,
    worker_count: usize,
) -> usize {
    let workers = worker_count.max(1);
    let target_chunks = workers
        .saturating_mul(dynamic_target_chunks_per_worker(active_len, changed_len))
        .max(workers);
    let size = active_len.div_ceil(target_chunks);
    size.clamp(PARALLEL_DYNAMIC_CHUNK_MIN, PARALLEL_DYNAMIC_CHUNK_MAX)
}

#[cfg(test)]
#[inline(always)]
fn active_broadcast_workers(worker_count: usize, broadcast_threads: usize) -> usize {
    worker_count.min(broadcast_threads)
}

#[inline(always)]
fn should_queue_prune<const EMIT_CHANGED: bool>(has_live: bool, changed: bool) -> bool {
    !has_live && (!EMIT_CHANGED || !changed)
}

#[inline(always)]
fn reserve_additional_capacity<T>(buf: &mut Vec<T>, additional: usize) {
    if additional == 0 {
        return;
    }
    let spare = buf.capacity().saturating_sub(buf.len());
    if spare < additional {
        // `Vec::reserve` takes an amount *in addition to the current length*.
        // Requesting `additional` ensures `capacity >= len + additional`.
        buf.reserve(additional);
    }
}

#[inline(always)]
unsafe fn vec_push_unchecked<T>(buf: &mut Vec<T>, value: T) {
    debug_assert!(buf.len() < buf.capacity());
    let len = buf.len();
    unsafe {
        std::ptr::write(buf.as_mut_ptr().add(len), value);
        buf.set_len(len + 1);
    }
}

/// Resolve the thread count from a config, falling back to auto-detect.
fn resolve_thread_count(config: &TurboLifeConfig) -> usize {
    let mut threads = config.thread_count.unwrap_or_else(auto_pool_thread_count);
    if let Some(cap) = config.max_threads {
        threads = threads.min(cap);
    }
    threads.max(1)
}

/// Resolve the kernel backend from a config, falling back to default policy.
fn resolve_kernel_backend(config: &TurboLifeConfig) -> KernelBackend {
    if let Some(backend) = config.kernel {
        // Validate AVX2 availability at runtime.
        if backend == KernelBackend::Avx2 {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx2") {
                    return KernelBackend::Avx2;
                }
            }
            return KernelBackend::Scalar;
        }
        if backend == KernelBackend::Neon {
            if neon_available() {
                return KernelBackend::Neon;
            }
            return KernelBackend::Scalar;
        }
        return backend;
    }
    detect_kernel_backend()
}

/// Configuration for a TurboLife engine instance.
///
/// Use `TurboLifeConfig::default()` for auto-tuned defaults, or customise
/// individual knobs via the builder methods.
#[derive(Clone, Debug, Default)]
pub struct TurboLifeConfig {
    /// Number of threads for the compute pool.
    /// `None` means auto-detect (physical cores, memory-bandwidth capped).
    pub thread_count: Option<usize>,
    /// Hard upper bound on threads regardless of auto-detection.
    /// `None` means no additional cap beyond `thread_count`.
    pub max_threads: Option<usize>,
    /// Kernel backend selection.
    /// `None` means auto-detect (calibrated AVX2-vs-scalar on x86_64,
    /// NEON on aarch64 when the CPU reports support).
    /// Set `TURBOLIFE_AUTO_KERNEL=0` to force scalar default policy.
    pub kernel: Option<KernelBackend>,
}

impl TurboLifeConfig {
    /// Set an explicit thread count for the compute pool.
    pub fn thread_count(mut self, n: usize) -> Self {
        self.thread_count = Some(n.max(1));
        self
    }

    /// Set a hard upper bound on threads.
    pub fn max_threads(mut self, n: usize) -> Self {
        self.max_threads = Some(n.max(1));
        self
    }

    /// Force a specific kernel backend.
    pub fn kernel(mut self, backend: KernelBackend) -> Self {
        self.kernel = Some(backend);
        self
    }
}

pub struct TurboLife {
    arena: TileArena,
    generation: u64,
    population_cache: Option<u64>,
    pool: rayon::ThreadPool,
    pool_threads: usize,
    backend: KernelBackend,
    /// Reusable per-tile marker for batch dedup in `set_cells`.
    /// Packed bitset indexed by `TileIdx.index()`, grown on demand.
    touched_bitmap: Vec<u64>,
    /// Direct-mapped tile result cache for kernel memoization.
    tile_cache: TileCache,
    /// Per-worker scratch buffers reused by parallel kernel schedulers.
    worker_scratch: Vec<WorkerScratch>,
    /// Reusable touched-tile list for `set_cells`.
    set_cells_scratch: Vec<TileIdx>,
}

impl Default for TurboLife {
    fn default() -> Self {
        Self::new()
    }
}

impl TurboLife {
    pub fn new() -> Self {
        Self::with_config(TurboLifeConfig::default())
    }

    /// Create a TurboLife engine with explicit configuration.
    pub fn with_config(config: TurboLifeConfig) -> Self {
        let configured_threads = resolve_thread_count(&config);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(configured_threads)
            .build()
            .expect("failed to build TurboLife rayon thread pool");
        let pool_threads = pool.current_num_threads().max(1);
        let backend = resolve_kernel_backend(&config);

        Self {
            arena: TileArena::new(),
            generation: 0,
            population_cache: Some(0),
            pool,
            pool_threads,
            backend,
            touched_bitmap: Vec::new(),
            tile_cache: TileCache::new(),
            worker_scratch: (0..pool_threads)
                .map(|_| WorkerScratch::default())
                .collect(),
            set_cells_scratch: Vec::new(),
        }
    }

    #[inline]
    fn prepare_worker_scratch(
        &mut self,
        worker_count: usize,
        active_len: usize,
        emit_changed: bool,
        track_neighbor_influence: bool,
    ) {
        if self.worker_scratch.len() < worker_count {
            self.worker_scratch
                .resize_with(worker_count, WorkerScratch::default);
        }
        if worker_count == 0 {
            return;
        }
        let chunk_len = active_len.div_ceil(worker_count);
        if emit_changed {
            if track_neighbor_influence {
                for scratch in self.worker_scratch.iter_mut().take(worker_count) {
                    scratch.clear();
                    scratch.reserve_for_chunk::<true, true>(chunk_len);
                }
            } else {
                for scratch in self.worker_scratch.iter_mut().take(worker_count) {
                    scratch.clear();
                    scratch.reserve_for_chunk::<true, false>(chunk_len);
                }
            }
        } else if track_neighbor_influence {
            for scratch in self.worker_scratch.iter_mut().take(worker_count) {
                scratch.clear();
                scratch.reserve_for_chunk::<false, true>(chunk_len);
            }
        } else {
            for scratch in self.worker_scratch.iter_mut().take(worker_count) {
                scratch.clear();
                scratch.reserve_for_chunk::<false, false>(chunk_len);
            }
        }
    }

    #[inline(always)]
    fn ensure_touched_capacity(&mut self, idx: usize) {
        let word = idx >> 6;
        if word < self.touched_bitmap.len() {
            return;
        }
        let target_words = self.arena.meta.len().div_ceil(64);
        self.touched_bitmap.resize(target_words.max(word + 1), 0);
    }

    #[inline(always)]
    fn touched_test_and_set(&mut self, idx: usize) -> bool {
        self.ensure_touched_capacity(idx);
        let word = idx >> 6;
        let bit = 1u64 << (idx & 63);
        let old = self.touched_bitmap[word];
        self.touched_bitmap[word] = old | bit;
        (old & bit) != 0
    }

    #[inline(always)]
    fn touched_clear(&mut self, idx: usize) {
        let word = idx >> 6;
        let bit = 1u64 << (idx & 63);
        self.touched_bitmap[word] &= !bit;
    }

    #[inline]
    fn derive_changed_from_active_minus_prune(&mut self) {
        self.arena.changed_influence.clear();
        std::mem::swap(&mut self.arena.changed_list, &mut self.arena.active_set);
        self.arena.active_set.clear();
        self.arena.active_set_dense_contiguous = false;
        if self.arena.changed_list.is_empty() || self.arena.prune_buf.is_empty() {
            return;
        }

        // Remove tiles queued for pruning from the synthesized changed list.
        // This avoids carrying released slots into the next active rebuild.
        for i in 0..self.arena.prune_buf.len() {
            let idx = self.arena.prune_buf[i];
            self.ensure_touched_capacity(idx.index());
            let word = idx.index() >> 6;
            let bit = 1u64 << (idx.index() & 63);
            self.touched_bitmap[word] |= bit;
        }

        let mut write = 0usize;
        for read in 0..self.arena.changed_list.len() {
            let idx = self.arena.changed_list[read];
            let ii = idx.index();
            let word = ii >> 6;
            let bit = 1u64 << (ii & 63);
            if word >= self.touched_bitmap.len() || (self.touched_bitmap[word] & bit) == 0 {
                self.arena.changed_list[write] = idx;
                write += 1;
            }
        }
        self.arena.changed_list.truncate(write);

        for i in 0..self.arena.prune_buf.len() {
            let idx = self.arena.prune_buf[i];
            self.touched_clear(idx.index());
        }
    }

    #[inline(always)]
    fn ensure_frontier_neighbors(
        &mut self,
        idx: TileIdx,
        tile_coord: (i64, i64),
        local_x: usize,
        local_y: usize,
    ) {
        let on_south = local_y == 0;
        let on_north = local_y == tile::TILE_SIZE - 1;
        let on_west = local_x == 0;
        let on_east = local_x == tile::TILE_SIZE - 1;

        if !(on_north || on_south || on_west || on_east) {
            return;
        }

        let (tx, ty) = tile_coord;
        let neighbors = self.arena.neighbors[idx.index()];

        if on_north && neighbors[Direction::North.index()] == NO_NEIGHBOR {
            self.arena.ensure_neighbor(tx, ty + 1);
        }
        if on_south && neighbors[Direction::South.index()] == NO_NEIGHBOR {
            self.arena.ensure_neighbor(tx, ty - 1);
        }
        if on_west && neighbors[Direction::West.index()] == NO_NEIGHBOR {
            self.arena.ensure_neighbor(tx - 1, ty);
        }
        if on_east && neighbors[Direction::East.index()] == NO_NEIGHBOR {
            self.arena.ensure_neighbor(tx + 1, ty);
        }
        if on_north && on_west && neighbors[Direction::NW.index()] == NO_NEIGHBOR {
            self.arena.ensure_neighbor(tx - 1, ty + 1);
        }
        if on_north && on_east && neighbors[Direction::NE.index()] == NO_NEIGHBOR {
            self.arena.ensure_neighbor(tx + 1, ty + 1);
        }
        if on_south && on_west && neighbors[Direction::SW.index()] == NO_NEIGHBOR {
            self.arena.ensure_neighbor(tx - 1, ty - 1);
        }
        if on_south && on_east && neighbors[Direction::SE.index()] == NO_NEIGHBOR {
            self.arena.ensure_neighbor(tx + 1, ty - 1);
        }
    }

    #[inline(always)]
    fn touches_tile_border(local_x: usize, local_y: usize) -> bool {
        local_x == 0
            || local_x == tile::TILE_SIZE - 1
            || local_y == 0
            || local_y == tile::TILE_SIZE - 1
    }

    #[inline(always)]
    fn update_current_border_for_row_mutation(
        &mut self,
        idx: TileIdx,
        local_x: usize,
        local_y: usize,
        row_after: u64,
    ) {
        if !Self::touches_tile_border(local_x, local_y) {
            return;
        }

        let mut border = self.arena.border(idx);

        if local_y == 0 {
            border.south = row_after;
        }
        if local_y == tile::TILE_SIZE - 1 {
            border.north = row_after;
        }

        if local_x == 0 {
            let row_bit = 1u64 << local_y;
            border.west = (border.west & !row_bit) | ((row_after & 1) << local_y);
        }
        if local_x == tile::TILE_SIZE - 1 {
            let row_bit = 1u64 << local_y;
            border.east =
                (border.east & !row_bit) | (((row_after >> (tile::TILE_SIZE - 1)) & 1) << local_y);
        }

        self.arena.set_current_border(idx, border);
    }

    #[inline]
    pub fn set_cell_alive(&mut self, x: i64, y: i64) {
        let tile_coord = (x.div_euclid(TILE_SIZE_I64), y.div_euclid(TILE_SIZE_I64));
        let local_x = x.rem_euclid(TILE_SIZE_I64) as usize;
        let local_y = y.rem_euclid(TILE_SIZE_I64) as usize;

        let idx = self
            .arena
            .idx_at_cached(tile_coord)
            .unwrap_or_else(|| self.arena.allocate_absent(tile_coord));

        let row_after: u64;
        {
            let buf = self.arena.current_buf_mut(idx);
            let mask = 1u64 << local_x;
            let row = &mut buf[local_y];
            if (*row & mask) != 0 {
                return;
            }
            *row |= mask;
            row_after = *row;
        }

        self.update_current_border_for_row_mutation(idx, local_x, local_y, row_after);

        let should_mark_changed;
        {
            let meta = self.arena.meta_mut(idx);
            let was_dirty = meta.alt_phase_dirty();
            meta.population = POPULATION_UNKNOWN;
            meta.set_has_live(true);
            meta.set_alt_phase_dirty(true);
            should_mark_changed = !was_dirty;
        }
        self.population_cache = None;
        if should_mark_changed {
            self.arena.mark_changed(idx);
        }
        self.ensure_frontier_neighbors(idx, tile_coord, local_x, local_y);
    }

    #[inline]
    pub fn set_cell(&mut self, x: i64, y: i64, alive: bool) {
        if alive {
            self.set_cell_alive(x, y);
            return;
        }

        let tile_coord = (x.div_euclid(TILE_SIZE_I64), y.div_euclid(TILE_SIZE_I64));
        let local_x = x.rem_euclid(TILE_SIZE_I64) as usize;
        let local_y = y.rem_euclid(TILE_SIZE_I64) as usize;

        let Some(idx) = self.arena.idx_at_cached(tile_coord) else {
            return;
        };

        let row_after: u64;
        let has_live_after_clear;

        {
            let buf = self.arena.current_buf_mut(idx);
            let mask = 1u64 << local_x;
            let row = &mut buf[local_y];
            if (*row & mask) == 0 {
                return;
            }
            *row &= !mask;
            row_after = *row;
            has_live_after_clear = if row_after != 0 {
                true
            } else {
                buf.iter().any(|&r| r != 0)
            };
        }

        self.update_current_border_for_row_mutation(idx, local_x, local_y, row_after);

        let should_mark_changed;
        {
            let meta = self.arena.meta_mut(idx);
            let was_dirty = meta.alt_phase_dirty();
            meta.population = POPULATION_UNKNOWN;
            meta.set_has_live(has_live_after_clear);
            meta.set_alt_phase_dirty(true);
            should_mark_changed = !was_dirty;
        }
        self.population_cache = None;
        if should_mark_changed {
            self.arena.mark_changed(idx);
        }
    }

    /// Batch-update many cells, amortizing per-tile metadata recompute.
    pub fn set_cells<I>(&mut self, cells: I)
    where
        I: IntoIterator<Item = (i64, i64, bool)>,
    {
        let mut touched_tiles = std::mem::take(&mut self.set_cells_scratch);
        touched_tiles.clear();

        for (x, y, alive) in cells {
            let tile_coord = (x.div_euclid(TILE_SIZE_I64), y.div_euclid(TILE_SIZE_I64));
            let local_x = x.rem_euclid(TILE_SIZE_I64) as usize;
            let local_y = y.rem_euclid(TILE_SIZE_I64) as usize;

            let idx = match self.arena.idx_at_cached(tile_coord) {
                Some(existing) => existing,
                None if alive => self.arena.allocate_absent(tile_coord),
                None => continue,
            };

            let changed = {
                let buf = self.arena.current_buf_mut(idx);
                let mask = 1u64 << local_x;
                let row = &mut buf[local_y];
                let was_alive = (*row & mask) != 0;
                if was_alive == alive {
                    false
                } else {
                    if alive {
                        *row |= mask;
                    } else {
                        *row &= !mask;
                    }
                    true
                }
            };

            if !changed {
                continue;
            }

            let i = idx.index();
            if !self.touched_test_and_set(i) {
                touched_tiles.push(idx);
                if !self.arena.meta(idx).alt_phase_dirty() {
                    self.arena.mark_changed(idx);
                }
            }

            if alive {
                self.ensure_frontier_neighbors(idx, tile_coord, local_x, local_y);
            }
        }

        if touched_tiles.is_empty() {
            self.set_cells_scratch = touched_tiles;
            return;
        }

        for idx in &touched_tiles {
            let buf = self.arena.current_buf(*idx);
            let (border, has_live) = tile::recompute_border_and_has_live(buf);

            self.arena.set_current_border(*idx, border);
            {
                let meta = self.arena.meta_mut(*idx);
                meta.population = POPULATION_UNKNOWN;
                meta.set_has_live(has_live);
                meta.set_alt_phase_dirty(true);
            }
        }

        // Clear touched flags for reuse (O(touched) not O(arena)).
        for idx in &touched_tiles {
            self.touched_clear(idx.index());
        }

        self.population_cache = None;
        touched_tiles.clear();
        self.set_cells_scratch = touched_tiles;
    }

    /// Batch-set many live cells.
    pub fn set_cells_alive<I>(&mut self, cells: I)
    where
        I: IntoIterator<Item = (i64, i64)>,
    {
        self.set_cells(cells.into_iter().map(|(x, y)| (x, y, true)));
    }

    pub fn get_cell(&self, x: i64, y: i64) -> bool {
        let tile_coord = (x.div_euclid(TILE_SIZE_I64), y.div_euclid(TILE_SIZE_I64));
        let local_x = x.rem_euclid(TILE_SIZE_I64) as usize;
        let local_y = y.rem_euclid(TILE_SIZE_I64) as usize;

        self.arena
            .idx_at(tile_coord)
            .map(|idx| {
                self.arena.meta(idx).occupied()
                    && tile::get_local_cell(self.arena.current_buf(idx), local_x, local_y)
            })
            .unwrap_or(false)
    }

    fn step_impl(&mut self) {
        rebuild_active_set(&mut self.arena);

        if self.arena.active_set.is_empty() {
            self.generation += 1;
            return;
        }

        let active_len = self.arena.active_set.len();
        let changed_len = self.arena.changed_scratch.len();
        self.arena.expand_buf.clear();
        self.arena.prune_buf.clear();
        let use_serial_cache =
            active_len <= SERIAL_CACHE_MAX_ACTIVE && self.backend != KernelBackend::Neon;
        if use_serial_cache {
            self.tile_cache.begin_step();
        }
        let bp = self.arena.border_phase;
        let cp = self.arena.cell_phase;
        let backend = self.backend;
        let thread_count = self.pool_threads;
        let effective_threads = tuned_parallel_threads(active_len, thread_count);
        let run_parallel = effective_threads > 1;
        // Throughput-first mode: skip per-tile directional influence tracking
        // and use uniform neighbor activation to reduce kernel-side write churn.
        let track_neighbor_influence = false;
        #[cfg(target_arch = "aarch64")]
        let assume_changed_neon = !track_neighbor_influence
            && backend == KernelBackend::Neon
            && active_len >= ASSUME_CHANGED_NEON_MIN_ACTIVE;
        #[cfg(target_arch = "aarch64")]
        let assume_changed_mode = assume_changed_neon;
        #[cfg(not(target_arch = "aarch64"))]
        let assume_changed_mode = false;
        let emit_changed = !assume_changed_mode;

        let (cb_lo, cb_hi) = self.arena.cell_bufs.split_at_mut(1);
        let (current_vec, next_vec) = if cp == 0 {
            (&cb_lo[0], &mut cb_hi[0])
        } else {
            (&cb_hi[0], &mut cb_lo[0])
        };
        let (bd_lo, bd_hi) = self.arena.borders.split_at_mut(1);
        let (borders_read, next_borders_vec) = if bp == 0 {
            (&bd_lo[0], &mut bd_hi[0])
        } else {
            (&bd_hi[0], &mut bd_lo[0])
        };
        let (lm_lo, lm_hi) = self.arena.border_live_masks.split_at_mut(1);
        let (live_masks_read, next_live_masks_vec) = if bp == 0 {
            (lm_lo[0].as_slice(), &mut lm_hi[0])
        } else {
            (lm_hi[0].as_slice(), &mut lm_lo[0])
        };

        debug_assert_eq!(self.arena.meta.len(), current_vec.len());
        debug_assert_eq!(self.arena.meta.len(), next_vec.len());
        debug_assert_eq!(self.arena.meta.len(), borders_read.len());
        debug_assert_eq!(self.arena.meta.len(), next_borders_vec.len());
        debug_assert_eq!(self.arena.meta.len(), live_masks_read.len());
        debug_assert_eq!(self.arena.meta.len(), next_live_masks_vec.len());
        debug_assert_eq!(self.arena.meta.len(), self.arena.neighbors.len());

        macro_rules! dispatch_by_emit {
            ($macro_name:ident, $advance:path, $track:expr) => {{
                if emit_changed {
                    $macro_name!($advance, $track, true);
                } else {
                    $macro_name!($advance, $track, false);
                }
            }};
        }

        if !run_parallel {
            reserve_additional_capacity(&mut self.arena.changed_list, active_len);
            if track_neighbor_influence {
                reserve_additional_capacity(&mut self.arena.changed_influence, active_len);
            }
            reserve_additional_capacity(&mut self.arena.prune_buf, active_len);
            reserve_additional_capacity(&mut self.arena.expand_buf, active_len.saturating_mul(8));

            // Serial path: cached kernel with multi-level prefetching.
            let neighbors_ptr = self.arena.neighbors.as_ptr();
            let borders_north_read_ptr = borders_read.north_ptr();
            let borders_south_read_ptr = borders_read.south_ptr();
            let borders_west_read_ptr = borders_read.west_ptr();
            let borders_east_read_ptr = borders_read.east_ptr();
            let live_masks_read_ptr = live_masks_read.as_ptr();
            let current_ptr = current_vec.as_ptr();
            let next_ptr = next_vec.as_mut_ptr();
            let next_borders_north_ptr = next_borders_vec.north_mut_ptr();
            let next_borders_south_ptr = next_borders_vec.south_mut_ptr();
            let next_borders_west_ptr = next_borders_vec.west_mut_ptr();
            let next_borders_east_ptr = next_borders_vec.east_mut_ptr();
            let next_live_masks_ptr = next_live_masks_vec.as_mut_ptr();
            let meta_ptr = self.arena.meta.as_mut_ptr();
            let active_ptr = self.arena.active_set.as_ptr();
            let cache_ptr = &mut self.tile_cache as *mut TileCache;
            #[cfg(target_arch = "x86_64")]
            let prefetch_neighbor_borders = active_len >= PREFETCH_NEIGHBOR_BORDERS_MIN_ACTIVE;
            #[cfg(target_arch = "aarch64")]
            let prefetch_neighbor_borders = active_len >= PREFETCH_NEIGHBOR_BORDERS_MIN_ACTIVE;

            macro_rules! serial_loop_cached {
                ($advance:path, $track:expr, $emit_changed:literal) => {{
                    for i in 0..active_len {
                        let idx = unsafe { *active_ptr.add(i) };
                        let ii = idx.index();

                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            use std::arch::x86_64::*;

                            // Prefetch tile i+2 cell buffers into L2.
                            if i + 2 < active_len {
                                let far_ii = (*active_ptr.add(i + 2)).index();
                                _mm_prefetch(current_ptr.add(far_ii) as *const i8, _MM_HINT_T1);
                                _mm_prefetch(next_ptr.add(far_ii) as *const i8, _MM_HINT_T1);
                            }
                            // Prefetch tile i+1: cell buffers into L1, plus neighbor borders
                            // for ghost zone gather.
                            if i + 1 < active_len {
                                let near_ii = (*active_ptr.add(i + 1)).index();
                                _mm_prefetch(current_ptr.add(near_ii) as *const i8, _MM_HINT_T0);
                                _mm_prefetch(next_ptr.add(near_ii) as *const i8, _MM_HINT_T0);
                                _mm_prefetch(neighbors_ptr.add(near_ii) as *const i8, _MM_HINT_T0);
                                if prefetch_neighbor_borders {
                                    prefetch_neighbor_border_lines(
                                        neighbors_ptr,
                                        borders_north_read_ptr,
                                        borders_south_read_ptr,
                                        borders_west_read_ptr,
                                        borders_east_read_ptr,
                                        near_ii,
                                    );
                                }
                            }
                        }

                        #[cfg(target_arch = "aarch64")]
                        unsafe {
                            if PREFETCH_TILE_DATA_AARCH64 {
                                // Prefetch a farther-ahead tile's cell buffers into L2.
                                if i + PREFETCH_TILE_FAR_AHEAD_AARCH64 < active_len {
                                    let far_ii = (*active_ptr
                                        .add(i + PREFETCH_TILE_FAR_AHEAD_AARCH64))
                                    .index();
                                    prefetch_l2_read(current_ptr.add(far_ii));
                                    prefetch_l2_write(next_ptr.add(far_ii));
                                }
                                // Prefetch a near-ahead tile into L1 plus neighbor metadata.
                                if i + PREFETCH_TILE_NEAR_AHEAD_AARCH64 < active_len {
                                    let near_ii = (*active_ptr
                                        .add(i + PREFETCH_TILE_NEAR_AHEAD_AARCH64))
                                    .index();
                                    prefetch_l1_read(current_ptr.add(near_ii));
                                    prefetch_l1_write(next_ptr.add(near_ii));
                                    prefetch_l1_read(neighbors_ptr.add(near_ii));
                                    if prefetch_neighbor_borders {
                                        prefetch_neighbor_border_lines(
                                            neighbors_ptr,
                                            borders_north_read_ptr,
                                            borders_south_read_ptr,
                                            borders_west_read_ptr,
                                            borders_east_read_ptr,
                                            near_ii,
                                        );
                                    }
                                }
                            }
                        }

                        let result = unsafe {
                            $advance(
                                current_ptr,
                                next_ptr,
                                meta_ptr,
                                next_borders_north_ptr,
                                next_borders_south_ptr,
                                next_borders_west_ptr,
                                next_borders_east_ptr,
                                borders_north_read_ptr,
                                borders_south_read_ptr,
                                borders_west_read_ptr,
                                borders_east_read_ptr,
                                neighbors_ptr,
                                live_masks_read_ptr,
                                next_live_masks_ptr,
                                ii,
                                cache_ptr,
                            )
                        };

                        let changed = result.changed;
                        let has_live = result.has_live;
                        if changed && $emit_changed {
                            unsafe {
                                vec_push_unchecked(&mut self.arena.changed_list, idx);
                            }
                            if $track {
                                unsafe {
                                    vec_push_unchecked(
                                        &mut self.arena.changed_influence,
                                        result.neighbor_influence_mask,
                                    );
                                }
                            }
                        }
                        if should_queue_prune::<$emit_changed>(has_live, changed) {
                            unsafe {
                                vec_push_unchecked(&mut self.arena.prune_buf, idx);
                            }
                        }

                        let missing = result.missing_mask;
                        if missing != 0 {
                            unsafe {
                                append_expand_candidates_unchecked(
                                    &mut self.arena.expand_buf,
                                    idx,
                                    missing,
                                    result.live_mask,
                                );
                            }
                        }
                    }
                }};
            }

            macro_rules! serial_loop_fused {
                ($advance:path, $track:expr, $emit_changed:literal) => {{
                    for i in 0..active_len {
                        let idx = unsafe { *active_ptr.add(i) };
                        let ii = idx.index();

                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            use std::arch::x86_64::*;

                            // Prefetch tile i+2 cell buffers into L2.
                            if i + 2 < active_len {
                                let far_ii = (*active_ptr.add(i + 2)).index();
                                _mm_prefetch(current_ptr.add(far_ii) as *const i8, _MM_HINT_T1);
                                _mm_prefetch(next_ptr.add(far_ii) as *const i8, _MM_HINT_T1);
                            }
                            // Prefetch tile i+1: cell buffers into L1, plus neighbor borders
                            // for ghost zone gather.
                            if i + 1 < active_len {
                                let near_ii = (*active_ptr.add(i + 1)).index();
                                _mm_prefetch(current_ptr.add(near_ii) as *const i8, _MM_HINT_T0);
                                _mm_prefetch(next_ptr.add(near_ii) as *const i8, _MM_HINT_T0);
                                _mm_prefetch(neighbors_ptr.add(near_ii) as *const i8, _MM_HINT_T0);
                                if prefetch_neighbor_borders {
                                    prefetch_neighbor_border_lines(
                                        neighbors_ptr,
                                        borders_north_read_ptr,
                                        borders_south_read_ptr,
                                        borders_west_read_ptr,
                                        borders_east_read_ptr,
                                        near_ii,
                                    );
                                }
                            }
                        }

                        #[cfg(target_arch = "aarch64")]
                        unsafe {
                            if PREFETCH_TILE_DATA_AARCH64 {
                                // Prefetch a farther-ahead tile's cell buffers into L2.
                                if i + PREFETCH_TILE_FAR_AHEAD_AARCH64 < active_len {
                                    let far_ii = (*active_ptr
                                        .add(i + PREFETCH_TILE_FAR_AHEAD_AARCH64))
                                    .index();
                                    prefetch_l2_read(current_ptr.add(far_ii));
                                    prefetch_l2_write(next_ptr.add(far_ii));
                                }
                                // Prefetch a near-ahead tile into L1 plus neighbor metadata.
                                if i + PREFETCH_TILE_NEAR_AHEAD_AARCH64 < active_len {
                                    let near_ii = (*active_ptr
                                        .add(i + PREFETCH_TILE_NEAR_AHEAD_AARCH64))
                                    .index();
                                    prefetch_l1_read(current_ptr.add(near_ii));
                                    prefetch_l1_write(next_ptr.add(near_ii));
                                    prefetch_l1_read(neighbors_ptr.add(near_ii));
                                    if prefetch_neighbor_borders {
                                        prefetch_neighbor_border_lines(
                                            neighbors_ptr,
                                            borders_north_read_ptr,
                                            borders_south_read_ptr,
                                            borders_west_read_ptr,
                                            borders_east_read_ptr,
                                            near_ii,
                                        );
                                    }
                                }
                            }
                        }

                        let result = unsafe {
                            $advance(
                                current_ptr,
                                next_ptr,
                                meta_ptr,
                                next_borders_north_ptr,
                                next_borders_south_ptr,
                                next_borders_west_ptr,
                                next_borders_east_ptr,
                                borders_north_read_ptr,
                                borders_south_read_ptr,
                                borders_west_read_ptr,
                                borders_east_read_ptr,
                                neighbors_ptr,
                                live_masks_read_ptr,
                                next_live_masks_ptr,
                                ii,
                            )
                        };

                        let changed = result.changed;
                        let has_live = result.has_live;
                        if changed && $emit_changed {
                            unsafe {
                                vec_push_unchecked(&mut self.arena.changed_list, idx);
                            }
                            if $track {
                                unsafe {
                                    vec_push_unchecked(
                                        &mut self.arena.changed_influence,
                                        result.neighbor_influence_mask,
                                    );
                                }
                            }
                        }
                        if should_queue_prune::<$emit_changed>(has_live, changed) {
                            unsafe {
                                vec_push_unchecked(&mut self.arena.prune_buf, idx);
                            }
                        }

                        let missing = result.missing_mask;
                        if missing != 0 {
                            unsafe {
                                append_expand_candidates_unchecked(
                                    &mut self.arena.expand_buf,
                                    idx,
                                    missing,
                                    result.live_mask,
                                );
                            }
                        }
                    }
                }};
            }

            if use_serial_cache {
                if track_neighbor_influence {
                    match backend {
                        KernelBackend::Scalar => {
                            dispatch_by_emit!(
                                serial_loop_cached,
                                advance_tile_cached_scalar_track,
                                true
                            )
                        }
                        KernelBackend::Avx2 => {
                            #[cfg(target_arch = "x86_64")]
                            {
                                dispatch_by_emit!(
                                    serial_loop_cached,
                                    advance_tile_cached_avx2_track,
                                    true
                                );
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            {
                                dispatch_by_emit!(
                                    serial_loop_cached,
                                    advance_tile_cached_scalar_track,
                                    true
                                );
                            }
                        }
                        KernelBackend::Neon => {
                            dispatch_by_emit!(
                                serial_loop_cached,
                                advance_tile_cached_scalar_track,
                                true
                            );
                        }
                    }
                } else {
                    match backend {
                        KernelBackend::Scalar => {
                            dispatch_by_emit!(
                                serial_loop_cached,
                                advance_tile_cached_scalar_no_track,
                                false
                            )
                        }
                        KernelBackend::Avx2 => {
                            #[cfg(target_arch = "x86_64")]
                            {
                                dispatch_by_emit!(
                                    serial_loop_cached,
                                    advance_tile_cached_avx2_no_track,
                                    false
                                );
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            {
                                dispatch_by_emit!(
                                    serial_loop_cached,
                                    advance_tile_cached_scalar_no_track,
                                    false
                                );
                            }
                        }
                        KernelBackend::Neon => {
                            dispatch_by_emit!(
                                serial_loop_cached,
                                advance_tile_cached_scalar_no_track,
                                false
                            );
                        }
                    }
                }
            } else if track_neighbor_influence {
                match backend {
                    KernelBackend::Scalar => {
                        dispatch_by_emit!(serial_loop_fused, advance_tile_fused_scalar_track, true)
                    }
                    KernelBackend::Avx2 => {
                        #[cfg(target_arch = "x86_64")]
                        {
                            dispatch_by_emit!(
                                serial_loop_fused,
                                advance_tile_fused_avx2_track,
                                true
                            );
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            dispatch_by_emit!(
                                serial_loop_fused,
                                advance_tile_fused_scalar_track,
                                true
                            );
                        }
                    }
                    KernelBackend::Neon => {
                        #[cfg(target_arch = "aarch64")]
                        {
                            dispatch_by_emit!(
                                serial_loop_fused,
                                advance_tile_fused_neon_track,
                                true
                            );
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        {
                            dispatch_by_emit!(
                                serial_loop_fused,
                                advance_tile_fused_scalar_track,
                                true
                            );
                        }
                    }
                }
            } else {
                match backend {
                    KernelBackend::Scalar => {
                        dispatch_by_emit!(
                            serial_loop_fused,
                            advance_tile_fused_scalar_no_track,
                            false
                        )
                    }
                    KernelBackend::Avx2 => {
                        #[cfg(target_arch = "x86_64")]
                        {
                            dispatch_by_emit!(
                                serial_loop_fused,
                                advance_tile_fused_avx2_no_track,
                                false
                            );
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            dispatch_by_emit!(
                                serial_loop_fused,
                                advance_tile_fused_scalar_no_track,
                                false
                            );
                        }
                    }
                    KernelBackend::Neon => {
                        #[cfg(target_arch = "aarch64")]
                        {
                            if assume_changed_neon {
                                dispatch_by_emit!(
                                    serial_loop_fused,
                                    advance_tile_fused_neon_assume_changed_no_track_fast,
                                    false
                                );
                            } else {
                                dispatch_by_emit!(
                                    serial_loop_fused,
                                    advance_tile_fused_neon_no_track_fast,
                                    false
                                );
                            }
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        {
                            dispatch_by_emit!(
                                serial_loop_fused,
                                advance_tile_fused_scalar_no_track,
                                false
                            );
                        }
                    }
                }
            }
        } else {
            // Parallel kernel compute with a hybrid scheduler:
            // - static contiguous slices for very large frontiers
            // - lock-free dynamic chunking otherwise for better load balance
            // Both reuse per-worker scratch buffers to avoid collect/reduce churn.
            let active_set = &self.arena.active_set;
            let active_ptr = SendConstPtr::new(active_set.as_ptr());
            let current_ptr = SendConstPtr::new(current_vec.as_ptr());
            let next_ptr = SendPtr::new(next_vec.as_mut_ptr());
            let meta_ptr = SendPtr::new(self.arena.meta.as_mut_ptr());
            let neighbors_ptr = SendConstPtr::new(self.arena.neighbors.as_ptr());
            let next_borders_north_ptr = SendPtr::new(next_borders_vec.north_mut_ptr());
            let next_borders_south_ptr = SendPtr::new(next_borders_vec.south_mut_ptr());
            let next_borders_west_ptr = SendPtr::new(next_borders_vec.west_mut_ptr());
            let next_borders_east_ptr = SendPtr::new(next_borders_vec.east_mut_ptr());
            let borders_north_read_ptr = SendConstPtr::new(borders_read.north_ptr());
            let borders_south_read_ptr = SendConstPtr::new(borders_read.south_ptr());
            let borders_west_read_ptr = SendConstPtr::new(borders_read.west_ptr());
            let borders_east_read_ptr = SendConstPtr::new(borders_read.east_ptr());
            let live_masks_read_ptr = SendConstPtr::new(live_masks_read.as_ptr());
            let next_live_masks_ptr = SendPtr::new(next_live_masks_vec.as_mut_ptr());
            let worker_count = effective_threads.min(active_len);
            let active_workers = worker_count;
            let static_chunk_size = active_len.div_ceil(active_workers);
            // Keep dynamic scheduling whenever the frontier is not huge; it handles
            // heterogeneous cores much better than fixed contiguous slices.
            let use_static_schedule = active_len >= PARALLEL_STATIC_SCHEDULE_THRESHOLD;
            self.prepare_worker_scratch(
                worker_count,
                active_len,
                emit_changed,
                track_neighbor_influence,
            );
            let scratch_ptr = SendPtr::new(self.worker_scratch.as_mut_ptr());
            #[cfg(target_arch = "x86_64")]
            let prefetch_neighbor_borders = active_len >= PREFETCH_NEIGHBOR_BORDERS_MIN_ACTIVE;
            #[cfg(target_arch = "aarch64")]
            let prefetch_neighbor_borders = active_len >= PREFETCH_NEIGHBOR_BORDERS_MIN_ACTIVE;

            macro_rules! prefetch_for_work_item {
                ($i:expr, $end:expr) => {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        use std::arch::x86_64::*;

                        if $i + 2 < $end {
                            let far_ii = (*active_ptr.get().add($i + 2)).index();
                            _mm_prefetch(current_ptr.get().add(far_ii) as *const i8, _MM_HINT_T1);
                            _mm_prefetch(next_ptr.get().add(far_ii) as *const i8, _MM_HINT_T1);
                        }
                        if $i + 1 < $end {
                            let near_ii = (*active_ptr.get().add($i + 1)).index();
                            _mm_prefetch(current_ptr.get().add(near_ii) as *const i8, _MM_HINT_T0);
                            _mm_prefetch(next_ptr.get().add(near_ii) as *const i8, _MM_HINT_T0);
                            _mm_prefetch(
                                neighbors_ptr.get().add(near_ii) as *const i8,
                                _MM_HINT_T0,
                            );
                            if prefetch_neighbor_borders {
                                prefetch_neighbor_border_lines(
                                    neighbors_ptr.get(),
                                    borders_north_read_ptr.get(),
                                    borders_south_read_ptr.get(),
                                    borders_west_read_ptr.get(),
                                    borders_east_read_ptr.get(),
                                    near_ii,
                                );
                            }
                        }
                    }

                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        if PREFETCH_TILE_DATA_AARCH64 {
                            if $i + PREFETCH_TILE_FAR_AHEAD_AARCH64 < $end {
                                let far_ii =
                                    (*active_ptr.get().add($i + PREFETCH_TILE_FAR_AHEAD_AARCH64))
                                        .index();
                                prefetch_l2_read(current_ptr.get().add(far_ii));
                                prefetch_l2_write(next_ptr.get().add(far_ii));
                            }
                            if $i + PREFETCH_TILE_NEAR_AHEAD_AARCH64 < $end {
                                let near_ii =
                                    (*active_ptr.get().add($i + PREFETCH_TILE_NEAR_AHEAD_AARCH64))
                                        .index();
                                prefetch_l1_read(current_ptr.get().add(near_ii));
                                prefetch_l1_write(next_ptr.get().add(near_ii));
                                prefetch_l1_read(neighbors_ptr.get().add(near_ii));
                                if prefetch_neighbor_borders {
                                    prefetch_neighbor_border_lines(
                                        neighbors_ptr.get(),
                                        borders_north_read_ptr.get(),
                                        borders_south_read_ptr.get(),
                                        borders_west_read_ptr.get(),
                                        borders_east_read_ptr.get(),
                                        near_ii,
                                    );
                                }
                            }
                        }
                    }
                };
            }

            macro_rules! process_work_item {
                (
                    $advance:path,
                    $scratch:expr,
                    $i:expr,
                    $end:expr,
                    $track:expr,
                    $emit_changed:literal
                ) => {{
                    prefetch_for_work_item!($i, $end);
                    let idx = unsafe { *active_ptr.get().add($i) };
                    let ii = idx.index();
                    let result = unsafe {
                        $advance(
                            current_ptr.get(),
                            next_ptr.get(),
                            meta_ptr.get(),
                            next_borders_north_ptr.get(),
                            next_borders_south_ptr.get(),
                            next_borders_west_ptr.get(),
                            next_borders_east_ptr.get(),
                            borders_north_read_ptr.get(),
                            borders_south_read_ptr.get(),
                            borders_west_read_ptr.get(),
                            borders_east_read_ptr.get(),
                            neighbors_ptr.get(),
                            live_masks_read_ptr.get(),
                            next_live_masks_ptr.get(),
                            ii,
                        )
                    };

                    let changed = result.changed;
                    let has_live = result.has_live;
                    if changed && $emit_changed {
                        unsafe {
                            vec_push_unchecked(&mut ($scratch).changed, idx);
                        }
                        if $track {
                            unsafe {
                                vec_push_unchecked(
                                    &mut ($scratch).changed_influence,
                                    result.neighbor_influence_mask,
                                );
                            }
                        }
                    }
                    if should_queue_prune::<$emit_changed>(has_live, changed) {
                        unsafe {
                            vec_push_unchecked(&mut ($scratch).prune, idx);
                        }
                    }

                    let missing = result.missing_mask;
                    if missing != 0 {
                        unsafe {
                            append_expand_candidates_unchecked(
                                &mut ($scratch).expand,
                                idx,
                                missing,
                                result.live_mask,
                            );
                        }
                    }
                }};
            }

            macro_rules! parallel_kernel_static {
                ($advance:path, $track:expr, $emit_changed:literal) => {{
                    let _ = rayon::broadcast(|ctx| {
                        let worker_id = ctx.index();
                        if worker_id >= active_workers {
                            return;
                        }
                        let start = worker_id.saturating_mul(static_chunk_size);
                        if start >= active_len {
                            return;
                        }
                        let end = start.saturating_add(static_chunk_size).min(active_len);
                        let scratch = unsafe { &mut *scratch_ptr.get().add(worker_id) };
                        scratch.reserve_for_additional_work::<$emit_changed, $track>(end - start);
                        for i in start..end {
                            process_work_item!($advance, scratch, i, end, $track, $emit_changed);
                        }
                    });
                }};
            }

            if use_static_schedule {
                if track_neighbor_influence {
                    match backend {
                        KernelBackend::Scalar => {
                            dispatch_by_emit!(
                                parallel_kernel_static,
                                advance_tile_fused_scalar_track,
                                true
                            )
                        }
                        KernelBackend::Avx2 => {
                            #[cfg(target_arch = "x86_64")]
                            {
                                dispatch_by_emit!(
                                    parallel_kernel_static,
                                    advance_tile_fused_avx2_track,
                                    true
                                )
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            {
                                dispatch_by_emit!(
                                    parallel_kernel_static,
                                    advance_tile_fused_scalar_track,
                                    true
                                )
                            }
                        }
                        KernelBackend::Neon => {
                            #[cfg(target_arch = "aarch64")]
                            {
                                dispatch_by_emit!(
                                    parallel_kernel_static,
                                    advance_tile_fused_neon_track,
                                    true
                                )
                            }
                            #[cfg(not(target_arch = "aarch64"))]
                            {
                                dispatch_by_emit!(
                                    parallel_kernel_static,
                                    advance_tile_fused_scalar_track,
                                    true
                                )
                            }
                        }
                    }
                } else {
                    match backend {
                        KernelBackend::Scalar => {
                            dispatch_by_emit!(
                                parallel_kernel_static,
                                advance_tile_fused_scalar_no_track,
                                false
                            )
                        }
                        KernelBackend::Avx2 => {
                            #[cfg(target_arch = "x86_64")]
                            {
                                dispatch_by_emit!(
                                    parallel_kernel_static,
                                    advance_tile_fused_avx2_no_track,
                                    false
                                )
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            {
                                dispatch_by_emit!(
                                    parallel_kernel_static,
                                    advance_tile_fused_scalar_no_track,
                                    false
                                )
                            }
                        }
                        KernelBackend::Neon => {
                            #[cfg(target_arch = "aarch64")]
                            {
                                if assume_changed_neon {
                                    dispatch_by_emit!(
                                        parallel_kernel_static,
                                        advance_tile_fused_neon_assume_changed_no_track_fast,
                                        false
                                    )
                                } else {
                                    dispatch_by_emit!(
                                        parallel_kernel_static,
                                        advance_tile_fused_neon_no_track_fast,
                                        false
                                    )
                                }
                            }
                            #[cfg(not(target_arch = "aarch64"))]
                            {
                                dispatch_by_emit!(
                                    parallel_kernel_static,
                                    advance_tile_fused_scalar_no_track,
                                    false
                                )
                            }
                        }
                    }
                }
            } else {
                let cursor = AtomicUsize::new(0);

                macro_rules! parallel_kernel_lockfree {
                    ($advance:path, $track:expr, $emit_changed:literal) => {{
                        let cursor_ref = &cursor;
                        let _ = rayon::broadcast(|ctx| {
                            let worker_id = ctx.index();
                            if worker_id >= active_workers {
                                return;
                            }
                            let scratch = unsafe { &mut *scratch_ptr.get().add(worker_id) };
                            loop {
                                let observed = cursor_ref.load(Ordering::Relaxed);
                                if observed >= active_len {
                                    break;
                                }
                                let chunk_size = dynamic_parallel_chunk_size(
                                    active_len - observed,
                                    changed_len,
                                    active_workers,
                                );
                                let start = cursor_ref.fetch_add(chunk_size, Ordering::Relaxed);
                                if start >= active_len {
                                    break;
                                }
                                let end = start.saturating_add(chunk_size).min(active_len);
                                scratch.reserve_for_additional_work::<$emit_changed, $track>(
                                    end - start,
                                );
                                for i in start..end {
                                    process_work_item!(
                                        $advance,
                                        scratch,
                                        i,
                                        end,
                                        $track,
                                        $emit_changed
                                    );
                                }
                            }
                        });
                    }};
                }

                if track_neighbor_influence {
                    match backend {
                        KernelBackend::Scalar => {
                            dispatch_by_emit!(
                                parallel_kernel_lockfree,
                                advance_tile_fused_scalar_track,
                                true
                            )
                        }
                        KernelBackend::Avx2 => {
                            #[cfg(target_arch = "x86_64")]
                            {
                                dispatch_by_emit!(
                                    parallel_kernel_lockfree,
                                    advance_tile_fused_avx2_track,
                                    true
                                )
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            {
                                dispatch_by_emit!(
                                    parallel_kernel_lockfree,
                                    advance_tile_fused_scalar_track,
                                    true
                                )
                            }
                        }
                        KernelBackend::Neon => {
                            #[cfg(target_arch = "aarch64")]
                            {
                                dispatch_by_emit!(
                                    parallel_kernel_lockfree,
                                    advance_tile_fused_neon_track,
                                    true
                                )
                            }
                            #[cfg(not(target_arch = "aarch64"))]
                            {
                                dispatch_by_emit!(
                                    parallel_kernel_lockfree,
                                    advance_tile_fused_scalar_track,
                                    true
                                )
                            }
                        }
                    }
                } else {
                    match backend {
                        KernelBackend::Scalar => {
                            dispatch_by_emit!(
                                parallel_kernel_lockfree,
                                advance_tile_fused_scalar_no_track,
                                false
                            )
                        }
                        KernelBackend::Avx2 => {
                            #[cfg(target_arch = "x86_64")]
                            {
                                dispatch_by_emit!(
                                    parallel_kernel_lockfree,
                                    advance_tile_fused_avx2_no_track,
                                    false
                                )
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            {
                                dispatch_by_emit!(
                                    parallel_kernel_lockfree,
                                    advance_tile_fused_scalar_no_track,
                                    false
                                )
                            }
                        }
                        KernelBackend::Neon => {
                            #[cfg(target_arch = "aarch64")]
                            {
                                if assume_changed_neon {
                                    dispatch_by_emit!(
                                        parallel_kernel_lockfree,
                                        advance_tile_fused_neon_assume_changed_no_track_fast,
                                        false
                                    )
                                } else {
                                    dispatch_by_emit!(
                                        parallel_kernel_lockfree,
                                        advance_tile_fused_neon_no_track_fast,
                                        false
                                    )
                                }
                            }
                            #[cfg(not(target_arch = "aarch64"))]
                            {
                                dispatch_by_emit!(
                                    parallel_kernel_lockfree,
                                    advance_tile_fused_scalar_no_track,
                                    false
                                )
                            }
                        }
                    }
                }
            }

            let mut total_expand = 0usize;
            let mut total_prune = 0usize;
            let mut total_changed = 0usize;
            let mut max_expand = (0usize, 0usize);
            let mut max_prune = (0usize, 0usize);
            let mut max_changed = (0usize, 0usize);
            for worker_id in 0..worker_count {
                let scratch = &self.worker_scratch[worker_id];
                let expand_len = scratch.expand.len();
                let prune_len = scratch.prune.len();
                let changed_len = scratch.changed.len();

                total_expand = total_expand.saturating_add(expand_len);
                total_prune = total_prune.saturating_add(prune_len);
                total_changed = total_changed.saturating_add(changed_len);

                if expand_len > max_expand.0 {
                    max_expand = (expand_len, worker_id);
                }
                if prune_len > max_prune.0 {
                    max_prune = (prune_len, worker_id);
                }
                if changed_len > max_changed.0 {
                    max_changed = (changed_len, worker_id);
                }

                if track_neighbor_influence {
                    debug_assert_eq!(changed_len, scratch.changed_influence.len());
                }
            }

            macro_rules! merge_worker_vectors {
                ($dst:expr, $field:ident, $total:expr, $max_pair:expr) => {{
                    if $total == 0 {
                        $dst.clear();
                    } else {
                        let base_worker = $max_pair.1;
                        std::mem::swap(&mut $dst, &mut self.worker_scratch[base_worker].$field);
                        let additional = $total.saturating_sub($dst.len());
                        reserve_additional_capacity(&mut $dst, additional);
                        for worker_id in 0..worker_count {
                            if worker_id == base_worker {
                                continue;
                            }
                            $dst.append(&mut self.worker_scratch[worker_id].$field);
                        }
                    }
                }};
            }

            merge_worker_vectors!(self.arena.expand_buf, expand, total_expand, max_expand);
            merge_worker_vectors!(self.arena.prune_buf, prune, total_prune, max_prune);
            if emit_changed {
                merge_worker_vectors!(self.arena.changed_list, changed, total_changed, max_changed);
                if track_neighbor_influence {
                    merge_worker_vectors!(
                        self.arena.changed_influence,
                        changed_influence,
                        total_changed,
                        max_changed
                    );
                } else {
                    self.arena.changed_influence.clear();
                }
            } else {
                self.arena.changed_list.clear();
                self.arena.changed_influence.clear();
            }
        }
        if assume_changed_mode {
            self.derive_changed_from_active_minus_prune();
        }
        if !self.arena.changed_list.is_empty() {
            if track_neighbor_influence {
                debug_assert_eq!(
                    self.arena.changed_list.len(),
                    self.arena.changed_influence.len()
                );
                self.arena.mark_changed_bitmap_unsynced();
            } else {
                debug_assert!(self.arena.changed_influence.is_empty());
                self.arena.mark_changed_bitmap_unsynced_uniform_all();
            }
        }

        self.arena.flip_borders();
        self.arena.flip_cells();

        finalize_prune_and_expand(&mut self.arena);

        self.population_cache = None;
        self.generation += 1;
    }

    pub fn step(&mut self) {
        // Split borrow: take a shared ref to the pool before mutably borrowing self.
        let pool = &self.pool as *const rayon::ThreadPool;
        // SAFETY: `pool` is not modified during `install`, and `step_impl` only
        // mutates arena/generation/population_cache - never the pool itself.
        unsafe { &*pool }.install(|| {
            self.step_impl();
        });
    }

    pub fn step_n(&mut self, n: u64) {
        if n == 0 {
            return;
        }

        // Split borrow: take a shared ref to the pool before mutably borrowing self.
        let pool = &self.pool as *const rayon::ThreadPool;
        // SAFETY: `pool` is not modified during `install`, and `step_impl` only
        // mutates arena/generation/population_cache - never the pool itself.
        unsafe { &*pool }.install(|| {
            let mut remaining = n;
            while remaining >= 4 {
                self.step_impl();
                self.step_impl();
                self.step_impl();
                self.step_impl();
                remaining -= 4;
            }
            while remaining != 0 {
                self.step_impl();
                remaining -= 1;
            }
        });
    }

    pub fn population(&mut self) -> u64 {
        if let Some(cached) = self.population_cache {
            return cached;
        }

        let cp = self.arena.cell_phase;
        let mut total = 0u64;
        let occupied_bits = &self.arena.occupied_bits;
        let cell_vec = &self.arena.cell_bufs[cp];
        let meta_vec = &mut self.arena.meta;
        for (word_idx, &word) in occupied_bits.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let i = (word_idx << 6) + bit;
                if i == 0 || i >= meta_vec.len() {
                    bits &= bits - 1;
                    continue;
                }
                let meta = &mut meta_vec[i];
                if !meta.has_live() {
                    meta.population = 0;
                } else {
                    let pop = if meta.population != POPULATION_UNKNOWN {
                        meta.population
                    } else {
                        let computed = tile::compute_population(&cell_vec[i].0);
                        meta.population = computed;
                        computed
                    };
                    total += pop as u64;
                }
                bits &= bits - 1;
            }
        }

        self.population_cache = Some(total);
        total
    }

    pub fn is_empty(&mut self) -> bool {
        self.population() == 0
    }

    pub fn bounds(&self) -> Option<(i64, i64, i64, i64)> {
        let mut min_x = i64::MAX;
        let mut min_y = i64::MAX;
        let mut max_x = i64::MIN;
        let mut max_y = i64::MIN;
        let mut seen = false;

        self.for_each_live(|x, y| {
            seen = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        });

        seen.then_some((min_x, min_y, max_x, max_y))
    }

    pub fn for_each_live<F: FnMut(i64, i64)>(&self, mut f: F) {
        let cp = self.arena.cell_phase;
        for (word_idx, &word) in self.arena.occupied_bits.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let i = (word_idx << 6) + bit;
                if i == 0 || i >= self.arena.meta.len() {
                    bits &= bits - 1;
                    continue;
                }
                let current = &self.arena.cell_bufs[cp][i].0;
                let coord = self.arena.coords[i];
                let base_x = coord.0 * TILE_SIZE_I64;
                let base_y = coord.1 * TILE_SIZE_I64;

                for (row_index, &row) in current.iter().enumerate() {
                    let mut row_bits = row;
                    while row_bits != 0 {
                        let col = row_bits.trailing_zeros() as i64;
                        f(base_x + col, base_y + row_index as i64);
                        row_bits &= row_bits - 1;
                    }
                }
                bits &= bits - 1;
            }
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use rand::{Rng, SeedableRng};

    use super::{
        KernelBackend, PARALLEL_KERNEL_MIN_ACTIVE, TILE_SIZE_I64, TurboLife, TurboLifeConfig,
        active_broadcast_workers, auto_pool_thread_count_for_physical, churn_at_most_percent,
        churn_below_percent, dynamic_parallel_chunk_size, dynamic_target_chunks_per_worker,
        memory_parallel_cap, physical_core_count,
    };

    const PARALLEL_TEST_TILE_GRID: i64 = 12;
    const PARALLEL_TEST_CELLS_PER_TILE: usize = 16;

    fn seed_parallel_scheduler_fixture(engine: &mut TurboLife) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xA55A_CE11_1234_5678);
        let mut cells = Vec::new();
        for ty in 0..PARALLEL_TEST_TILE_GRID {
            for tx in 0..PARALLEL_TEST_TILE_GRID {
                let base_x = tx * TILE_SIZE_I64;
                let base_y = ty * TILE_SIZE_I64;
                for _ in 0..PARALLEL_TEST_CELLS_PER_TILE {
                    let local_x = rng.random_range(0i64..TILE_SIZE_I64);
                    let local_y = rng.random_range(0i64..TILE_SIZE_I64);
                    cells.push((base_x + local_x, base_y + local_y));
                }
            }
        }
        engine.set_cells_alive(cells);
    }

    fn run_parallel_scheduler_fixture(threads: usize, steps: u64) -> (u64, u64) {
        let mut engine = TurboLife::with_config(
            TurboLifeConfig::default()
                .thread_count(threads)
                .kernel(KernelBackend::Scalar),
        );
        seed_parallel_scheduler_fixture(&mut engine);
        assert!(engine.arena.changed_list.len() >= PARALLEL_KERNEL_MIN_ACTIVE);

        engine.step_n(steps);
        let pop = engine.population();
        let mut hasher = DefaultHasher::new();
        engine.for_each_live(|x, y| {
            x.hash(&mut hasher);
            y.hash(&mut hasher);
        });
        (pop, hasher.finish())
    }

    fn assert_border_live_mask_cache_sync(engine: &TurboLife) {
        for phase in 0..2 {
            assert_eq!(
                engine.arena.borders[phase].len(),
                engine.arena.border_live_masks[phase].len()
            );
            for i in 0..engine.arena.borders[phase].len() {
                let border = engine.arena.borders[phase].get(i);
                assert_eq!(
                    engine.arena.border_live_masks[phase][i],
                    border.live_mask(),
                    "phase={phase} idx={i}"
                );
            }
        }
    }

    #[test]
    fn memory_parallel_cap_stays_monotonic() {
        let mut prev = 0usize;
        for thread_count in 1..=128 {
            let capped = memory_parallel_cap(thread_count);
            assert!(
                capped <= thread_count,
                "cap should never exceed the requested threads: {thread_count} -> {capped}"
            );
            assert!(
                capped >= prev,
                "cap should be monotonic with thread count: {thread_count} -> {capped}, prev={prev}"
            );
            prev = capped;
        }
    }

    #[test]
    fn memory_parallel_cap_keeps_high_core_boundary() {
        assert_eq!(memory_parallel_cap(8), 8);
        assert_eq!(memory_parallel_cap(9), 8);
        assert_eq!(memory_parallel_cap(10), 8);
        assert_eq!(memory_parallel_cap(11), 8);
        assert_eq!(memory_parallel_cap(12), 8);
        assert_eq!(memory_parallel_cap(16), 8);
    }

    #[test]
    fn active_broadcast_workers_caps_to_available_threads() {
        assert_eq!(active_broadcast_workers(1, 1), 1);
        assert_eq!(active_broadcast_workers(2, 8), 2);
        assert_eq!(active_broadcast_workers(8, 2), 2);
        assert_eq!(active_broadcast_workers(0, 4), 0);
    }

    #[test]
    fn auto_pool_thread_count_targets_bandwidth_sweet_spot() {
        assert_eq!(auto_pool_thread_count_for_physical(0), 1);
        assert_eq!(auto_pool_thread_count_for_physical(1), 1);
        assert_eq!(auto_pool_thread_count_for_physical(4), 4);
        assert_eq!(auto_pool_thread_count_for_physical(8), 8);
        assert_eq!(auto_pool_thread_count_for_physical(9), 6);
        assert_eq!(auto_pool_thread_count_for_physical(12), 6);
        assert_eq!(auto_pool_thread_count_for_physical(16), 8);
        assert_eq!(auto_pool_thread_count_for_physical(24), 12);
    }

    #[test]
    fn dynamic_chunk_targets_stay_flat_across_frontier_sizes() {
        assert_eq!(dynamic_target_chunks_per_worker(1, 1), 2);
        assert_eq!(dynamic_target_chunks_per_worker(2_047, 2_047), 2);
        assert_eq!(dynamic_target_chunks_per_worker(2_048, 100), 2);
        assert_eq!(dynamic_target_chunks_per_worker(2_048, 900), 2);
        assert_eq!(dynamic_target_chunks_per_worker(16_383, 9_000), 2);
        assert_eq!(dynamic_target_chunks_per_worker(16_384, 16_384), 2);
    }

    #[test]
    fn churn_helpers_handle_large_frontiers_without_saturation_bias() {
        assert!(churn_at_most_percent(6_500, 10_000, 65));
        assert!(!churn_below_percent(6_500, 10_000, 65));
        assert!(churn_below_percent(6_499, 10_000, 65));

        assert!(!churn_at_most_percent(usize::MAX, usize::MAX, 65));
        assert!(!churn_below_percent(usize::MAX, usize::MAX, 65));
        assert!(churn_at_most_percent(usize::MAX / 2, usize::MAX, 65));
    }

    #[test]
    fn dynamic_chunk_size_obeys_bounds_and_tiers() {
        let small = dynamic_parallel_chunk_size(800, 600, 4);
        let medium_balanced = dynamic_parallel_chunk_size(4_096, 1_200, 4);
        let medium_high = dynamic_parallel_chunk_size(4_096, 3_000, 4);
        let large = dynamic_parallel_chunk_size(65_536, 50_000, 4);
        assert_eq!(small, 100);
        assert_eq!(medium_balanced, 512);
        assert_eq!(medium_high, 512);
        assert_eq!(large, 2_048);
    }

    #[test]
    fn prune_queue_gate_handles_assume_changed_mode() {
        assert!(!super::should_queue_prune::<true>(true, false));
        assert!(!super::should_queue_prune::<true>(true, true));
        assert!(!super::should_queue_prune::<false>(true, false));
        assert!(!super::should_queue_prune::<false>(true, true));

        assert!(super::should_queue_prune::<true>(false, false));
        assert!(!super::should_queue_prune::<true>(false, true));
        assert!(super::should_queue_prune::<false>(false, false));
        assert!(super::should_queue_prune::<false>(false, true));
    }

    #[test]
    fn worker_scratch_is_cacheline_aligned() {
        assert_eq!(std::mem::align_of::<super::WorkerScratch>(), 64);
    }

    #[test]
    fn worker_scratch_reserve_for_chunk_reaches_requested_capacity() {
        let mut scratch = super::WorkerScratch::default();
        scratch.reserve_for_chunk::<true, true>(8);

        let chunk_target = scratch.changed.capacity() + 1;
        scratch.reserve_for_chunk::<true, true>(chunk_target);
        assert!(
            scratch.changed.capacity() >= chunk_target,
            "changed capacity {} < chunk target {chunk_target}",
            scratch.changed.capacity()
        );
        assert!(
            scratch.changed_influence.capacity() >= chunk_target,
            "changed_influence capacity {} < chunk target {chunk_target}",
            scratch.changed_influence.capacity()
        );
        assert!(
            scratch.prune.capacity() >= chunk_target,
            "prune capacity {} < chunk target {chunk_target}",
            scratch.prune.capacity()
        );

        let expand_chunk_target = scratch.expand.capacity().div_ceil(3) + 1;
        let expand_target = expand_chunk_target.saturating_mul(3);
        scratch.reserve_for_chunk::<false, false>(expand_chunk_target);
        assert!(
            scratch.expand.capacity() >= expand_target,
            "expand capacity {} < expand target {expand_target}",
            scratch.expand.capacity()
        );
    }

    #[test]
    fn derive_changed_from_active_minus_prune_excludes_pruned_tiles() {
        let mut engine = TurboLife::new();
        let keep_a = engine.arena.allocate((0, 0));
        let drop_b = engine.arena.allocate((1, 0));
        let keep_c = engine.arena.allocate((2, 0));

        engine.arena.active_set = vec![keep_a, drop_b, keep_c];
        engine.arena.active_set_dense_contiguous = true;
        engine.arena.prune_buf = vec![drop_b];
        engine.arena.changed_influence = vec![1, 2, 3];

        engine.derive_changed_from_active_minus_prune();

        assert_eq!(engine.arena.changed_list, vec![keep_a, keep_c]);
        assert!(engine.arena.changed_influence.is_empty());
        assert!(engine.arena.active_set.is_empty());
        assert!(!engine.arena.active_set_dense_contiguous);

        // Ensure prune-mark scratch cleanup does not leak into future calls.
        engine.arena.active_set = vec![drop_b];
        engine.arena.prune_buf.clear();
        engine.derive_changed_from_active_minus_prune();
        assert_eq!(engine.arena.changed_list, vec![drop_b]);
    }

    #[test]
    fn default_pool_uses_auto_thread_count() {
        let engine = TurboLife::new();
        let expected = auto_pool_thread_count_for_physical(physical_core_count());
        assert_eq!(engine.pool.current_num_threads(), expected);
    }

    #[test]
    fn pool_thread_metadata_tracks_runtime_pool_size() {
        let engine = TurboLife::with_config(TurboLifeConfig::default().thread_count(3));
        let runtime_threads = engine.pool.current_num_threads();

        assert_eq!(engine.pool_threads, runtime_threads);
        assert_eq!(engine.worker_scratch.len(), runtime_threads);
    }

    #[test]
    fn empty_universe_advances_generation() {
        let mut engine = TurboLife::new();
        assert_eq!(engine.generation(), 0);
        engine.step();
        assert_eq!(engine.generation(), 1);
        assert_eq!(engine.population(), 0);
    }

    #[test]
    fn step_changed_tiles_are_deduped_for_followup_mutation() {
        let mut engine = TurboLife::new();
        engine.set_cells_alive([(10, 10), (11, 10), (12, 10)]);
        engine.step();

        assert!(!engine.get_cell(10, 10));
        let changed_before = engine.arena.changed_list.len();
        assert!(changed_before > 0);

        engine.set_cell(10, 10, true);
        let changed_after = engine.arena.changed_list.len();

        assert_eq!(
            changed_after, changed_before,
            "set_cell should not duplicate tiles already queued in changed_list"
        );
    }

    #[test]
    fn parallel_step_changed_tiles_are_deduped_for_followup_mutation() {
        let mut engine = TurboLife::with_config(
            TurboLifeConfig::default()
                .thread_count(4)
                .kernel(KernelBackend::Scalar),
        );
        seed_parallel_scheduler_fixture(&mut engine);
        engine.step();

        let changed_before = engine.arena.changed_list.len();
        assert!(changed_before > 0);
        let tile_idx = engine.arena.changed_list[0];
        let coord = engine.arena.coords[tile_idx.index()];
        let x = coord.0 * TILE_SIZE_I64 + 1;
        let y = coord.1 * TILE_SIZE_I64 + 1;
        let alive = engine.get_cell(x, y);
        engine.set_cell(x, y, !alive);
        let changed_after = engine.arena.changed_list.len();

        assert_eq!(
            changed_after, changed_before,
            "set_cell should not duplicate tiles already queued in changed_list after parallel steps"
        );
    }

    #[test]
    fn set_cell_requeues_stable_tile_after_unrelated_steps() {
        let mut engine = TurboLife::new();
        // Stable block in tile (0, 0).
        engine.set_cells_alive([(0, 0), (1, 0), (0, 1), (1, 1)]);
        // Distant blinker keeps subsequent generations non-empty.
        engine.set_cells_alive([(256, 256), (257, 256), (258, 256)]);

        engine.step();
        engine.step();

        let changed_before = engine.arena.changed_list.len();
        engine.set_cell(0, 0, false);
        let changed_after = engine.arena.changed_list.len();

        assert_eq!(
            changed_after,
            changed_before + 1,
            "set_cell should requeue a stable tile even after unrelated steps"
        );
    }

    #[test]
    fn set_cell_alive_fast_path_updates_frontier_and_border_cache() {
        let mut engine = TurboLife::new();

        engine.set_cell_alive(5, 5);
        assert!(engine.get_cell(5, 5));
        assert!(engine.arena.idx_at((1, 0)).is_none());
        assert_border_live_mask_cache_sync(&engine);

        engine.set_cell_alive(63, 5);
        assert!(engine.get_cell(63, 5));
        assert!(engine.arena.idx_at((1, 0)).is_some());
        assert_border_live_mask_cache_sync(&engine);

        let changed_before = engine.arena.changed_list.len();
        engine.set_cell_alive(63, 5);
        assert_eq!(engine.arena.changed_list.len(), changed_before);
    }

    #[test]
    fn border_live_mask_cache_stays_synced_across_mutations_and_steps() {
        let mut engine = TurboLife::new();
        assert_border_live_mask_cache_sync(&engine);

        engine.set_cells_alive([(0, 0), (63, 63), (64, 0), (64, 1), (128, 128)]);
        assert_border_live_mask_cache_sync(&engine);

        engine.set_cell(63, 63, false);
        engine.set_cell(1, 1, true);
        assert_border_live_mask_cache_sync(&engine);

        engine.step();
        assert_border_live_mask_cache_sync(&engine);

        engine.step_n(3);
        assert_border_live_mask_cache_sync(&engine);

        engine.set_cells([(0, 0, false), (127, 127, true), (190, -5, true)]);
        assert_border_live_mask_cache_sync(&engine);
    }

    #[test]
    fn set_cells_dead_to_dead_batch_marks_alt_phase_dirty() {
        let mut engine = TurboLife::new();
        engine.set_cells([(0, 0, true), (0, 0, false)]);

        let idx = engine
            .arena
            .idx_at((0, 0))
            .expect("set_cells should allocate the tile");
        let meta = engine.arena.meta(idx);
        assert!(!meta.has_live());
        assert!(meta.alt_phase_dirty());
    }

    #[test]
    fn parallel_dynamic_scheduler_matches_single_thread() {
        let single = run_parallel_scheduler_fixture(1, 4);
        let dynamic = run_parallel_scheduler_fixture(2, 4);
        assert_eq!(single, dynamic);
    }

    #[test]
    fn parallel_static_scheduler_matches_single_thread() {
        let single = run_parallel_scheduler_fixture(1, 4);
        let static_sched = run_parallel_scheduler_fixture(4, 4);
        assert_eq!(single, static_sched);
    }
}
