/**
 * common.metal — Native Metal Shading Language (MSL) definitions
 *
 * This file provides the macros, type definitions, and utility functions needed
 * by all OpenMM Metal compute kernels. It replaces the earlier cl2Metal
 * compatibility layer with native MSL constructs.
 *
 * Key differences from the OpenCL/cl2Metal version:
 *   - No #pragma OPENCL EXTENSION — MSL needs none.
 *   - Address-space qualifiers use MSL keywords (device, threadgroup).
 *   - Thread indices arrive as kernel parameters, not built-in functions.
 *     The MM_THREAD_ARGS / MM_THREAD_PARAMS / MM_THREAD_PASS macros wire
 *     them through the call graph.
 *   - SIMD intrinsics (simd_sum, simd_ballot, …) are first-class MSL
 *     built-ins — no __asm("air.…") hacks required.
 *   - Atomics use metal::atomic_fetch_add_explicit.
 */

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// 1. Address-space qualifiers
// ---------------------------------------------------------------------------

#define KERNEL  kernel
#define DEVICE
#define LOCAL       threadgroup
#define LOCAL_ARG   threadgroup
#define GLOBAL      device
#define RESTRICT
#define restrict

// OpenCL __keyword compatibility (used by bonded force template code)
#define __kernel   kernel
#define __global   device
#define __local    threadgroup
#define __constant constant

// ---------------------------------------------------------------------------
// 2. Thread-index macros
//
//    In OpenCL the indices are free functions (get_global_id(0), …).
//    In MSL they must be declared as kernel parameters with [[…]] attributes.
//
//    Every kernel should end its parameter list with MM_THREAD_ARGS.
//    Helper (non-kernel) functions that need the indices accept
//    MM_THREAD_PARAMS and callers pass them with MM_THREAD_PASS.
// ---------------------------------------------------------------------------

// Thread indices are stored in threadgroup memory so that ANY function
// (kernel or device/helper) can access them via these macros.
// Each kernel must call MM_INIT_THREAD_STATE as its first statement.
// This avoids requiring every helper function to take thread IDs as parameters.

// Forward declarations of threadgroup variables (defined per-threadgroup)
// We use a unique struct to avoid name collisions.
struct _mm_thread_state {
    uint global_id;
    uint local_id;
    uint global_size;
    uint local_size;
    uint group_id;
    uint num_groups;
};

// Each thread stores its own state in thread address space
// (MSL 'thread' is like CUDA __register__)

#define GLOBAL_ID    _mm_ts.global_id
#define LOCAL_ID     _mm_ts.local_id
#define GLOBAL_SIZE  _mm_ts.global_size
#define LOCAL_SIZE   _mm_ts.local_size
#define GROUP_ID     _mm_ts.group_id
#define NUM_GROUPS   _mm_ts.num_groups

// Kernel parameters for thread position — appended to every kernel signature
#define MM_THREAD_ARGS \
    uint _mm_gid   [[thread_position_in_grid]],       \
    uint _mm_lid   [[thread_position_in_threadgroup]], \
    uint _mm_gsz   [[threads_per_grid]],               \
    uint _mm_lsz   [[threads_per_threadgroup]],        \
    uint _mm_grid  [[threadgroup_position_in_grid]],   \
    uint _mm_ngrp  [[threadgroups_per_grid]]

// Initialize thread state — MUST be first line in every kernel body
#define MM_INIT_THREAD_STATE \
    _mm_thread_state _mm_ts; \
    _mm_ts.global_id = _mm_gid; \
    _mm_ts.local_id = _mm_lid; \
    _mm_ts.global_size = _mm_gsz; \
    _mm_ts.local_size = _mm_lsz; \
    _mm_ts.group_id = _mm_grid; \
    _mm_ts.num_groups = _mm_ngrp;

// For helper functions that need thread state passed explicitly
#define MM_THREAD_PARAMS _mm_thread_state _mm_ts
#define MM_THREAD_PASS _mm_ts

// ---------------------------------------------------------------------------
// 3. OpenCL built-in compatibility shims
//
//    Existing kernel source calls get_global_id(0) etc. directly in many
//    places. Map them to the parameter variables above so the code compiles
//    without rewriting every kernel.
// ---------------------------------------------------------------------------

#define get_global_id(dim)   _mm_ts.global_id
#define get_local_id(dim)    _mm_ts.local_id
#define get_global_size(dim) _mm_ts.global_size
#define get_local_size(dim)  _mm_ts.local_size
#define get_group_id(dim)    _mm_ts.group_id
#define get_num_groups(dim)  _mm_ts.num_groups

// ---------------------------------------------------------------------------
// 4. Synchronisation
// ---------------------------------------------------------------------------

#define SYNC_THREADS threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
#define MEM_FENCE    simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

// Default for SYNC_WARPS — MetalContext.cpp may override this via
// compilation defines.  Provide a safe default (full threadgroup barrier)
// so kernels compile even without the runtime define.
#ifndef SYNC_WARPS
#define SYNC_WARPS   simdgroup_barrier(mem_flags::mem_threadgroup);
#endif

// OpenCL barrier() / mem_fence() compatibility.
// Several kernel files call barrier(CLK_LOCAL_MEM_FENCE+CLK_GLOBAL_MEM_FENCE)
// directly rather than through the SYNC_THREADS macro.

#define CLK_LOCAL_MEM_FENCE  1
#define CLK_GLOBAL_MEM_FENCE 2

inline void barrier(int /* flags */) {
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
}

inline void mem_fence(int /* flags */) {
    simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
}

// ---------------------------------------------------------------------------
// 5. Atomics
// ---------------------------------------------------------------------------

// 32-bit atomic add (int / uint).
// OpenCL kernels call ATOMIC_ADD(&buffer[i], value) where buffer is
// __global unsigned long* (fixed-point forces) or __global int*.
// Metal requires atomic types; we reinterpret the device pointer.

// Metal does NOT have atomic_ulong. 64-bit atomic add is emulated
// with two 32-bit atomic adds (low word + carry to high word),
// matching the original OpenCL fallback in the old common.metal.

// ATOMIC_ADD — C++ overloaded for unsigned long, unsigned int, and int.
// Uses function overloading so the macro dispatches correctly by pointer type.

inline void _mm_atomic_add(device unsigned long* dest, unsigned long value) {
    device atomic_uint* word = reinterpret_cast<device atomic_uint*>(dest);
    unsigned int lower = (unsigned int)(value);
    unsigned int upper = (unsigned int)(value >> 32);
    unsigned int result = atomic_fetch_add_explicit(&word[0], lower, memory_order_relaxed);
    int carry = (lower + (unsigned long)result >= 0x100000000UL ? 1 : 0);
    upper += carry;
    if (upper != 0)
        atomic_fetch_add_explicit(&word[1], upper, memory_order_relaxed);
}

inline unsigned int _mm_atomic_add(device unsigned int* dest, unsigned int value) {
    device atomic_uint* a = reinterpret_cast<device atomic_uint*>(dest);
    return atomic_fetch_add_explicit(a, value, memory_order_relaxed);
}

inline int _mm_atomic_add(device int* dest, int value) {
    device atomic_int* a = reinterpret_cast<device atomic_int*>(dest);
    return atomic_fetch_add_explicit(a, value, memory_order_relaxed);
}

inline void _mm_atomic_add(device float* dest, float value) {
    device atomic<float>* a = reinterpret_cast<device atomic<float>*>(dest);
    atomic_fetch_add_explicit(a, value, memory_order_relaxed);
}

#define ATOMIC_ADD(dest, value) _mm_atomic_add(dest, value)

// Legacy aliases kept for kernels that call them directly.
inline void _mm_atomic_add_ulong(device unsigned long* dest, unsigned long value) { _mm_atomic_add(dest, value); }
inline void _mm_atomic_add_uint(device unsigned int* dest, unsigned int value) { _mm_atomic_add(dest, value); }
inline void _mm_atomic_add_int(device int* dest, int value) { _mm_atomic_add(dest, value); }

// OpenCL atom_inc — atomic increment, returns old value.
inline unsigned int atom_inc(volatile device unsigned int* dest) {
    device atomic_uint* a = reinterpret_cast<device atomic_uint*>(
        const_cast<device unsigned int*>(dest));
    return atomic_fetch_add_explicit(a, 1u, memory_order_relaxed);
}

inline int atom_inc(volatile device int* dest) {
    device atomic_int* a = reinterpret_cast<device atomic_int*>(
        const_cast<device int*>(dest));
    return atomic_fetch_add_explicit(a, 1, memory_order_relaxed);
}

// Legacy OpenCL name used by some kernels directly.
inline unsigned long atom_add(volatile device unsigned long* dest, unsigned long value) {
    device atomic_uint* word = reinterpret_cast<device atomic_uint*>(
        const_cast<device unsigned long*>(dest));
    unsigned int lower = (unsigned int)(value);
    unsigned int upper = (unsigned int)(value >> 32);
    unsigned int result = atomic_fetch_add_explicit(&word[0], lower, memory_order_relaxed);
    int carry = (lower + (unsigned long)result >= 0x100000000UL ? 1 : 0);
    upper += carry;
    if (upper != 0)
        atomic_fetch_add_explicit(&word[1], upper, memory_order_relaxed);
    return 0;
}

inline unsigned int atom_add(volatile device unsigned int* dest, unsigned int value) {
    device atomic_uint* a = reinterpret_cast<device atomic_uint*>(
        const_cast<device unsigned int*>(dest));
    return atomic_fetch_add_explicit(a, value, memory_order_relaxed);
}

inline int atom_add(volatile device int* dest, int value) {
    device atomic_int* a = reinterpret_cast<device atomic_int*>(
        const_cast<device int*>(dest));
    return atomic_fetch_add_explicit(a, value, memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// 6. Type definitions
// ---------------------------------------------------------------------------

typedef long          mm_long;
typedef unsigned long mm_ulong;

// ---------------------------------------------------------------------------
// 7. Vector-type constructors  (make_* macros)
//
//    OpenCL uses C-style compound literals; MSL uses C++ constructors.
// ---------------------------------------------------------------------------

#define make_short2(x...)  short2(x)
#define make_short3(x...)  short3(x)
#define make_short4(x...)  short4(x)
#define make_int2(x...)    int2(x)
#define make_int3(x...)    int3(x)
#define make_int4(x...)    int4(x)
#define make_float2(x...)  float2(x)
#define make_float3(x...)  float3(x)
#define make_float4(x...)  float4(x)
#define make_double2(x...) double2(x)
#define make_double3(x...) double3(x)
#define make_double4(x...) double4(x)

// ---------------------------------------------------------------------------
// 8. Swizzle helper
// ---------------------------------------------------------------------------

#define trimTo3(v) (v).xyz

// cross() overloads for float4/double4 — MSL's built-in cross() only accepts float3/double3.
// These extract .xyz, compute the cross product, and return a 4-component vector (w=0)
// to match OpenCL semantics where cross(float4, float4) -> float4.
inline float4 cross(float4 a, float4 b) { return float4(cross(a.xyz, b.xyz), 0.0f); }
inline half4 cross(half4 a, half4 b) { return half4(cross(a.xyz, b.xyz), half(0)); }

// ---------------------------------------------------------------------------
// 9. Math-function aliases (CUDA compatibility)
//
//    CUDA has separate *f functions for single precision.  MSL (like OpenCL)
//    overloads the same name for float and double, so map them through.
// ---------------------------------------------------------------------------

#define sqrtf(x)      sqrt(x)
#define rsqrtf(x)     rsqrt(x)
#define expf(x)       exp(x)
#define logf(x)       log(x)
#define powf(x, y)    pow(x, y)
#define cosf(x)       cos(x)
#define sinf(x)       sin(x)
#define tanf(x)       tan(x)
#define acosf(x)      acos(x)
#define asinf(x)      asin(x)
#define atanf(x)      atan(x)
#define atan2f(x, y)  atan2(x, y)

// ---------------------------------------------------------------------------
// 9b. OpenCL native_* math function compatibility
//
//     OpenCL has native_sqrt(), native_rsqrt(), etc. that use fast
//     hardware approximations.  MSL doesn't have these names — the standard
//     sqrt/rsqrt/exp/log ARE the fast versions on Apple GPUs.
//     Map them through so kernel code using native_* compiles.
// ---------------------------------------------------------------------------

#define native_sqrt(x)   sqrt(x)
#define native_rsqrt(x)  rsqrt(x)
#define native_recip(x)  (1.0f/(x))
#define native_exp(x)    exp(x)
#define native_log(x)    log(x)

// ---------------------------------------------------------------------------
// 10. Fixed-point conversion utility
// ---------------------------------------------------------------------------

inline long realToFixedPoint(real x) {
    // Clamp to avoid undefined behavior when converting inf/NaN to long.
    // Metal produces 0 for (long)inf, unlike OpenCL which saturates.
    // Max fixed-point range: ~2.1e9 in real units (fits in long after scaling).
    real scaled = x * (real)0x100000000;
    scaled = metal::clamp(scaled, (real)(-9.2e18f), (real)(9.2e18f));
    return (long)scaled;
}

// ---------------------------------------------------------------------------
// 11. VENDOR_APPLE — always defined on native Metal
// ---------------------------------------------------------------------------

#ifndef VENDOR_APPLE
#define VENDOR_APPLE 1
#endif

// ---------------------------------------------------------------------------
// 12. Loop-unrolling helpers
// ---------------------------------------------------------------------------

#define FORCE_UNROLL_4(__expr, __start_index) \
    __expr(__start_index + 0); \
    __expr(__start_index + 1); \
    __expr(__start_index + 2); \
    __expr(__start_index + 3);

#define FORCE_UNROLL_32(__expr) \
    FORCE_UNROLL_4(__expr, 0);  \
    FORCE_UNROLL_4(__expr, 4);  \
    FORCE_UNROLL_4(__expr, 8);  \
    FORCE_UNROLL_4(__expr, 12); \
    FORCE_UNROLL_4(__expr, 16); \
    FORCE_UNROLL_4(__expr, 20); \
    FORCE_UNROLL_4(__expr, 24); \
    FORCE_UNROLL_4(__expr, 28);

// ---------------------------------------------------------------------------
// 13. SIMD intrinsics  (native MSL — no inline asm needed)
//
//     MSL provides these as first-class built-ins in <metal_simdgroup>.
//     We define the OpenCL sub_group_* wrappers that existing kernels use.
// ---------------------------------------------------------------------------

// simd_sum, simd_ballot, simd_all, simd_any, simd_is_first, simd_shuffle
// are all native MSL built-ins — available directly.

// ctz() — count trailing zeros.  MSL provides ctz() natively.
// clz() — count leading zeros.   MSL provides clz() natively.
// popcount() — population count.  MSL provides popcount() natively.

// Sub-group wrappers used by existing kernel code.

inline float sub_group_reduce_add(float x) {
    return simd_sum(x);
}

inline int sub_group_elect() {
    return select(0, 1, simd_is_first());
}

inline int sub_group_all(int predicate) {
    return select(0, 1, simd_all(predicate != 0));
}

inline int sub_group_any(int predicate) {
    return select(0, 1, simd_any(predicate != 0));
}

inline int sub_group_non_uniform_all(int predicate) {
    return select(0, 1, simd_all(predicate != 0));
}

inline int sub_group_non_uniform_any(int predicate) {
    return select(0, 1, simd_any(predicate != 0));
}

inline uint4 sub_group_ballot(int predicate) {
    uint4 output = uint4(0);
    // simd_ballot returns simd_vote, extract the bitmask via explicit conversion
    simd_vote vote = simd_ballot(predicate != 0);
    ulong mask = (ulong)(vote);
    output.x = (uint)(mask & 0xFFFFFFFF);
    output.y = (uint)(mask >> 32);
    return output;
}

// ---------------------------------------------------------------------------
// 14. Half-precision load/store compatibility
//
//     OpenCL's vstorea_half3 / vloada_half3 use a stride of 4 elements
//     (half3 padded to half4 alignment).  Provide equivalent wrappers.
// ---------------------------------------------------------------------------

inline void vstorea_half3_rtp(float3 data, int index, device half* p) {
    p[index * 4]     = half(data.x);
    p[index * 4 + 1] = half(data.y);
    p[index * 4 + 2] = half(data.z);
}

inline half3 vloada_half3(int index, const device half* p) {
    return half3(p[index * 4], p[index * 4 + 1], p[index * 4 + 2]);
}

// ---------------------------------------------------------------------------
// 15. float16 / int16 / float8 compatibility
//
//     OpenCL provides 8- and 16-wide vector types (float8, float16, int16).
//     MSL vectors go up to 4.  We emulate them as structs with element access.
//
//     We use mm_float16 / mm_int16 / mm_float8 as the struct names and
//     provide typedefs for the OpenCL names (float16, int16, float8) so
//     existing kernel code compiles unmodified.
// ---------------------------------------------------------------------------

struct mm_float16 {
    float4 v[4];

    inline float operator[](int i) const {
        return v[i / 4][i % 4];
    }
};
// Note: can't typedef float16 — Metal has a built-in with that name.
// Kernel code must use mm_float16 directly.

struct mm_int16 {
    int4 v[4];

    inline int operator[](int i) const {
        return v[i / 4][i % 4];
    }
};
// Note: can't typedef int16 — Metal has a built-in with that name.

struct mm_float8 {
    float s0, s1, s2, s3, s4, s5, s6, s7;
};
// Note: can't typedef float8 — Metal has a built-in with that name.

// ---------------------------------------------------------------------------
// 16. select() compatibility
//
//     MSL's select(a, b, cond) has the same semantics as OpenCL:
//       result = cond ? b : a
//     No wrapper needed — it's a built-in.
// ---------------------------------------------------------------------------

// (No additional code required — MSL select() matches OpenCL select().)

// ---------------------------------------------------------------------------
// 17. convert_* compatibility
//
//     OpenCL uses convert_float4(), convert_int4(), etc.  MSL uses
//     constructor syntax: float4(intVec).  Provide macros for the
//     conversions that appear in the kernel source.
// ---------------------------------------------------------------------------

#define convert_float(x)  float(x)
#define convert_float2(x) float2(x)
#define convert_float3(x) float3(x)
#define convert_float4(x) float4(x)
#define convert_int(x)    int(x)
#define convert_int2(x)   int2(x)
#define convert_int3(x)   int3(x)
#define convert_int4(x)   int4(x)
#define convert_uint(x)   uint(x)
#define convert_uint4(x)  uint4(x)

// ---------------------------------------------------------------------------
// End of common.metal
// ---------------------------------------------------------------------------
