/* Code for CUDA stream compaction. Roughly based on:
    Billeter M, Olsson O, Assarsson U. Efficient Stream Compaction on Wide SIMD Many-Core Architectures.
        High Performance Graphics 2009.

    Notes:
        - paper recommends 128 threads/block, so this is hard coded.
        - I only implement the prefix-sum based compact primitive, and not the POPC one, as that is more
          complicated and performs poorly on current hardware
        - I only implement the scattered- and staged-write variant of phase III as it they have reasonable
          performance across most of the tested workloads in the paper. The selective variant is not
          implemented.
        - The prefix sum of per-block element counts (phase II) is not done in a particularly efficient
          manner. It is, however, done in a very easy to program manner, and integrated into the top of
          phase III, reducing the number of kernel invocations required. If one wanted to use existing code,
          it'd be easy to take the CUDA SDK scanLargeArray sample, and do a prefix sum over dgBlockCounts in
          a phase II kernel. You could also adapt the existing prescan128 to take an initial value, and scan
          dgBlockCounts in stages.

  Date:         23 Aug 2009
  Author:       CUDA version by Imran Haque (ihaque@cs.stanford.edu), converted to Metal by Peter Eastman
  Affiliation:  Stanford University
  License:      Public Domain
*/

// Phase 1: Count valid elements per thread block
// Hard-code 128 thd/blk
unsigned int sumReduce128(LOCAL unsigned int* arr, MM_THREAD_PARAMS) {
    // Parallel reduce element counts
    // Assumes 128 thd/block
    int thread = LOCAL_ID;
    if (thread < 64) arr[thread] += arr[thread+64];
    barrier(CLK_LOCAL_MEM_FENCE);
#ifdef WARPS_ARE_ATOMIC
    if (thread < 32) {
        arr[thread] += arr[thread+32];
        if (thread < 16) arr[thread] += arr[thread+16];
        if (thread < 8) arr[thread] += arr[thread+8];
        if (thread < 4) arr[thread] += arr[thread+4];
        if (thread < 2) arr[thread] += arr[thread+2];
        if (thread < 1) arr[thread] += arr[thread+1];
    }
#else
    if (thread < 32) arr[thread] += arr[thread+32];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thread < 16) arr[thread] += arr[thread+16];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thread < 8) arr[thread] += arr[thread+8];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thread < 4) arr[thread] += arr[thread+4];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thread < 2) arr[thread] += arr[thread+2];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thread < 1) arr[thread] += arr[thread+1];
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
    return arr[0];
}

KERNEL void countElts(GLOBAL unsigned int* restrict dgBlockCounts, GLOBAL const unsigned int* restrict dgValid, const unsigned int len, LOCAL unsigned int* restrict dsCount,
        MM_THREAD_ARGS) {
    dsCount[LOCAL_ID] = 0;
    unsigned int ub;
    const unsigned int eltsPerBlock = len/NUM_GROUPS + ((len % NUM_GROUPS) ? 1 : 0);
    ub = (len < (GROUP_ID+1)*eltsPerBlock) ? len : ((GROUP_ID + 1)*eltsPerBlock);
    for (int base = GROUP_ID * eltsPerBlock; base < (GROUP_ID+1)*eltsPerBlock; base += LOCAL_SIZE) {
        if ((base + LOCAL_ID) < ub && dgValid[base+LOCAL_ID])
            dsCount[LOCAL_ID]++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    unsigned int blockCount = sumReduce128(dsCount, MM_THREAD_PASS);
    if (LOCAL_ID == 0) dgBlockCounts[GROUP_ID] = blockCount;
    return;
}

// Phase 2/3: Move valid elements using SIMD compaction (phase 2 is done implicitly at top of __global__ method)
// Exclusive prefix scan over 128 elements
// Assumes 128 threads
// Taken from cuda SDK "scan" sample for naive scan, with small modifications
int exclusivePrescan128(LOCAL const unsigned int* in, LOCAL unsigned int* outAndTemp, MM_THREAD_PARAMS) {
    const int n=128;
    LOCAL unsigned int* temp = outAndTemp;
    int pout = 1, pin = 0;

    // load input into temp
    // This is exclusive scan, so shift right by one and set first elt to 0
    int thread = LOCAL_ID;
    temp[pout*n + LOCAL_ID] = (LOCAL_ID > 0) ? in[LOCAL_ID-1] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = 1; offset < n; offset *= 2)
    {
        pout = 1 - pout; // swap double buffer indices
        pin  = 1 - pout;
        barrier(CLK_LOCAL_MEM_FENCE);
        temp[pout*n+LOCAL_ID] = temp[pin*n+LOCAL_ID];
        if (LOCAL_ID >= (uint)offset)
            temp[pout*n+LOCAL_ID] += temp[pin*n+LOCAL_ID - offset];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    return outAndTemp[127]+in[127]; // Return sum of all elements
}

int compactSIMDPrefixSum(LOCAL const unsigned int* dsData, LOCAL const unsigned int* dsValid, LOCAL unsigned int* dsCompact, LOCAL unsigned int* dsLocalIndex, MM_THREAD_PARAMS) {
    int numValid = exclusivePrescan128(dsValid, dsLocalIndex, MM_THREAD_PASS);
    int thread = LOCAL_ID;
    if (dsValid[LOCAL_ID]) dsCompact[dsLocalIndex[LOCAL_ID]] = dsData[LOCAL_ID];
    return numValid;
}

KERNEL void moveValidElementsStaged(GLOBAL const unsigned int* restrict dgData, GLOBAL unsigned int* restrict dgCompact, GLOBAL const unsigned int* restrict dgValid,
            GLOBAL const unsigned int* restrict dgBlockCounts, unsigned int len, GLOBAL unsigned int* restrict dNumValidElements,
            LOCAL unsigned int* restrict inBlock, LOCAL unsigned int* restrict validBlock, LOCAL unsigned int* restrict compactBlock,
        MM_THREAD_ARGS) {
    LOCAL unsigned int dsLocalIndex[256];
    int blockOutOffset=0;
    // Sum up the blockCounts before us to find our offset
    int thread = LOCAL_ID;
    for (int base = 0; base < (int)GROUP_ID; base += (int)LOCAL_SIZE) {
        // Load up the count of valid elements for each block before us in batches of 128
        if ((base + (int)LOCAL_ID) < (int)GROUP_ID) {
            validBlock[LOCAL_ID] = dgBlockCounts[base+LOCAL_ID];
        } else {
            validBlock[LOCAL_ID] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // Parallel reduce these counts
        // Accumulate in the final offset variable
        blockOutOffset += sumReduce128(validBlock, MM_THREAD_PASS);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    unsigned int ub;
    const unsigned int eltsPerBlock = len/NUM_GROUPS + ((len % NUM_GROUPS) ? 1 : 0);
    ub = (len < (GROUP_ID+1)*eltsPerBlock) ? len : ((GROUP_ID + 1)*eltsPerBlock);
    for (int base = GROUP_ID * eltsPerBlock; base < (GROUP_ID+1)*eltsPerBlock; base += LOCAL_SIZE) {
        if ((base + LOCAL_ID) < ub) {
            validBlock[LOCAL_ID] = dgValid[base+LOCAL_ID];
            inBlock[LOCAL_ID] = dgData[base+LOCAL_ID];
        } else {
            validBlock[LOCAL_ID] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        int numValidBlock = compactSIMDPrefixSum(inBlock, validBlock, compactBlock, dsLocalIndex, MM_THREAD_PASS);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (LOCAL_ID < (uint)numValidBlock) {
            dgCompact[blockOutOffset + LOCAL_ID] = compactBlock[LOCAL_ID];
        }
        blockOutOffset += numValidBlock;
    }
    if (GROUP_ID == (NUM_GROUPS-1) && LOCAL_ID == 0) {
        *dNumValidElements = blockOutOffset;
    }
}
