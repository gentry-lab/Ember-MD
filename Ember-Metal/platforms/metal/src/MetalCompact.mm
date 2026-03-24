

/* Code for Metal stream compaction. Roughly based on:
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

#import <Metal/Metal.h>
#include "MetalCompact.h"
#include "MetalKernel.h"
#include "MetalKernelSources.h"

using namespace OpenMM;

MetalCompact::MetalCompact(MetalContext& context) : context(context) {
    dgBlockCounts.initialize<unsigned int>(context, context.getNumThreadBlocks(), "dgBlockCounts");
    ComputeProgram program;
    try {
        program = context.compileProgram(MetalKernelSources::compact);
    } catch (OpenMMException& e) {
        fprintf(stderr, "[Metal] Compact (prefix sum) kernel compilation failed\n");
        throw;
    }
    countKernel = program->createKernel("countElts");
    moveValidKernel = program->createKernel("moveValidElementsStaged");
}

void MetalCompact::compactStream(MetalArray& dOut, MetalArray& dIn, MetalArray& dValid, MetalArray& numValid) {
    // Figure out # elements per block
    unsigned int len = dIn.getSize();
    unsigned int numBlocks = context.getNumThreadBlocks();
    if (numBlocks*128 > len)
        numBlocks = (len+127)/128;

    // Phase 1: Calculate number of valid elements per thread block
    countKernel->setArg(0, dgBlockCounts);
    countKernel->setArg(1, dValid);
    unsigned int lenArg = len;
    countKernel->setArg(2, lenArg);
    dynamic_cast<MetalKernel&>(*countKernel).setThreadgroupMemoryArg(3, 128*sizeof(unsigned int));
    countKernel->execute(len, 128);

    // Phase 2/3: Move valid elements using SIMD compaction
    moveValidKernel->setArg(0, dIn);
    moveValidKernel->setArg(1, dOut);
    moveValidKernel->setArg(2, dValid);
    moveValidKernel->setArg(3, dgBlockCounts);
    moveValidKernel->setArg(4, lenArg);
    moveValidKernel->setArg(5, numValid);
    dynamic_cast<MetalKernel&>(*moveValidKernel).setThreadgroupMemoryArg(6, 128*sizeof(unsigned int));
    dynamic_cast<MetalKernel&>(*moveValidKernel).setThreadgroupMemoryArg(7, 128*sizeof(unsigned int));
    dynamic_cast<MetalKernel&>(*moveValidKernel).setThreadgroupMemoryArg(8, 128*sizeof(unsigned int));
    moveValidKernel->execute(len, 128);
}
