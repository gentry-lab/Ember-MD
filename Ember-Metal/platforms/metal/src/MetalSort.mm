/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2010-2021 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#import <Metal/Metal.h>
#include "MetalSort.h"
#include "MetalKernel.h"
#include "MetalKernelSources.h"
#include <algorithm>
#include <map>
#include <string>

using namespace OpenMM;
using namespace std;

MetalSort::MetalSort(MetalContext& context, SortTrait* trait, unsigned int length, bool uniform) :
        context(context), trait(trait), dataLength(length), uniform(uniform) {
    @autoreleasepool {
        // Create kernels.

        std::map<std::string, std::string> replacements;
        replacements["DATA_TYPE"] = trait->getDataType();
        replacements["KEY_TYPE"] =  trait->getKeyType();
        replacements["SORT_KEY"] = trait->getSortKey();
        replacements["MIN_KEY"] = trait->getMinKey();
        replacements["MAX_KEY"] = trait->getMaxKey();
        replacements["MAX_VALUE"] = trait->getMaxValue();
        replacements["UNIFORM"] = (uniform ? "1" : "0");
        ComputeProgram program;
        try {
            program = context.compileProgram(context.replaceStrings(MetalKernelSources::sort, replacements));
        } catch (OpenMMException& e) {
            fprintf(stderr, "[Metal] Sort kernel compilation failed (dataType=%s, keyType=%s, %zu elements)\n",
                    trait->getDataType(), trait->getKeyType(), dataLength);
            throw;
        }
        shortListKernel = program->createKernel("sortShortList");
        shortList2Kernel = program->createKernel("sortShortList2");
        computeRangeKernel = program->createKernel("computeRange");
        assignElementsKernel = program->createKernel(uniform ? "assignElementsToBuckets" : "assignElementsToBuckets2");
        computeBucketPositionsKernel = program->createKernel("computeBucketPositions");
        copyToBucketsKernel = program->createKernel("copyDataToBuckets");
        sortBucketsKernel = program->createKernel("sortBuckets");

        // Work out the work group sizes for various kernels.
        // On Metal/Apple Silicon, we use fixed reasonable defaults since Metal
        // doesn't expose CL_DEVICE_MAX_WORK_GROUP_SIZE or CL_DEVICE_LOCAL_MEM_SIZE
        // through the same API. Apple GPUs support up to 1024 threads per threadgroup
        // and have 32KB of threadgroup memory.

        unsigned int maxGroupSize = 256;  // Conservative default for Apple Silicon

        // Query actual max threadgroup size from compute pipeline states
        unsigned int maxRangeSize = std::min(maxGroupSize, (unsigned int)computeRangeKernel->getMaxBlockSize());
        unsigned int maxPositionsSize = std::min(maxGroupSize, (unsigned int)computeBucketPositionsKernel->getMaxBlockSize());

        // Apple Silicon has 32KB threadgroup memory
        int maxSharedMem = 32768;
        int maxLocalBuffer = (maxSharedMem/trait->getDataSize())/2;
        int maxShortList = max(maxLocalBuffer, (int) MetalContext::ThreadBlockSize*context.getNumThreadBlocks());

        // Apple Silicon is neither NVIDIA nor AMD, so use the conservative path.
        maxShortList = min(1024, maxShortList);
        useShortList2 = false;

        isShortList = (length <= (unsigned int)maxShortList);
        for (rangeKernelSize = 1; rangeKernelSize*2 <= maxRangeSize; rangeKernelSize *= 2)
            ;
        positionsKernelSize = std::min(rangeKernelSize, maxPositionsSize);
        sortKernelSize = (isShortList ? rangeKernelSize : rangeKernelSize/2);
        if (rangeKernelSize > length)
            rangeKernelSize = length;
        if (sortKernelSize > (unsigned int)maxLocalBuffer)
            sortKernelSize = maxLocalBuffer;
        unsigned int targetBucketSize = sortKernelSize/2;
        unsigned int numBuckets = length/targetBucketSize;
        if (numBuckets < 1)
            numBuckets = 1;
        if (positionsKernelSize > numBuckets)
            positionsKernelSize = numBuckets;

        // Create workspace arrays.

        dataRange.initialize(context, 2, trait->getKeySize(), "sortDataRange");
        bucketOffset.initialize<unsigned int>(context, numBuckets, "bucketOffset");
        bucketOfElement.initialize<unsigned int>(context, length, "bucketOfElement");
        offsetInBucket.initialize<unsigned int>(context, length, "offsetInBucket");
        buckets.initialize(context, length, trait->getDataSize(), "buckets");
    }
}

MetalSort::~MetalSort() {
    delete trait;
}

void MetalSort::sort(MetalArray& data) {
    if (data.getSize() != dataLength || data.getElementSize() != trait->getDataSize())
        throw OpenMMException("MetalSort called with different data size");
    if (data.getSize() == 0)
        return;
    if (isShortList) {
        // We can use a simpler sort kernel that does the entire operation in one kernel.

        try {
            if (useShortList2) {
                shortList2Kernel->setArg(0, data);
                shortList2Kernel->setArg(1, buckets);
                int lenArg = (int)dataLength;
                shortList2Kernel->setArg(2, lenArg);
                shortList2Kernel->execute(dataLength);
                buckets.copyTo(data);
            }
            else {
                shortListKernel->setArg(0, data);
                unsigned int lenArg = dataLength;
                shortListKernel->setArg(1, lenArg);
                dynamic_cast<MetalKernel&>(*shortListKernel).setThreadgroupMemoryArg(2, dataLength*trait->getDataSize());
                shortListKernel->execute(sortKernelSize, sortKernelSize);
            }
            return;
        }
        catch (exception& ex) {
            // This can happen if we chose too large a size for the kernel.  Switch
            // over to the standard sorting method.
            fprintf(stderr, "[Metal] Short-list sort failed (%s), falling back to standard sort\n", ex.what());
            isShortList = false;
        }
    }

    // Compute the range of data values.

    unsigned int numBuckets = bucketOffset.getSize();
    computeRangeKernel->setArg(0, data);
    unsigned int dataSizeArg = (unsigned int)data.getSize();
    computeRangeKernel->setArg(1, dataSizeArg);
    computeRangeKernel->setArg(2, dataRange);
    dynamic_cast<MetalKernel&>(*computeRangeKernel).setThreadgroupMemoryArg(3, rangeKernelSize*trait->getKeySize());
    dynamic_cast<MetalKernel&>(*computeRangeKernel).setThreadgroupMemoryArg(4, rangeKernelSize*trait->getKeySize());
    int numBucketsInt = (int)numBuckets;
    computeRangeKernel->setArg(5, numBucketsInt);
    computeRangeKernel->setArg(6, bucketOffset);
    computeRangeKernel->execute(rangeKernelSize, rangeKernelSize);

    // Assign array elements to buckets.

    assignElementsKernel->setArg(0, data);
    int dataSizeInt = (int)data.getSize();
    assignElementsKernel->setArg(1, dataSizeInt);
    assignElementsKernel->setArg(2, numBucketsInt);
    assignElementsKernel->setArg(3, dataRange);
    assignElementsKernel->setArg(4, bucketOffset);
    assignElementsKernel->setArg(5, bucketOfElement);
    assignElementsKernel->setArg(6, offsetInBucket);
    assignElementsKernel->execute(data.getSize());

    // Compute the position of each bucket.

    computeBucketPositionsKernel->setArg(0, numBucketsInt);
    computeBucketPositionsKernel->setArg(1, bucketOffset);
    dynamic_cast<MetalKernel&>(*computeBucketPositionsKernel).setThreadgroupMemoryArg(2, positionsKernelSize*sizeof(int));
    computeBucketPositionsKernel->execute(positionsKernelSize, positionsKernelSize);

    // Copy the data into the buckets.

    copyToBucketsKernel->setArg(0, data);
    copyToBucketsKernel->setArg(1, buckets);
    copyToBucketsKernel->setArg(2, dataSizeInt);
    copyToBucketsKernel->setArg(3, bucketOffset);
    copyToBucketsKernel->setArg(4, bucketOfElement);
    copyToBucketsKernel->setArg(5, offsetInBucket);
    copyToBucketsKernel->execute(data.getSize());

    // Sort each bucket.

    sortBucketsKernel->setArg(0, data);
    sortBucketsKernel->setArg(1, buckets);
    sortBucketsKernel->setArg(2, numBucketsInt);
    sortBucketsKernel->setArg(3, bucketOffset);
    dynamic_cast<MetalKernel&>(*sortBucketsKernel).setThreadgroupMemoryArg(4, sortKernelSize*trait->getDataSize());
    sortBucketsKernel->execute(((data.getSize()+sortKernelSize-1)/sortKernelSize)*sortKernelSize, sortKernelSize);
}
