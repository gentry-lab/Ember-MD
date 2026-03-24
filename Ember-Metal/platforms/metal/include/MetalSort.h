#ifndef __OPENMM_OPENCLSORT_H__
#define __OPENMM_OPENCLSORT_H__

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

#include "MetalArray.h"
#include "MetalContext.h"
#include "openmm/common/ComputeProgram.h"
#include "openmm/common/windowsExportCommon.h"

namespace OpenMM {

/**
 * This class sorts arrays of values.
 */

class OPENMM_EXPORT_COMMON MetalSort {
public:
    class SortTrait;
    MetalSort(MetalContext& context, SortTrait* trait, unsigned int length, bool uniform=true);
    ~MetalSort();
    void sort(MetalArray& data);
private:
    MetalContext& context;
    SortTrait* trait;
    MetalArray dataRange;
    MetalArray bucketOfElement;
    MetalArray offsetInBucket;
    MetalArray bucketOffset;
    MetalArray buckets;
    ComputeKernel shortListKernel, shortList2Kernel, computeRangeKernel, assignElementsKernel, computeBucketPositionsKernel, copyToBucketsKernel, sortBucketsKernel;
    unsigned int dataLength, rangeKernelSize, positionsKernelSize, sortKernelSize;
    bool isShortList, useShortList2, uniform;
};

/**
 * A subclass of SortTrait defines the type of value to sort, and the key for sorting them.
 */
class MetalSort::SortTrait {
public:
    virtual ~SortTrait() {
    }
    virtual int getDataSize() const = 0;
    virtual int getKeySize() const = 0;
    virtual const char* getDataType() const = 0;
    virtual const char* getKeyType() const = 0;
    virtual const char* getMinKey() const = 0;
    virtual const char* getMaxKey() const = 0;
    virtual const char* getMaxValue() const = 0;
    virtual const char* getSortKey() const = 0;
};

} // namespace OpenMM

#endif // __OPENMM_OPENCLSORT_H__
