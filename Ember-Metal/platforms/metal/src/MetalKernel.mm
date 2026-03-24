/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2019 Stanford University and the Authors.           *
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
#include "MetalKernel.h"
#include "MetalArray.h"
#include "MetalContext.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/internal/AssertionUtilities.h"
#include <cstring>

using namespace OpenMM;
using namespace std;

MetalKernel::MetalKernel(MetalContext& context, void* pipelineState, const std::string& name) :
        context(context), pipelineState(pipelineState), kernelName(name) {
}

MetalKernel::~MetalKernel() {
    if (pipelineState != NULL) {
        CFRelease(pipelineState);
        pipelineState = NULL;
    }
}

string MetalKernel::getName() const {
    return kernelName;
}

int MetalKernel::getMaxBlockSize() const {
    @autoreleasepool {
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pipelineState;
        return (int)[pso maxTotalThreadsPerThreadgroup];
    }
}

void MetalKernel::execute(int threads, int blockSize) {
    // Build the primitive args vector indexed by position (matching arrayArgs size)
    std::vector<std::vector<uint8_t>> primArgs(arrayArgs.size());
    for (auto& pair : primitiveArgValues) {
        if (pair.first < (int)primArgs.size())
            primArgs[pair.first] = pair.second;
    }
    context.executeKernel(pipelineState, kernelName, threads, blockSize,
                          arrayArgs, primArgs, threadgroupMemorySizes);
}

void MetalKernel::addArrayArg(ArrayInterface& value) {
    int index = arrayArgs.size();
    addEmptyArg();
    setArrayArg(index, value);
}

void MetalKernel::addPrimitiveArg(const void* value, int size) {
    int index = arrayArgs.size();
    addEmptyArg();
    setPrimitiveArg(index, value, size);
}

void MetalKernel::addEmptyArg() {
    arrayArgs.push_back(NULL);
}

void MetalKernel::setArrayArg(int index, ArrayInterface& value) {
    // Auto-grow arrayArgs if needed (OpenCL allows setting args at any index)
    while (index >= (int)arrayArgs.size())
        arrayArgs.push_back(NULL);
    arrayArgs[index] = &context.unwrap(value);
    primitiveArgValues.erase(index);
    threadgroupMemorySizes.erase(index);
}

void MetalKernel::setPrimitiveArg(int index, const void* value, int size) {
    // Auto-grow arrayArgs if needed (OpenCL allows setting args at any index)
    while (index >= (int)arrayArgs.size())
        arrayArgs.push_back(NULL);
    arrayArgs[index] = NULL;
    vector<uint8_t> bytes(size);
    memcpy(bytes.data(), value, size);
    primitiveArgValues[index] = std::move(bytes);
    threadgroupMemorySizes.erase(index);
}

void MetalKernel::addThreadgroupMemoryArg(int size) {
    int index = arrayArgs.size();
    addEmptyArg();
    threadgroupMemorySizes[index] = size;
}

void MetalKernel::setThreadgroupMemoryArg(int index, int size) {
    // Auto-grow arrayArgs if needed
    while (index >= (int)arrayArgs.size())
        arrayArgs.push_back(NULL);
    threadgroupMemorySizes[index] = size;
    arrayArgs[index] = NULL;
    primitiveArgValues.erase(index);
}
