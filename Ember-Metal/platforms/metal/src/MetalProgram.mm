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
#include "MetalProgram.h"
#include "MetalKernel.h"
#include "openmm/OpenMMException.h"

using namespace OpenMM;
using namespace std;

MetalProgram::MetalProgram(MetalContext& context, void* mtlLibrary) : context(context), mtlLibrary(mtlLibrary) {
    // mtlLibrary is a retained id<MTLLibrary> stored as void*
}

MetalProgram::~MetalProgram() {
    if (mtlLibrary != NULL) {
        CFRelease(mtlLibrary);
        mtlLibrary = NULL;
    }
}

ComputeKernel MetalProgram::createKernel(const string& name) {
    @autoreleasepool {
        id<MTLLibrary> library = (__bridge id<MTLLibrary>)mtlLibrary;

        // Extract the function from the library
        id<MTLFunction> function = [library newFunctionWithName:@(name.c_str())];
        if (function == nil) {
            // List available functions for debugging
            NSArray<NSString*>* names = [library functionNames];
            string available = "";
            for (NSString* n in names) {
                if (available.size() > 0) available += ", ";
                available += [n UTF8String];
            }
            fprintf(stderr, "[Metal] Kernel '%s' NOT FOUND in library. Available: [%s]\n",
                    name.c_str(), available.c_str());
            throw OpenMMException("Kernel function not found in Metal library: " + name +
                                  ". Available: [" + available + "]");
        }

        // Create a compute pipeline state from the function
        NSError* error = nil;
        id<MTLDevice> device = (__bridge id<MTLDevice>)context.getMTLDevice();
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:function error:&error];
        if (pso == nil || error != nil) {
            string errorMsg = "Error creating compute pipeline state for kernel '" + name + "'";
            if (error != nil) {
                errorMsg += ": " + string([[error localizedDescription] UTF8String]);
                if ([error localizedFailureReason] != nil)
                    errorMsg += "\nReason: " + string([[error localizedFailureReason] UTF8String]);
            }
            fprintf(stderr, "[Metal] Pipeline creation FAILED for '%s': %s\n", name.c_str(), errorMsg.c_str());
            throw OpenMMException(errorMsg);
        }

        // Store pipeline state as void* with a retain for MetalKernel ownership.
        void* psoPtr = (__bridge void*)pso;
        CFRetain(psoPtr);
        return shared_ptr<ComputeKernelImpl>(new MetalKernel(context, psoPtr, name));
    }
}
