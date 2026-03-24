/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2012-2022 Stanford University and the Authors.      *
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
#include "MetalArray.h"
#include "MetalContext.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>

using namespace OpenMM;

MetalArray::MetalArray() : buffer(NULL), ownsBuffer(false) {
}

MetalArray::MetalArray(MetalContext& context, size_t size, int elementSize, const std::string& name, int flags) : buffer(NULL) {
    initialize(context, size, elementSize, name, flags);
}

MetalArray::MetalArray(MetalContext& context, void* existingBuffer, size_t size, int elementSize, const std::string& name) : buffer(NULL) {
    initialize(context, existingBuffer, size, elementSize, name);
}

MetalArray::~MetalArray() {
    if (buffer != NULL && ownsBuffer) {
        CFRelease(buffer);
        buffer = NULL;
    }
}

void MetalArray::initialize(ComputeContext& context, size_t size, int elementSize, const std::string& name) {
    initialize(dynamic_cast<MetalContext&>(context), size, elementSize, name, 0);
}

void MetalArray::initialize(MetalContext& context, size_t size, int elementSize, const std::string& name, int flags) {
    if (buffer != NULL)
        throw OpenMMException("MetalArray has already been initialized");
    this->context = &context;
    this->size = size;
    this->elementSize = elementSize;
    this->name = name;
    this->flags = flags;
    ownsBuffer = true;
    @autoreleasepool {
        try {
            id<MTLDevice> device = (__bridge id<MTLDevice>)context.getMTLDevice();
            NSUInteger bufferLength = (NSUInteger)(size * elementSize);
            if (bufferLength == 0)
                bufferLength = 4; // Metal does not allow zero-length buffers
            id<MTLBuffer> mtlBuffer = [device newBufferWithLength:bufferLength
                                                         options:MTLResourceStorageModeShared];
            if (mtlBuffer == nil) {
                std::stringstream str;
                str << "Error creating array " << name << ": Failed to allocate Metal buffer of size " << bufferLength;
                throw OpenMMException(str.str());
            }
            // Zero-initialize the buffer contents
            void* contents = [mtlBuffer contents];
            if (contents == nil) {
                std::stringstream str;
                str << "Error creating array " << name << ": Metal buffer contents pointer is nil (size " << bufferLength << ")";
                fprintf(stderr, "[Metal] %s\n", str.str().c_str());
                throw OpenMMException(str.str());
            }
            memset(contents, 0, bufferLength);
            // Store as void* and retain for our ownership
            buffer = (__bridge void*)mtlBuffer;
            CFRetain(buffer);
        }
        catch (OpenMMException&) {
            throw;
        }
        catch (...) {
            std::stringstream str;
            str << "Error creating array " << name << ": Unknown error during Metal buffer allocation";
            throw OpenMMException(str.str());
        }
    }
}

void MetalArray::initialize(MetalContext& context, void* existingBuffer, size_t size, int elementSize, const std::string& name) {
    if (this->buffer != NULL)
        throw OpenMMException("MetalArray has already been initialized");
    this->context = &context;
    this->buffer = existingBuffer;
    this->size = size;
    this->elementSize = elementSize;
    this->name = name;
    ownsBuffer = false;
}

void MetalArray::resize(size_t size) {
    if (buffer == NULL)
        throw OpenMMException("MetalArray has not been initialized");
    if (!ownsBuffer)
        throw OpenMMException("Cannot resize an array that does not own its storage");
    // Release old buffer
    CFRelease(buffer);
    buffer = NULL;
    initialize(*context, size, elementSize, name, flags);
}

ComputeContext& MetalArray::getContext() {
    return *context;
}

void MetalArray::uploadSubArray(const void* data, int offset, int elements, bool blocking) {
    if (buffer == NULL)
        throw OpenMMException("MetalArray has not been initialized");
    if (offset < 0 || offset + elements > (int)getSize())
        throw OpenMMException("uploadSubArray: data exceeds range of array");
    @autoreleasepool {
        try {
            id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
            // With shared storage mode, CPU and GPU share the same memory.
            // A simple memcpy is sufficient for data transfer.
            void* contents = [mtlBuffer contents];
            if (contents == nil) {
                std::stringstream str;
                str << "Error uploading array " << name << ": buffer contents pointer is nil";
                fprintf(stderr, "[Metal] %s\n", str.str().c_str());
                throw OpenMMException(str.str());
            }
            uint8_t* dst = (uint8_t*)contents + (size_t)offset * elementSize;
            memcpy(dst, data, (size_t)elements * elementSize);
        }
        catch (OpenMMException&) {
            throw;
        }
        catch (...) {
            std::stringstream str;
            str << "Error uploading array " << name;
            throw OpenMMException(str.str());
        }
    }
}

void MetalArray::download(void* data, bool blocking) const {
    if (buffer == NULL)
        throw OpenMMException("MetalArray has not been initialized");
    context->flushQueue();  // Ensure all pending GPU writes are complete before CPU read
    @autoreleasepool {
        try {
            id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
            // With shared storage mode, we can directly memcpy from the buffer contents.
            void* contents = [mtlBuffer contents];
            if (contents == nil) {
                std::stringstream str;
                str << "Error downloading array " << name << ": buffer contents pointer is nil";
                fprintf(stderr, "[Metal] %s\n", str.str().c_str());
                throw OpenMMException(str.str());
            }
            if (data == nil) {
                std::stringstream str;
                str << "Error downloading array " << name << ": destination pointer is nil";
                fprintf(stderr, "[Metal] %s\n", str.str().c_str());
                throw OpenMMException(str.str());
            }
            memcpy(data, contents, size * elementSize);
        }
        catch (OpenMMException&) {
            throw;
        }
        catch (...) {
            std::stringstream str;
            str << "Error downloading array " << name;
            throw OpenMMException(str.str());
        }
    }
}

void MetalArray::copyTo(ArrayInterface& dest) const {
    if (buffer == NULL)
        throw OpenMMException("MetalArray has not been initialized");
    if (dest.getSize() != size || dest.getElementSize() != elementSize)
        throw OpenMMException("Error copying array " + name + " to " + dest.getName() + ": The destination array does not match the size of the array");
    // Fold blit into persistent command buffer — no flush/wait needed.
    MetalArray& mtlDest = context->unwrap(dest);
    @autoreleasepool {
        try {
            id<MTLBuffer> srcBuffer = (__bridge id<MTLBuffer>)buffer;
            id<MTLBuffer> dstBuffer = (__bridge id<MTLBuffer>)mtlDest.getDeviceBuffer();
            id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)context->getCommandBufferForBlit();
            id<MTLBlitCommandEncoder> blitEncoder = [cb blitCommandEncoder];
            if (blitEncoder == nil)
                throw OpenMMException("Failed to create blit encoder for copyTo: " + name);
            [blitEncoder copyFromBuffer:srcBuffer
                           sourceOffset:0
                               toBuffer:dstBuffer
                      destinationOffset:0
                                   size:(NSUInteger)(size * elementSize)];
            [blitEncoder endEncoding];
        }
        catch (OpenMMException&) {
            throw;
        }
        catch (...) {
            std::stringstream str;
            str << "Error copying array " << name << " to " << dest.getName();
            throw OpenMMException(str.str());
        }
    }
}
