/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2009-2023 Stanford University and the Authors.      *
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
#include "MetalFFT3D.h"
#include "MetalContext.h"
#include "MetalExpressionUtilities.h"
#include "MetalKernelSources.h"
#include "SimTKOpenMMRealType.h"
#include <algorithm>
#include <map>
#include <sstream>
#include <string>

using namespace OpenMM;
using namespace std;

#ifdef USE_VKFFT

MetalFFT3D::MetalFFT3D(MetalContext& context, int xsize, int ysize, int zsize, bool realToComplex) :
        context(context), xsize(xsize), ysize(ysize), zsize(zsize) {
    app = {};
    VkFFTConfiguration config = {};
    config.FFTdim = 3;
    config.size[0] = zsize;
    config.size[1] = ysize;
    config.size[2] = xsize;
    config.performR2C = realToComplex;
    config.doublePrecision = context.getUseDoublePrecision();

    // VkFFT Metal backend (VKFFT_BACKEND=5) expects MTL::Device* and MTL::CommandQueue*.
    // In Objective-C++, id<MTLDevice> and MTL::Device* are toll-free bridgeable
    // (same underlying pointer), so we reinterpret_cast the void* from MetalContext.
    id<MTLDevice> device = (__bridge id<MTLDevice>)context.getMTLDevice();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context.getMTLCommandQueue();
    config.device = reinterpret_cast<MTL::Device*>(device);
    config.queue = reinterpret_cast<MTL::CommandQueue*>(queue);

    config.inverseReturnToInputBuffer = true;
    config.isInputFormatted = 1;
    config.inputBufferStride[0] = zsize;
    config.inputBufferStride[1] = ysize*zsize;
    config.inputBufferStride[2] = xsize*ysize*zsize;
    VkFFTResult result = initializeVkFFT(&app, config);
    if (result != VKFFT_SUCCESS)
        throw OpenMMException("Error initializing VkFFT: "+context.intToString(result));
}

MetalFFT3D::~MetalFFT3D() {
    deleteVkFFT(&app);
}

void MetalFFT3D::execFFT(MetalArray& in, MetalArray& out, bool forward) {
    context.flushQueue();  // Ensure all pending compute work completes before VkFFT
    @autoreleasepool {
        VkFFTLaunchParams params = {};
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context.getMTLCommandQueue();
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        if (commandBuffer == nil)
            throw OpenMMException("Failed to create command buffer in execFFT");
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil)
            throw OpenMMException("Failed to create compute encoder in execFFT");

        id<MTLBuffer> inBuffer = (__bridge id<MTLBuffer>)in.getDeviceBuffer();
        id<MTLBuffer> outBuffer = (__bridge id<MTLBuffer>)out.getDeviceBuffer();

        // VkFFT Metal backend expects MTL::Buffer** and MTL::CommandBuffer*/MTL::ComputeCommandEncoder*
        MTL::Buffer* inBufCpp = reinterpret_cast<MTL::Buffer*>(inBuffer);
        MTL::Buffer* outBufCpp = reinterpret_cast<MTL::Buffer*>(outBuffer);

        if (forward) {
            params.inputBuffer = &inBufCpp;
            params.buffer = &outBufCpp;
        }
        else {
            params.inputBuffer = &outBufCpp;
            params.buffer = &inBufCpp;
        }
        params.commandBuffer = reinterpret_cast<MTL::CommandBuffer*>(commandBuffer);
        params.commandEncoder = reinterpret_cast<MTL::ComputeCommandEncoder*>(encoder);

        VkFFTResult result = VkFFTAppend(&app, forward ? -1 : 1, &params);
        if (result != VKFFT_SUCCESS) {
            fprintf(stderr, "[Metal] VkFFT %s failed with error code %d\n",
                    forward ? "forward" : "inverse", (int)result);
            throw OpenMMException("Error executing VkFFT: "+context.intToString(result));
        }

        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if ([commandBuffer status] == MTLCommandBufferStatusError) {
            NSError* error = [commandBuffer error];
            std::string errorMsg = "GPU FFT execution failed (" +
                std::string(forward ? "forward" : "inverse") + ")";
            if (error != nil)
                errorMsg += ": " + std::string([[error localizedDescription] UTF8String]);
            fprintf(stderr, "[Metal] %s\n", errorMsg.c_str());
            throw OpenMMException(errorMsg);
        }
    }
}

#else

MetalFFT3D::MetalFFT3D(MetalContext& context, int xsize, int ysize, int zsize, bool realToComplex) :
        context(context), xsize(xsize), ysize(ysize), zsize(zsize), r2cLocalMemSize(0) {
    packRealAsComplex = false;
    int packedXSize = xsize;
    int packedYSize = ysize;
    int packedZSize = zsize;
    if (realToComplex) {
        packRealAsComplex = true;
        int packedAxis, bufferSize;
        if (xsize%2 == 0) {
            packedAxis = 0;
            packedXSize /= 2;
            bufferSize = packedXSize;
        }
        else if (ysize%2 == 0) {
            packedAxis = 1;
            packedYSize /= 2;
            bufferSize = packedYSize;
        }
        else if (zsize%2 == 0) {
            packedAxis = 2;
            packedZSize /= 2;
            bufferSize = packedZSize;
        }
        else
            packRealAsComplex = false;
        if (packRealAsComplex) {
            r2cLocalMemSize = bufferSize * (context.getUseDoublePrecision() ? sizeof(mm_double2) : sizeof(mm_float2));
            map<string, string> defines;
            defines["XSIZE"] = context.intToString(xsize);
            defines["YSIZE"] = context.intToString(ysize);
            defines["ZSIZE"] = context.intToString(zsize);
            defines["PACKED_AXIS"] = context.intToString(packedAxis);
            defines["PACKED_XSIZE"] = context.intToString(packedXSize);
            defines["PACKED_YSIZE"] = context.intToString(packedYSize);
            defines["PACKED_ZSIZE"] = context.intToString(packedZSize);
            defines["M_PI"] = context.doubleToString(M_PI);
            // Create the Metal library for R2C pack/unpack kernels
            void* r2cLibrary = context.createProgram(MetalKernelSources::fftR2C, defines);
            @autoreleasepool {
                id<MTLLibrary> lib = (__bridge id<MTLLibrary>)r2cLibrary;
                id<MTLDevice> device = (__bridge id<MTLDevice>)context.getMTLDevice();
                NSError* error = nil;

                auto makePipeline = [&](const char* name) -> void* {
                    id<MTLFunction> fn = [lib newFunctionWithName:@(name)];
                    if (fn == nil)
                        throw OpenMMException(string("FFT R2C function not found: ") + name);
                    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
                    if (pso == nil)
                        throw OpenMMException(string("Failed to create R2C pipeline: ") + name);
                    void* result = (__bridge void*)pso;
                    CFRetain(result);
                    return result;
                };

                packForwardPipeline = makePipeline("packForwardData");
                unpackForwardPipeline = makePipeline("unpackForwardData");
                packBackwardPipeline = makePipeline("packBackwardData");
                unpackBackwardPipeline = makePipeline("unpackBackwardData");
            }
            CFRelease(r2cLibrary);
        }
    }
    bool inputIsReal = (realToComplex && !packRealAsComplex);
    zPipeline = createKernelPipeline(packedXSize, packedYSize, packedZSize, zthreads, 0, true, inputIsReal);
    xPipeline = createKernelPipeline(packedYSize, packedZSize, packedXSize, xthreads, 1, true, inputIsReal);
    yPipeline = createKernelPipeline(packedZSize, packedXSize, packedYSize, ythreads, 2, true, inputIsReal);
    invzPipeline = createKernelPipeline(packedXSize, packedYSize, packedZSize, zthreads, 0, false, inputIsReal);
    invxPipeline = createKernelPipeline(packedYSize, packedZSize, packedXSize, xthreads, 1, false, inputIsReal);
    invyPipeline = createKernelPipeline(packedZSize, packedXSize, packedYSize, ythreads, 2, false, inputIsReal);
}

MetalFFT3D::~MetalFFT3D() {
    auto releasePipeline = [](void*& p) { if (p) { CFRelease(p); p = NULL; } };
    releasePipeline(zPipeline);
    releasePipeline(xPipeline);
    releasePipeline(yPipeline);
    releasePipeline(invzPipeline);
    releasePipeline(invxPipeline);
    releasePipeline(invyPipeline);
    if (packRealAsComplex) {
        releasePipeline(packForwardPipeline);
        releasePipeline(unpackForwardPipeline);
        releasePipeline(packBackwardPipeline);
        releasePipeline(unpackBackwardPipeline);
    }
}

void MetalFFT3D::execFFT(MetalArray& in, MetalArray& out, bool forward) {
    void* pipeline1 = (forward ? zPipeline : invzPipeline);
    void* pipeline2 = (forward ? xPipeline : invxPipeline);
    void* pipeline3 = (forward ? yPipeline : invyPipeline);
    int elementSize = context.getUseDoublePrecision() ? sizeof(mm_double2) : sizeof(mm_float2);

    @autoreleasepool {
        if (packRealAsComplex) {
            int gridSize = xsize*ysize*zsize/2;
            void* packPipeline = (forward ? packForwardPipeline : packBackwardPipeline);
            void* unpackPipeline = (forward ? unpackForwardPipeline : unpackBackwardPipeline);

            // Pack: in → out
            // Forward: packForwardData has no threadgroup memory
            // Backward: packBackwardData has LOCAL real2* w at arg 2
            std::vector<MetalArray*> packArgs = {&in, &out};
            std::vector<std::vector<uint8_t>> emptyPrim;
            std::map<int, int> emptyTG;
            std::map<int, int> packTG;
            if (!forward)
                packTG[2] = r2cLocalMemSize;
            context.executeKernel(packPipeline, "packData", gridSize, -1, packArgs, emptyPrim, packTG);

            // FFT stages with buffer ping-pong (out→in, in→out, out→in)
            int zBuf = zthreads * elementSize;
            int xBuf = xthreads * elementSize;
            int yBuf = ythreads * elementSize;
            std::map<int, int> zTG = {{2, zBuf}, {3, zBuf}, {4, zBuf}};
            std::map<int, int> xTG = {{2, xBuf}, {3, xBuf}, {4, xBuf}};
            std::map<int, int> yTG = {{2, yBuf}, {3, yBuf}, {4, yBuf}};
            std::vector<MetalArray*> a1 = {&out, &in};
            context.executeKernel(pipeline1, "execFFT", gridSize, zthreads, a1, emptyPrim, zTG);
            std::vector<MetalArray*> a2 = {&in, &out};
            context.executeKernel(pipeline2, "execFFT", gridSize, xthreads, a2, emptyPrim, xTG);
            std::vector<MetalArray*> a3 = {&out, &in};
            context.executeKernel(pipeline3, "execFFT", gridSize, ythreads, a3, emptyPrim, yTG);

            // Unpack: in → out
            // Forward: unpackForwardData has LOCAL real2* w at arg 2
            // Backward: unpackBackwardData has no threadgroup memory
            std::vector<MetalArray*> unpackArgs = {&in, &out};
            std::map<int, int> unpackTG;
            if (forward)
                unpackTG[2] = r2cLocalMemSize;
            context.executeKernel(unpackPipeline, "unpackData", gridSize, -1, unpackArgs, emptyPrim, unpackTG);
        }
        else {
            int totalSize = xsize*ysize*zsize;
            int zBuf = zthreads * elementSize;
            int xBuf = xthreads * elementSize;
            int yBuf = ythreads * elementSize;
            std::map<int, int> zTG = {{2, zBuf}, {3, zBuf}, {4, zBuf}};
            std::map<int, int> xTG = {{2, xBuf}, {3, xBuf}, {4, xBuf}};
            std::map<int, int> yTG = {{2, yBuf}, {3, yBuf}, {4, yBuf}};
            std::vector<std::vector<uint8_t>> emptyPrim;

            // Stage 1: in → out
            std::vector<MetalArray*> a1 = {&in, &out};
            context.executeKernel(pipeline1, "execFFT", totalSize, zthreads, a1, emptyPrim, zTG);
            // Stage 2: out → in
            std::vector<MetalArray*> a2 = {&out, &in};
            context.executeKernel(pipeline2, "execFFT", totalSize, xthreads, a2, emptyPrim, xTG);
            // Stage 3: in → out
            std::vector<MetalArray*> a3 = {&in, &out};
            context.executeKernel(pipeline3, "execFFT", totalSize, ythreads, a3, emptyPrim, yTG);
        }
    }
}

void* MetalFFT3D::createKernelPipeline(int xsize, int ysize, int zsize, int& threads, int axis, bool forward, bool inputIsReal) {
    @autoreleasepool {
        int maxThreads = min(256, context.getMaxThreadBlockSize());
        while (maxThreads > 128 && maxThreads-64 >= zsize)
            maxThreads -= 64;
        while (true) {
            bool loopRequired = (zsize > maxThreads);
            stringstream source;
            int blocksPerGroup = (loopRequired ? 1 : max(1, maxThreads/zsize));
            int stage = 0;
            int L = zsize;
            int m = 1;

            // Factor zsize, generating an appropriate block of code for each factor.
            while (L > 1) {
                int input = stage%2;
                int output = 1-input;
                int radix;
                if (L%7 == 0)
                    radix = 7;
                else if (L%5 == 0)
                    radix = 5;
                else if (L%4 == 0)
                    radix = 4;
                else if (L%3 == 0)
                    radix = 3;
                else if (L%2 == 0)
                    radix = 2;
                else
                    throw OpenMMException("Illegal size for FFT: "+context.intToString(zsize));
                source<<"{\n";
                L = L/radix;
                source<<"// Pass "<<(stage+1)<<" (radix "<<radix<<")\n";
                if (loopRequired) {
                    source<<"for (int i = get_local_id(0); i < "<<(L*m)<<"; i += get_local_size(0)) {\n";
                    source<<"int base = i;\n";
                }
                else {
                    source<<"if (get_local_id(0) < "<<(blocksPerGroup*L*m)<<") {\n";
                    source<<"int block = get_local_id(0)/"<<(L*m)<<";\n";
                    source<<"int i = get_local_id(0)-block*"<<(L*m)<<";\n";
                    source<<"int base = i+block*"<<zsize<<";\n";
                }
                source<<"int j = i/"<<m<<";\n";
                // Generate FFT butterfly operations
                if (radix == 7) {
                    source<<"real2 c0 = data"<<input<<"[base];\n";
                    source<<"real2 c1 = data"<<input<<"[base+"<<(L*m)<<"];\n";
                    source<<"real2 c2 = data"<<input<<"[base+"<<(2*L*m)<<"];\n";
                    source<<"real2 c3 = data"<<input<<"[base+"<<(3*L*m)<<"];\n";
                    source<<"real2 c4 = data"<<input<<"[base+"<<(4*L*m)<<"];\n";
                    source<<"real2 c5 = data"<<input<<"[base+"<<(5*L*m)<<"];\n";
                    source<<"real2 c6 = data"<<input<<"[base+"<<(6*L*m)<<"];\n";
                    source<<"real2 d0 = c1+c6;\n";
                    source<<"real2 d1 = c1-c6;\n";
                    source<<"real2 d2 = c2+c5;\n";
                    source<<"real2 d3 = c2-c5;\n";
                    source<<"real2 d4 = c4+c3;\n";
                    source<<"real2 d5 = c4-c3;\n";
                    source<<"real2 d6 = d2+d0;\n";
                    source<<"real2 d7 = d5+d3;\n";
                    source<<"real2 b0 = c0+d6+d4;\n";
                    source<<"real2 b1 = "<<context.doubleToString((cos(2*M_PI/7)+cos(4*M_PI/7)+cos(6*M_PI/7))/3-1)<<"*(d6+d4);\n";
                    source<<"real2 b2 = "<<context.doubleToString((2*cos(2*M_PI/7)-cos(4*M_PI/7)-cos(6*M_PI/7))/3)<<"*(d0-d4);\n";
                    source<<"real2 b3 = "<<context.doubleToString((cos(2*M_PI/7)-2*cos(4*M_PI/7)+cos(6*M_PI/7))/3)<<"*(d4-d2);\n";
                    source<<"real2 b4 = "<<context.doubleToString((cos(2*M_PI/7)+cos(4*M_PI/7)-2*cos(6*M_PI/7))/3)<<"*(d2-d0);\n";
                    source<<"real2 b5 = -(SIGN)*"<<context.doubleToString((sin(2*M_PI/7)+sin(4*M_PI/7)-sin(6*M_PI/7))/3)<<"*(d7+d1);\n";
                    source<<"real2 b6 = -(SIGN)*"<<context.doubleToString((2*sin(2*M_PI/7)-sin(4*M_PI/7)+sin(6*M_PI/7))/3)<<"*(d1-d5);\n";
                    source<<"real2 b7 = -(SIGN)*"<<context.doubleToString((sin(2*M_PI/7)-2*sin(4*M_PI/7)-sin(6*M_PI/7))/3)<<"*(d5-d3);\n";
                    source<<"real2 b8 = -(SIGN)*"<<context.doubleToString((sin(2*M_PI/7)+sin(4*M_PI/7)+2*sin(6*M_PI/7))/3)<<"*(d3-d1);\n";
                    source<<"real2 t0 = b0+b1;\n";
                    source<<"real2 t1 = b2+b3;\n";
                    source<<"real2 t2 = b4-b3;\n";
                    source<<"real2 t3 = -b2-b4;\n";
                    source<<"real2 t4 = b6+b7;\n";
                    source<<"real2 t5 = b8-b7;\n";
                    source<<"real2 t6 = -b8-b6;\n";
                    source<<"real2 t7 = t0+t1;\n";
                    source<<"real2 t8 = t0+t2;\n";
                    source<<"real2 t9 = t0+t3;\n";
                    source<<"real2 t10 = real2(t4.y+b5.y, -(t4.x+b5.x));\n";
                    source<<"real2 t11 = real2(t5.y+b5.y, -(t5.x+b5.x));\n";
                    source<<"real2 t12 = real2(t6.y+b5.y, -(t6.x+b5.x));\n";
                    source<<"data"<<output<<"[base+6*j*"<<m<<"] = b0;\n";
                    source<<"data"<<output<<"[base+(6*j+1)*"<<m<<"] = multiplyComplex(w[j*"<<zsize<<"/"<<(7*L)<<"], t7-t10);\n";
                    source<<"data"<<output<<"[base+(6*j+2)*"<<m<<"] = multiplyComplex(w[j*"<<(2*zsize)<<"/"<<(7*L)<<"], t9-t12);\n";
                    source<<"data"<<output<<"[base+(6*j+3)*"<<m<<"] = multiplyComplex(w[j*"<<(3*zsize)<<"/"<<(7*L)<<"], t8+t11);\n";
                    source<<"data"<<output<<"[base+(6*j+4)*"<<m<<"] = multiplyComplex(w[j*"<<(4*zsize)<<"/"<<(7*L)<<"], t8-t11);\n";
                    source<<"data"<<output<<"[base+(6*j+5)*"<<m<<"] = multiplyComplex(w[j*"<<(5*zsize)<<"/"<<(7*L)<<"], t9+t12);\n";
                    source<<"data"<<output<<"[base+(6*j+6)*"<<m<<"] = multiplyComplex(w[j*"<<(6*zsize)<<"/"<<(7*L)<<"], t7+t10);\n";
                }
                else if (radix == 5) {
                    source<<"real2 c0 = data"<<input<<"[base];\n";
                    source<<"real2 c1 = data"<<input<<"[base+"<<(L*m)<<"];\n";
                    source<<"real2 c2 = data"<<input<<"[base+"<<(2*L*m)<<"];\n";
                    source<<"real2 c3 = data"<<input<<"[base+"<<(3*L*m)<<"];\n";
                    source<<"real2 c4 = data"<<input<<"[base+"<<(4*L*m)<<"];\n";
                    source<<"real2 d0 = c1+c4;\n";
                    source<<"real2 d1 = c2+c3;\n";
                    source<<"real2 d2 = "<<context.doubleToString(sin(0.4*M_PI))<<"*(c1-c4);\n";
                    source<<"real2 d3 = "<<context.doubleToString(sin(0.4*M_PI))<<"*(c2-c3);\n";
                    source<<"real2 d4 = d0+d1;\n";
                    source<<"real2 d5 = "<<context.doubleToString(0.25*sqrt(5.0))<<"*(d0-d1);\n";
                    source<<"real2 d6 = c0-0.25f*d4;\n";
                    source<<"real2 d7 = d6+d5;\n";
                    source<<"real2 d8 = d6-d5;\n";
                    string coeff = context.doubleToString(sin(0.2*M_PI)/sin(0.4*M_PI));
                    source<<"real2 d9 = (SIGN)*real2(d2.y+"<<coeff<<"*d3.y, -d2.x-"<<coeff<<"*d3.x);\n";
                    source<<"real2 d10 = (SIGN)*real2("<<coeff<<"*d2.y-d3.y, d3.x-"<<coeff<<"*d2.x);\n";
                    source<<"data"<<output<<"[base+4*j*"<<m<<"] = c0+d4;\n";
                    source<<"data"<<output<<"[base+(4*j+1)*"<<m<<"] = multiplyComplex(w[j*"<<zsize<<"/"<<(5*L)<<"], d7+d9);\n";
                    source<<"data"<<output<<"[base+(4*j+2)*"<<m<<"] = multiplyComplex(w[j*"<<(2*zsize)<<"/"<<(5*L)<<"], d8+d10);\n";
                    source<<"data"<<output<<"[base+(4*j+3)*"<<m<<"] = multiplyComplex(w[j*"<<(3*zsize)<<"/"<<(5*L)<<"], d8-d10);\n";
                    source<<"data"<<output<<"[base+(4*j+4)*"<<m<<"] = multiplyComplex(w[j*"<<(4*zsize)<<"/"<<(5*L)<<"], d7-d9);\n";
                }
                else if (radix == 4) {
                    source<<"real2 c0 = data"<<input<<"[base];\n";
                    source<<"real2 c1 = data"<<input<<"[base+"<<(L*m)<<"];\n";
                    source<<"real2 c2 = data"<<input<<"[base+"<<(2*L*m)<<"];\n";
                    source<<"real2 c3 = data"<<input<<"[base+"<<(3*L*m)<<"];\n";
                    source<<"real2 d0 = c0+c2;\n";
                    source<<"real2 d1 = c0-c2;\n";
                    source<<"real2 d2 = c1+c3;\n";
                    source<<"real2 d3 = (SIGN)*real2(c1.y-c3.y, c3.x-c1.x);\n";
                    source<<"data"<<output<<"[base+3*j*"<<m<<"] = d0+d2;\n";
                    source<<"data"<<output<<"[base+(3*j+1)*"<<m<<"] = multiplyComplex(w[j*"<<zsize<<"/"<<(4*L)<<"], d1+d3);\n";
                    source<<"data"<<output<<"[base+(3*j+2)*"<<m<<"] = multiplyComplex(w[j*"<<(2*zsize)<<"/"<<(4*L)<<"], d0-d2);\n";
                    source<<"data"<<output<<"[base+(3*j+3)*"<<m<<"] = multiplyComplex(w[j*"<<(3*zsize)<<"/"<<(4*L)<<"], d1-d3);\n";
                }
                else if (radix == 3) {
                    source<<"real2 c0 = data"<<input<<"[base];\n";
                    source<<"real2 c1 = data"<<input<<"[base+"<<(L*m)<<"];\n";
                    source<<"real2 c2 = data"<<input<<"[base+"<<(2*L*m)<<"];\n";
                    source<<"real2 d0 = c1+c2;\n";
                    source<<"real2 d1 = c0-0.5f*d0;\n";
                    source<<"real2 d2 = (SIGN)*"<<context.doubleToString(sin(M_PI/3.0))<<"*real2(c1.y-c2.y, c2.x-c1.x);\n";
                    source<<"data"<<output<<"[base+2*j*"<<m<<"] = c0+d0;\n";
                    source<<"data"<<output<<"[base+(2*j+1)*"<<m<<"] = multiplyComplex(w[j*"<<zsize<<"/"<<(3*L)<<"], d1+d2);\n";
                    source<<"data"<<output<<"[base+(2*j+2)*"<<m<<"] = multiplyComplex(w[j*"<<(2*zsize)<<"/"<<(3*L)<<"], d1-d2);\n";
                }
                else if (radix == 2) {
                    source<<"real2 c0 = data"<<input<<"[base];\n";
                    source<<"real2 c1 = data"<<input<<"[base+"<<(L*m)<<"];\n";
                    source<<"data"<<output<<"[base+j*"<<m<<"] = c0+c1;\n";
                    source<<"data"<<output<<"[base+(j+1)*"<<m<<"] = multiplyComplex(w[j*"<<zsize<<"/"<<(2*L)<<"], c0-c1);\n";
                }
                source<<"}\n";
                m = m*radix;
                source<<"barrier(CLK_LOCAL_MEM_FENCE);\n";
                source<<"}\n";
                ++stage;
            }

            // Create the kernel.
            bool outputIsReal = (inputIsReal && axis == 2 && !forward);
            bool outputIsPacked = (inputIsReal && axis == 2 && forward);
            string outputSuffix = (outputIsReal ? ".x" : "");
            if (loopRequired) {
                if (outputIsPacked)
                    source<<"if (x < XSIZE/2+1)\n";
                source<<"for (int z = get_local_id(0); z < ZSIZE; z += get_local_size(0))\n";
                if (outputIsPacked)
                    source<<"out[y*(ZSIZE*(XSIZE/2+1))+z*(XSIZE/2+1)+x] = data"<<(stage%2)<<"[z]"<<outputSuffix<<";\n";
                else
                    source<<"out[y*(ZSIZE*XSIZE)+z*XSIZE+x] = data"<<(stage%2)<<"[z]"<<outputSuffix<<";\n";
            }
            else {
                if (outputIsPacked) {
                    source<<"if (index < XSIZE*YSIZE && x < XSIZE/2+1)\n";
                    source<<"out[y*(ZSIZE*(XSIZE/2+1))+(get_local_id(0)%ZSIZE)*(XSIZE/2+1)+x] = data"<<(stage%2)<<"[get_local_id(0)]"<<outputSuffix<<";\n";
                }
                else {
                    source<<"if (index < XSIZE*YSIZE)\n";
                    source<<"out[y*(ZSIZE*XSIZE)+(get_local_id(0)%ZSIZE)*XSIZE+x] = data"<<(stage%2)<<"[get_local_id(0)]"<<outputSuffix<<";\n";
                }
            }
            map<string, string> replacements;
            replacements["XSIZE"] = context.intToString(xsize);
            replacements["YSIZE"] = context.intToString(ysize);
            replacements["ZSIZE"] = context.intToString(zsize);
            replacements["BLOCKS_PER_GROUP"] = context.intToString(blocksPerGroup);
            replacements["M_PI"] = context.doubleToString(M_PI);
            replacements["COMPUTE_FFT"] = source.str();
            replacements["LOOP_REQUIRED"] = (loopRequired ? "1" : "0");
            replacements["SIGN"] = (forward ? "1" : "-1");
            replacements["INPUT_TYPE"] = (inputIsReal && axis == 0 && forward ? "real" : "real2");
            replacements["OUTPUT_TYPE"] = (outputIsReal ? "real" : "real2");
            replacements["INPUT_IS_REAL"] = (inputIsReal && axis == 0 && forward ? "1" : "0");
            replacements["INPUT_IS_PACKED"] = (inputIsReal && axis == 0 && !forward ? "1" : "0");
            replacements["OUTPUT_IS_PACKED"] = (outputIsPacked ? "1" : "0");
            void* library = context.createProgram(context.replaceStrings(MetalKernelSources::fft, replacements));
            threads = blocksPerGroup*zsize;
            // Create pipeline state from library
            id<MTLLibrary> mtlLibrary = (__bridge id<MTLLibrary>)library;
            id<MTLFunction> function = [mtlLibrary newFunctionWithName:@"execFFT"];
            if (function == nil) {
                CFRelease(library);
                throw OpenMMException("Failed to find execFFT function in Metal library");
            }
            NSError* error = nil;
            id<MTLDevice> device = (__bridge id<MTLDevice>)context.getMTLDevice();
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (pipeline == nil || error != nil) {
                std::string errorMsg = "Failed to create FFT pipeline state";
                if (error != nil)
                    errorMsg += ": " + std::string([[error localizedDescription] UTF8String]);
                fprintf(stderr, "[Metal] %s\n", errorMsg.c_str());
                CFRelease(library);
                throw OpenMMException(errorMsg);
            }
            int kernelMaxThreads = (int)[pipeline maxTotalThreadsPerThreadgroup];
            CFRelease(library);
            if (threads > kernelMaxThreads) {
                maxThreads = kernelMaxThreads;
                continue;
            }
            void* result = (__bridge void*)pipeline;
            CFRetain(result);
            return result;
        }
    }
}

#endif

int MetalFFT3D::findLegalDimension(int minimum) {
    if (minimum < 1)
        return 1;
#ifdef USE_VKFFT
    const int maxFactor = 13;
#else
    const int maxFactor = 7;
#endif
    while (true) {
        int unfactored = minimum;
        for (int factor = 2; factor <= maxFactor; factor++) {
            while (unfactored > 1 && unfactored%factor == 0)
                unfactored /= factor;
        }
        if (unfactored == 1)
            return minimum;
        minimum++;
    }
}
