/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2009-2020 Stanford University and the Authors.      *
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
#include <cmath>
#include "MetalContext.h"
#include "MetalArray.h"
#include "MetalBondedUtilities.h"
#include "MetalEvent.h"
#include "MetalForceInfo.h"
#include "MetalIntegrationUtilities.h"
#include "MetalKernelSources.h"
#include "MetalNonbondedUtilities.h"
#include "MetalProgram.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/VirtualSite.h"
#include "openmm/internal/ContextImpl.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <typeinfo>
#include <cstring>

using namespace OpenMM;
using namespace std;

const int MetalContext::ThreadBlockSize = 128;
const int MetalContext::TileSize = 32;

// https://stackoverflow.com/a/9753581
inline bool is_valid_int(const char *str) {
   // Handle negative numbers.
  if (*str == '-') {
    ++str;
  }

   // Handle empty string or just "-".
  if (!*str) {
    return false;
  }

   // Check for non-digit chars in the rest of the string.
   while (*str) {
     if (!isdigit(*str)) {
       return false;
     } else {
       ++str;
     }
   }

   return true;
}

MetalContext::MetalContext(const System& system, int platformIndex, int deviceIndex, const string& precision, MetalPlatform::PlatformData& platformData, MetalContext* originalContext) :
        ComputeContext(system), platformData(platformData), numForceBuffers(0), enableKernelProfiling(false), hasAssignedPosqCharges(false),
        integration(NULL), expression(NULL), bonded(NULL), nonbonded(NULL), pinnedBuffer(NULL), mtlDevice(NULL), mtlCommandQueue(NULL), profileStartTime(0) {

    char *optionProfileKernels = getenv("OPENMM_METAL_PROFILE_KERNELS");
    if (optionProfileKernels != nullptr) {
      if (strcmp(optionProfileKernels, "0") == 0) {
        this->enableKernelProfiling = false;
      } else if (strcmp(optionProfileKernels, "1") == 0) {
        this->enableKernelProfiling = true;
      } else {
        std::cout << std::endl;
        std::cout << METAL_LOG_HEADER << "Error: Invalid option for ";
        std::cout << "'OPENMM_METAL_PROFILE_KERNELS'." << std::endl;
        std::cout << METAL_LOG_HEADER << "Specified '" << optionProfileKernels << "', but ";
        std::cout << "expected either '0' or '1'." << std::endl;
        std::cout << METAL_LOG_HEADER << "Quitting now." << std::endl;
        exit(7);
      }
    }

    char *optionReduceEnergyThreadgroups = getenv("OPENMM_METAL_REDUCE_ENERGY_THREADGROUPS");
    if (optionReduceEnergyThreadgroups != nullptr) {
        bool fail = true;
        if (is_valid_int(optionReduceEnergyThreadgroups)) {
            int numThreadgroups = atoi(optionReduceEnergyThreadgroups);
            if (numThreadgroups >= 1 && numThreadgroups <= 1024) {
                fail = false;
                this->reduceEnergyThreadgroups = numThreadgroups;
            }
        }
        if (fail) {
            std::cout << std::endl;
            std::cout << METAL_LOG_HEADER << "Error: Invalid option for ";
            std::cout << "'OPENMM_METAL_REDUCE_ENERGY_THREADGROUPS'." << std::endl;
            std::cout << METAL_LOG_HEADER << "Specified '" << optionReduceEnergyThreadgroups << "', but ";
            std::cout << "expected a number between '1' and '1024'." << std::endl;
            std::cout << METAL_LOG_HEADER << "Quitting now." << std::endl;
            exit(9);
        }
    } else {
        this->reduceEnergyThreadgroups = 1024;
    }

    char *optionDumpSource = getenv("OPENMM_METAL_DUMP_SOURCE");
    this->enableDumpSource = (optionDumpSource != nullptr && strcmp(optionDumpSource, "1") == 0);
    this->dumpSourceCounter = 0;

    char *optionLogArgs = getenv("OPENMM_METAL_LOG_ARGS");
    this->enableLogArgs = (optionLogArgs != nullptr && strcmp(optionLogArgs, "1") == 0);

    // Command buffer batching state
    this->currentCommandBuffer = NULL;
    this->currentComputeEncoder = NULL;

    if (precision == "single") {
        useDoublePrecision = false;
        useMixedPrecision = false;
    }
    else if (precision == "mixed") {
        useDoublePrecision = false;
        useMixedPrecision = true;
    }
    else if (precision == "double") {
        useDoublePrecision = true;
        useMixedPrecision = false;
    }
    else
        throw OpenMMException("Illegal value for Precision: "+precision);

    @autoreleasepool {
        try {
            contextIndex = platformData.contexts.size();

            // ---- Metal Device Selection ----
            // Replace all OpenCL platform enumeration with Metal device discovery.
            // On Apple Silicon, there is typically one Metal GPU device.

            id<MTLDevice> selectedDevice = nil;

            if (deviceIndex == -1) {
                // Use the system default Metal device (best GPU)
                selectedDevice = MTLCreateSystemDefaultDevice();
            } else {
                // Support explicit device selection via index
                NSArray<id<MTLDevice>>* allDevices = MTLCopyAllDevices();
                if (allDevices == nil || [allDevices count] == 0) {
                    throw OpenMMException("No Metal devices available");
                }
                if (deviceIndex < 0 || deviceIndex >= (int)[allDevices count]) {
                    throw OpenMMException("Illegal value for DeviceIndex: " + intToString(deviceIndex));
                }
                selectedDevice = allDevices[deviceIndex];
            }

            if (selectedDevice == nil) {
                throw OpenMMException("No compatible Metal device is available");
            }

            // Store the device name and capabilities for logging
            string deviceName = string([[selectedDevice name] UTF8String]);
            std::cout << METAL_LOG_HEADER << "Using Metal device: " << deviceName << std::endl;
            fprintf(stderr, "[Metal] Device: %s\n", deviceName.c_str());
            fprintf(stderr, "[Metal]   Max buffer length: %llu MB\n",
                    (unsigned long long)[selectedDevice maxBufferLength] / (1024*1024));
            fprintf(stderr, "[Metal]   Recommended max working set: %llu MB\n",
                    (unsigned long long)[selectedDevice recommendedMaxWorkingSetSize] / (1024*1024));
            fprintf(stderr, "[Metal]   Max threads per threadgroup: %llu\n",
                    (unsigned long long)[selectedDevice maxThreadsPerThreadgroup].width);

            this->deviceIndex = deviceIndex < 0 ? 0 : deviceIndex;
            this->platformIndex = 0; // Metal has no "platform" concept like OpenCL

            // Apple GPUs always have SIMD width 32 and support high occupancy
            simdWidth = 32;
            maxThreadBlockSize = 1024; // Apple Silicon GPUs support up to 1024 threads per threadgroup
            int numThreadBlocksPerComputeUnit = 12;

            // Apple Silicon always supports 64-bit atomics
            supports64BitGlobalAtomics = true;
            // Apple GPU does not support double precision
            supportsDoublePrecision = false;

            if ((useDoublePrecision || useMixedPrecision) && !supportsDoublePrecision)
                throw OpenMMException("This device does not support double precision");

            // Set compilation defines
            compilationDefines["WORK_GROUP_SIZE"] = intToString(ThreadBlockSize);
            // VENDOR_APPLE define for native Metal backend -- we can use Metal SIMD intrinsics
            // directly (simd_sum, ctz, etc.) since we are compiling MSL, not going through cl2Metal.
            compilationDefines["VENDOR_APPLE"] = "";

            if (supports64BitGlobalAtomics)
                compilationDefines["SUPPORTS_64_BIT_ATOMICS"] = "";
            if (supportsDoublePrecision)
                compilationDefines["SUPPORTS_DOUBLE_PRECISION"] = "";

            // For Metal, threadgroup memory fence is threadgroup_barrier(mem_flags::mem_threadgroup)
            // but the kernels may use the define from common.metal; keep the same pattern.
            if (simdWidth >= 32)
                compilationDefines["SYNC_WARPS"] = "threadgroup_barrier(metal::mem_flags::mem_threadgroup)";
            else
                compilationDefines["SYNC_WARPS"] = "threadgroup_barrier(metal::mem_flags::mem_threadgroup)";

            // ---- Create Metal Command Queue ----
            if (originalContext == NULL) {
                // Retain the device and store as void*
                mtlDevice = (__bridge void*)selectedDevice;
                CFRetain(mtlDevice);

                id<MTLCommandQueue> queue = [selectedDevice newCommandQueue];
                if (queue == nil) {
                    throw OpenMMException("Failed to create Metal command queue");
                }
                // newCommandQueue returns a retained object; store it and retain for our void* ownership
                mtlCommandQueue = (__bridge void*)queue;
                CFRetain(mtlCommandQueue);

                if (enableKernelProfiling) {
                    printf("[Metal] Kernel profiling enabled.\n");
                    printf("[Metal] Will log performance data every 500 GPU commands.\n");
                    printf("[Metal] Logging raw profiling data.\n");
                    printf("[ ");
                }
            }
            else {
                // Share the device and queue from the original context
                // Retain both so every context owns its reference and can safely CFRelease in ~MetalContext
                mtlDevice = originalContext->mtlDevice;
                CFRetain(mtlDevice);
                mtlCommandQueue = originalContext->mtlCommandQueue;
                CFRetain(mtlCommandQueue);
            }

            if (reduceEnergyThreadgroups > 1) {
                compilationDefines["REDUCE_ENERGY_MULTIPLE_THREADGROUPS"] = "1";
            } else {
                compilationDefines["REDUCE_ENERGY_MULTIPLE_THREADGROUPS"] = "0";
            }

            numAtoms = system.getNumParticles();
            paddedNumAtoms = TileSize*((numAtoms+TileSize-1)/TileSize);
            numAtomBlocks = (paddedNumAtoms+(TileSize-1))/TileSize;

            // Get the number of GPU cores for thread block calculation.
            // On Apple Silicon, we can query the max threadgroup memory size
            // and compute units are not directly queryable. Use a reasonable estimate.
            // Apple M-series GPUs report max threadgroups in flight through the pipeline.
            // We use a heuristic: assume 10 compute units for base M-series, scale up.
            // A more accurate approach is to use IOKit to query GPU core count,
            // but for simplicity we use maxThreadgroupMemoryLength as a proxy.
            int estimatedComputeUnits = 10; // Conservative default for Apple Silicon
            // For M1: 8 cores, M1 Pro: 14-16, M1 Max: 24-32, M2: 8-10, M3: 10, M4: 10
            // The recommendation buffer count of 120 threadgroups (12 * 10) is a good default.
            numThreadBlocks = numThreadBlocksPerComputeUnit * estimatedComputeUnits;

            if (useDoublePrecision) {
                posq.initialize<mm_double4>(*this, paddedNumAtoms, "posq");
                velm.initialize<mm_double4>(*this, paddedNumAtoms, "velm");
                compilationDefines["USE_DOUBLE_PRECISION"] = "1";
                compilationDefines["convert_real4"] = "convert_double4";
                compilationDefines["make_real2"] = "make_double2";
                compilationDefines["make_real3"] = "make_double3";
                compilationDefines["make_real4"] = "make_double4";
                compilationDefines["convert_mixed4"] = "convert_double4";
                compilationDefines["make_mixed2"] = "make_double2";
                compilationDefines["make_mixed3"] = "make_double3";
                compilationDefines["make_mixed4"] = "make_double4";
            }
            else if (useMixedPrecision) {
                posq.initialize<mm_float4>(*this, paddedNumAtoms, "posq");
                posqCorrection.initialize<mm_float4>(*this, paddedNumAtoms, "posq");
                velm.initialize<mm_double4>(*this, paddedNumAtoms, "velm");
                compilationDefines["USE_MIXED_PRECISION"] = "1";
                compilationDefines["convert_real4"] = "convert_float4";
                compilationDefines["make_real2"] = "make_float2";
                compilationDefines["make_real3"] = "make_float3";
                compilationDefines["make_real4"] = "make_float4";
                compilationDefines["convert_mixed4"] = "convert_double4";
                compilationDefines["make_mixed2"] = "make_double2";
                compilationDefines["make_mixed3"] = "make_double3";
                compilationDefines["make_mixed4"] = "make_double4";
            }
            else {
                posq.initialize<mm_float4>(*this, paddedNumAtoms, "posq");
                velm.initialize<mm_float4>(*this, paddedNumAtoms, "velm");
                compilationDefines["convert_real4"] = "convert_float4";
                compilationDefines["make_real2"] = "make_float2";
                compilationDefines["make_real3"] = "make_float3";
                compilationDefines["make_real4"] = "make_float4";
                compilationDefines["convert_mixed4"] = "convert_float4";
                compilationDefines["make_mixed2"] = "make_float2";
                compilationDefines["make_mixed3"] = "make_float3";
                compilationDefines["make_mixed4"] = "make_float4";
            }
            longForceBuffer.initialize<int64_t>(*this, 3*paddedNumAtoms, "longForceBuffer");
            atomicForceBuffer.initialize<float>(*this, 3*paddedNumAtoms, "atomicForceBuffer");
            posCellOffsets.resize(paddedNumAtoms, mm_int4(0, 0, 0, 0));
            atomIndexDevice.initialize<int32_t>(*this, paddedNumAtoms, "atomIndexDevice");
            atomIndex.resize(paddedNumAtoms);
            for (int i = 0; i < paddedNumAtoms; ++i)
                atomIndex[i] = i;
            atomIndexDevice.upload(atomIndex);
        }
        catch (OpenMMException&) {
            throw;
        }
        catch (...) {
            throw OpenMMException("Error initializing Metal context");
        }
    }

    // Create utility kernels that are used in multiple places.

    @autoreleasepool {
        void* utilitiesLibrary = createProgram(MetalKernelSources::utilities);
        createUtilityKernel(utilitiesLibrary, "clearBuffer", clearBufferPipeline);
        createUtilityKernel(utilitiesLibrary, "clearTwoBuffers", clearTwoBuffersPipeline);
        createUtilityKernel(utilitiesLibrary, "clearThreeBuffers", clearThreeBuffersPipeline);
        createUtilityKernel(utilitiesLibrary, "clearFourBuffers", clearFourBuffersPipeline);
        createUtilityKernel(utilitiesLibrary, "clearFiveBuffers", clearFiveBuffersPipeline);
        createUtilityKernel(utilitiesLibrary, "clearSixBuffers", clearSixBuffersPipeline);
        createUtilityKernel(utilitiesLibrary, "reduceReal4Buffer", reduceReal4Pipeline);
        createUtilityKernel(utilitiesLibrary, "reduceForces", reduceForcesPipeline);
        createUtilityKernel(utilitiesLibrary, "reduceEnergy", reduceEnergyPipeline);
        createUtilityKernel(utilitiesLibrary, "setCharges", setChargesPipeline);

        // Decide whether fast math functions are sufficiently accurate to use.
        // On Metal, fast math functions (fast::sqrt, fast::rsqrt, etc.) are always
        // available and sufficiently accurate on Apple Silicon GPUs.

        if (!useDoublePrecision) {
            // On Apple Silicon Metal, the fast math variants are highly accurate.
            // We can test them similarly to the OpenCL version, but since we know
            // Apple GPU hardware provides excellent accuracy, we default to the
            // fast variants.
            void* determineNativeAccuracyPipeline = NULL;
            createUtilityKernel(utilitiesLibrary, "determineNativeAccuracy", determineNativeAccuracyPipeline);

            MetalArray valuesArray(*this, 20, sizeof(mm_float8), "values");
            vector<mm_float8> values(valuesArray.getSize());
            float nextValue = 1e-4f;
            for (auto& val : values) {
                val.s0 = nextValue;
                nextValue *= (float) M_PI;
            }
            valuesArray.upload(values);

            // Execute the accuracy test kernel
            int valuesSize = (int)values.size();
            executeUtilityKernel(determineNativeAccuracyPipeline, "determineNativeAccuracy",
                                 valuesSize, -1,
                                 {&valuesArray}, {{(const void*)&valuesSize, sizeof(int32_t)}});

            valuesArray.download(values);
            double maxSqrtError = 0.0, maxRsqrtError = 0.0, maxRecipError = 0.0, maxExpError = 0.0, maxLogError = 0.0;
            for (auto& val : values) {
                double v = val.s0;
                double correctSqrt = sqrt(v);
                maxSqrtError = max(maxSqrtError, fabs(correctSqrt-val.s1)/correctSqrt);
                maxRsqrtError = max(maxRsqrtError, fabs(1.0/correctSqrt-val.s2)*correctSqrt);
                maxRecipError = max(maxRecipError, fabs(1.0/v-val.s3)/val.s3);
                maxExpError = max(maxExpError, fabs(exp(v)-val.s4)/val.s4);
                maxLogError = max(maxLogError, fabs(log(v)-val.s5)/val.s5);
            }
            // On Metal, use "fast::" prefix variants if accurate enough, otherwise standard
            compilationDefines["SQRT"] = (maxSqrtError < 1e-6) ? "fast::sqrt" : "sqrt";
            compilationDefines["RSQRT"] = (maxRsqrtError < 1e-6) ? "fast::rsqrt" : "rsqrt";
            // MSL doesn't have fast::recip; on Apple GPUs, 1.0f/x is already fast
            compilationDefines["RECIP"] = "1.0f/";
            compilationDefines["EXP"] = (maxExpError < 1e-6) ? "fast::exp" : "exp";
            compilationDefines["LOG"] = (maxLogError < 1e-6) ? "fast::log" : "log";

            // Release the local pipeline state for the accuracy test kernel
            if (determineNativeAccuracyPipeline != NULL)
                CFRelease(determineNativeAccuracyPipeline);
        }
        else {
            compilationDefines["SQRT"] = "sqrt";
            compilationDefines["RSQRT"] = "rsqrt";
            compilationDefines["RECIP"] = "1.0/";
            compilationDefines["EXP"] = "exp";
            compilationDefines["LOG"] = "log";
        }
        compilationDefines["POW"] = "pow";
        compilationDefines["COS"] = "cos";
        compilationDefines["SIN"] = "sin";
        compilationDefines["TAN"] = "tan";
        compilationDefines["ACOS"] = "acos";
        compilationDefines["ASIN"] = "asin";
        compilationDefines["ATAN"] = "atan";
        compilationDefines["ERF(x)"] = "mm_erf(x)";
        compilationDefines["ERFC(x)"] = "mm_erfc(x)";

        // Release the utilities library (we've already created pipeline states from it)
        CFRelease(utilitiesLibrary);
    }

    // Set defines for applying periodic boundary conditions.

    Vec3 boxVectors[3];
    system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
    boxIsTriclinic = (boxVectors[0][1] != 0.0 || boxVectors[0][2] != 0.0 ||
                      boxVectors[1][0] != 0.0 || boxVectors[1][2] != 0.0 ||
                      boxVectors[2][0] != 0.0 || boxVectors[2][1] != 0.0);
    if (boxIsTriclinic) {
        compilationDefines["APPLY_PERIODIC_TO_DELTA(delta)"] =
            "{"
            "real scale3 = floor(delta.z*invPeriodicBoxSize.z+0.5f); \\\n"
            "delta.xyz -= scale3*periodicBoxVecZ.xyz; \\\n"
            "real scale2 = floor(delta.y*invPeriodicBoxSize.y+0.5f); \\\n"
            "delta.xy -= scale2*periodicBoxVecY.xy; \\\n"
            "real scale1 = floor(delta.x*invPeriodicBoxSize.x+0.5f); \\\n"
            "delta.x -= scale1*periodicBoxVecX.x;}";
        compilationDefines["APPLY_PERIODIC_TO_POS(pos)"] =
            "{"
            "real scale3 = floor(pos.z*invPeriodicBoxSize.z); \\\n"
            "pos.xyz -= scale3*periodicBoxVecZ.xyz; \\\n"
            "real scale2 = floor(pos.y*invPeriodicBoxSize.y); \\\n"
            "pos.xy -= scale2*periodicBoxVecY.xy; \\\n"
            "real scale1 = floor(pos.x*invPeriodicBoxSize.x); \\\n"
            "pos.x -= scale1*periodicBoxVecX.x;}";
        compilationDefines["APPLY_PERIODIC_TO_POS_WITH_CENTER(pos, center)"] =
            "{"
            "real scale3 = floor((pos.z-center.z)*invPeriodicBoxSize.z+0.5f); \\\n"
            "pos.x -= scale3*periodicBoxVecZ.x; \\\n"
            "pos.y -= scale3*periodicBoxVecZ.y; \\\n"
            "pos.z -= scale3*periodicBoxVecZ.z; \\\n"
            "real scale2 = floor((pos.y-center.y)*invPeriodicBoxSize.y+0.5f); \\\n"
            "pos.x -= scale2*periodicBoxVecY.x; \\\n"
            "pos.y -= scale2*periodicBoxVecY.y; \\\n"
            "real scale1 = floor((pos.x-center.x)*invPeriodicBoxSize.x+0.5f); \\\n"
            "pos.x -= scale1*periodicBoxVecX.x;}";
    }
    else {
        compilationDefines["APPLY_PERIODIC_TO_DELTA(delta)"] =
            "delta.xyz -= floor(delta.xyz*invPeriodicBoxSize.xyz+0.5f)*periodicBoxSize.xyz;";
        compilationDefines["APPLY_PERIODIC_TO_POS(pos)"] =
            "pos.xyz -= floor(pos.xyz*invPeriodicBoxSize.xyz)*periodicBoxSize.xyz;";
        compilationDefines["APPLY_PERIODIC_TO_POS_WITH_CENTER(pos, center)"] =
            "{"
            "pos.x -= floor((pos.x-center.x)*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x; \\\n"
            "pos.y -= floor((pos.y-center.y)*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y; \\\n"
            "pos.z -= floor((pos.z-center.z)*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;}";
    }

    // Create utilities objects.

    bonded = new MetalBondedUtilities(*this);
    nonbonded = new MetalNonbondedUtilities(*this);
    integration = new MetalIntegrationUtilities(*this, system);
    expression = new MetalExpressionUtilities(*this);
}

MetalContext::~MetalContext() {
    // Flush any pending batched GPU commands
    try { flushQueue(); } catch (...) {}

    for (auto force : forces)
        delete force;
    for (auto listener : reorderListeners)
        delete listener;
    for (auto computation : preComputations)
        delete computation;
    for (auto computation : postComputations)
        delete computation;
    if (pinnedBuffer != NULL) {
        CFRelease(pinnedBuffer);
        pinnedBuffer = NULL;
    }
    if (integration != NULL)
        delete integration;
    if (expression != NULL)
        delete expression;
    if (bonded != NULL)
        delete bonded;
    if (nonbonded != NULL)
        delete nonbonded;

    // Release utility pipeline states
    {
        auto releasePipeline = [](void*& p) {
            if (p != NULL) {
                CFRelease(p);
                p = NULL;
            }
        };
        releasePipeline(clearBufferPipeline);
        releasePipeline(clearTwoBuffersPipeline);
        releasePipeline(clearThreeBuffersPipeline);
        releasePipeline(clearFourBuffersPipeline);
        releasePipeline(clearFiveBuffersPipeline);
        releasePipeline(clearSixBuffersPipeline);
        releasePipeline(reduceReal4Pipeline);
        releasePipeline(reduceForcesPipeline);
        releasePipeline(reduceEnergyPipeline);
        releasePipeline(setChargesPipeline);
    }

    if (enableKernelProfiling) {
        printProfilingEvents();
        printf(" ]\n");
    }

    // Release device and command queue if we own them (originalContext == NULL case)
    // Note: We track ownership by checking if these were CFRetain'd by us.
    // For simplicity, we always release them here. If shared from originalContext,
    // the original will have its own retained references.
    if (mtlCommandQueue != NULL) {
        CFRelease(mtlCommandQueue);
        mtlCommandQueue = NULL;
    }
    if (mtlDevice != NULL) {
        CFRelease(mtlDevice);
        mtlDevice = NULL;
    }
}

void MetalContext::initialize() {
    bonded->initialize(system);
    numForceBuffers = std::max(numForceBuffers, (int) platformData.contexts.size());
    int energyBufferSize = max(numThreadBlocks*ThreadBlockSize, nonbonded->getNumEnergyBuffers());
    if (useDoublePrecision || useMixedPrecision) {
        std::cout << METAL_LOG_HEADER << "Detected unsupported precision: ";
        if (useDoublePrecision) {
            std::cout << "double";
        } else {
            std::cout << "mixed";
        }
        std::cout << " precision." << std::endl;
        std::cout << METAL_LOG_HEADER << "Quitting now." << std::endl;
        exit(10);
    }

    forceBuffers.initialize<mm_float4>(*this, paddedNumAtoms*numForceBuffers, "forceBuffers");
    force.initialize(*this, forceBuffers.getDeviceBuffer(), paddedNumAtoms, sizeof(mm_float4), "force");
    energyBuffer.initialize<float>(*this, energyBufferSize, "energyBuffer");
    energySum.initialize<float>(*this, reduceEnergyThreadgroups, "energySum");

    addAutoclearBuffer(longForceBuffer);
    addAutoclearBuffer(atomicForceBuffer);
    addAutoclearBuffer(forceBuffers);
    addAutoclearBuffer(energyBuffer);
    int numEnergyParamDerivs = energyParamDerivNames.size();
    if (numEnergyParamDerivs > 0) {
        if (useDoublePrecision || useMixedPrecision)
            energyParamDerivBuffer.initialize<double>(*this, numEnergyParamDerivs*energyBufferSize, "energyParamDerivBuffer");
        else
            energyParamDerivBuffer.initialize<float>(*this, numEnergyParamDerivs*energyBufferSize, "energyParamDerivBuffer");
        addAutoclearBuffer(energyParamDerivBuffer);
    }

    // Create pinned buffer. On Metal with shared storage mode, all buffers are
    // CPU-accessible, so the "pinned" buffer is just a regular MTLBuffer.
    @autoreleasepool {
        int bufferBytes = max(max((int) velm.getSize()*velm.getElementSize(),
                energyBufferSize*energyBuffer.getElementSize()),
                (int) longForceBuffer.getSize()*longForceBuffer.getElementSize());
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;
        id<MTLBuffer> pinned = [device newBufferWithLength:(NSUInteger)bufferBytes
                                                   options:MTLResourceStorageModeShared];
        if (pinned == nil) {
            throw OpenMMException("Failed to allocate pinned buffer");
        }
        pinnedBuffer = (__bridge void*)pinned;
        CFRetain(pinnedBuffer);
        pinnedMemory = [pinned contents];
        if (pinnedMemory == nil) {
            fprintf(stderr, "[Metal] Pinned buffer contents pointer is nil (size %d bytes)\n", bufferBytes);
            throw OpenMMException("Failed to get CPU access to pinned buffer");
        }
    }

    for (int i = 0; i < numAtoms; i++) {
        double mass = system.getParticleMass(i);
        if (useDoublePrecision || useMixedPrecision)
            ((mm_double4*) pinnedMemory)[i] = mm_double4(0.0, 0.0, 0.0, mass == 0.0 ? 0.0 : 1.0/mass);
        else
            ((mm_float4*) pinnedMemory)[i] = mm_float4(0.0f, 0.0f, 0.0f, mass == 0.0 ? 0.0f : (float) (1.0/mass));
    }
    velm.upload(pinnedMemory);
    findMoleculeGroups();
    nonbonded->initialize(system);
}

void MetalContext::initializeContexts() {
    getPlatformData().initializeContexts(system);
}

void MetalContext::addForce(ComputeForceInfo* force) {
    ComputeContext::addForce(force);
    MetalForceInfo* clinfo = dynamic_cast<MetalForceInfo*>(force);
    if (clinfo != NULL)
        requestForceBuffers(clinfo->getRequiredForceBuffers());
}

void MetalContext::requestForceBuffers(int minBuffers) {
    numForceBuffers = std::max(numForceBuffers, minBuffers);
}

void* MetalContext::createProgram(const string source, const char* optimizationFlags) {
    return createProgram(source, map<string, string>(), optimizationFlags);
}

void* MetalContext::createProgram(const string source, const map<string, string>& defines, const char* optimizationFlags) {
    @autoreleasepool {
        stringstream src;

        // Add preprocessor defines
        for (auto& pair : compilationDefines) {
            // Skip defines that are overridden by the caller
            if (defines.find(pair.first) == defines.end()) {
                src << "#define " << pair.first;
                if (!pair.second.empty())
                    src << " " << pair.second;
                src << endl;
            }
        }
        if (!compilationDefines.empty())
            src << endl;

        // Type definitions for MSL
        if (useDoublePrecision) {
            // Metal does not natively support double, but we define the types
            // for compatibility. Actual double-precision support requires
            // software emulation or is not supported.
            src << "typedef double real;\n";
            src << "typedef double2 real2;\n";
            src << "typedef double3 real3;\n";
            src << "typedef double4 real4;\n";
        }
        else {
            src << "typedef float real;\n";
            src << "typedef float2 real2;\n";
            src << "typedef float3 real3;\n";
            src << "typedef float4 real4;\n";
        }
        if (useDoublePrecision || useMixedPrecision) {
            src << "typedef double mixed;\n";
            src << "typedef double2 mixed2;\n";
            src << "typedef double3 mixed3;\n";
            src << "typedef double4 mixed4;\n";
        }
        else {
            src << "typedef float mixed;\n";
            src << "typedef float2 mixed2;\n";
            src << "typedef float3 mixed3;\n";
            src << "typedef float4 mixed4;\n";
        }

        // Portable erfc/erf helpers. Metal MSL has neither erf() nor erfc().
        // Numerical Recipes (Press et al.) eq. 6.2.16 — Chebyshev rational approximation.
        // Max error ~1.2e-7, different bias pattern from A&S 7.1.26 for better PME accumulation.
        // NOTE: Injected before common.metal, so metal:: namespace prefix is required.
        src << "#include <metal_stdlib>\n"
            << "inline float mm_erfc(float x) {\n"
            << "    float a = metal::fabs(x);\n"
            << "    float t = 1.0f / (1.0f + 0.5f * a);\n"
            << "    float tau = t * metal::exp(-a * a - 1.26551223f\n"
            << "        + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f\n"
            << "        + t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f\n"
            << "        + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f)))))))));\n"
            << "    return x >= 0.0f ? tau : 2.0f - tau;\n"
            << "}\n"
            << "inline float mm_erf(float x) { return 1.0f - mm_erfc(x); }\n"
            << "\n";

        // Include the common MSL header
        src << MetalKernelSources::common << endl;

        // Add caller-specific defines
        for (auto& pair : defines) {
            src << "#define " << pair.first;
            if (!pair.second.empty())
                src << " " << pair.second;
            src << endl;
        }
        if (!defines.empty())
            src << endl;

        // Add the source code
        src << source << endl;

        // Compile the Metal source
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;
        options.languageVersion = MTLLanguageVersion3_0;

        // MSL requires kernel function parameters to have address-space qualifiers.
        // Bare scalar types like "int size" are invalid in kernel signatures.
        // Transform them to "constant int& size" which works with setBytes on host.
        // This is a source-to-source fixup that only affects kernel signatures.
        string finalSource = src.str();
        // Debug: dump source to temp file for inspection
        if (finalSource.find("generateRandomNumbers") != string::npos) {
            FILE* dbg = fopen("/tmp/metal_source_pre_transform.metal", "w");
            if (dbg) { fwrite(finalSource.c_str(), 1, finalSource.size(), dbg); fclose(dbg); }
            fprintf(stderr, "[Metal Debug] Wrote pre-transform source to /tmp/metal_source_pre_transform.metal\n");
        }
        {
            // STEP 0: Inject MM_THREAD_ARGS into kernel signatures that don't have it.
            // Common platform kernels (.cc files) don't include MM_THREAD_ARGS because they're
            // shared with CUDA/OpenCL. In MSL, thread position must be a kernel parameter.
            auto findKernelDecl = [&](size_t startPos) -> size_t {
                size_t p1 = finalSource.find("KERNEL void", startPos);
                size_t p2 = finalSource.find("kernel void", startPos);
                size_t p3 = finalSource.find("__kernel void", startPos);
                return min({p1, p2, p3});
            };
            size_t pos = 0;
            while ((pos = findKernelDecl(pos)) != string::npos) {
                size_t sigStart = finalSource.find('(', pos);
                size_t sigEnd = finalSource.find(')', sigStart);
                if (sigStart == string::npos || sigEnd == string::npos) break;
                string sig = finalSource.substr(sigStart, sigEnd - sigStart + 1);
                if (sig.find("MM_THREAD_ARGS") == string::npos) {
                    // Inject MM_THREAD_ARGS before the closing )
                    // Check if the last non-whitespace before ) is a comma or (
                    size_t insertPos = sigEnd;
                    string injection;
                    // Check if params exist (not empty parens)
                    string trimmed = sig.substr(1, sig.size()-2);
                    bool hasParams = false;
                    for (char c : trimmed) { if (!isspace(c)) { hasParams = true; break; } }
                    if (hasParams) {
                        injection = ",\n        MM_THREAD_ARGS";
                    } else {
                        injection = "MM_THREAD_ARGS";
                    }
                    finalSource.insert(insertPos, injection);
                    // Also inject MM_INIT_THREAD_STATE after the opening {
                    size_t bracePos = finalSource.find('{', insertPos + injection.size());
                    if (bracePos != string::npos) {
                        string initCode = "\n    MM_INIT_THREAD_STATE\n";
                        finalSource.insert(bracePos + 1, initCode);
                        pos = bracePos + 1 + initCode.size();
                    } else {
                        pos = insertPos + injection.size() + 1;
                    }
                } else {
                    // Kernel already has MM_THREAD_ARGS — still need MM_INIT_THREAD_STATE
                    size_t bracePos = finalSource.find('{', sigEnd);
                    if (bracePos != string::npos) {
                        // Check if MM_INIT_THREAD_STATE already present
                        string after = finalSource.substr(bracePos + 1, 100);
                        if (after.find("MM_INIT_THREAD_STATE") == string::npos) {
                            string initCode = "\n    MM_INIT_THREAD_STATE\n";
                            finalSource.insert(bracePos + 1, initCode);
                            pos = bracePos + 1 + initCode.size();
                        } else {
                            pos = sigEnd + 1;
                        }
                    } else {
                        pos = sigEnd + 1;
                    }
                }
            }

            // STEP 1: Find kernel function signatures and wrap bare scalar params
            // Match: (kernel void FUNCNAME(..., TYPE PARAM, ...) where TYPE is a bare scalar
            // We wrap: int, uint, unsigned int, float, real, mixed, long, unsigned long, mm_long, mm_ulong
            // with "constant TYPE& " prefix
            // Approach: find "kernel void" or "KERNEL void", then scan the param list
            // For simplicity, we use a regex-free approach: replace common patterns

            // Helper: collapse whitespace runs to a single space, but preserve
            // preprocessor directives (#ifdef, #endif, #else, #ifndef) on their own lines.
            auto collapseWhitespace = [](const string& s) -> string {
                string result;
                result.reserve(s.size());
                bool lastWasSpace = false;
                for (size_t i = 0; i < s.size(); i++) {
                    char c = s[i];
                    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                        // Check if the next non-space char is '#' (preprocessor directive)
                        size_t next = i + 1;
                        while (next < s.size() && (s[next] == ' ' || s[next] == '\t')) next++;
                        if (next < s.size() && s[next] == '#') {
                            // Preserve newline before preprocessor directives
                            if (!result.empty() && result.back() != '\n')
                                result += '\n';
                            lastWasSpace = false;
                            continue;
                        }
                        // Also check if current position is right after #endif or #else
                        // (preserve newline after preprocessor directives)
                        if (c == '\n') {
                            // Check if the line we just finished is a preprocessor directive
                            size_t lineStart = result.rfind('\n');
                            if (lineStart == string::npos) lineStart = 0; else lineStart++;
                            string line = result.substr(lineStart);
                            // Trim leading spaces
                            size_t firstNonSpace = line.find_first_not_of(" \t");
                            if (firstNonSpace != string::npos && line[firstNonSpace] == '#') {
                                result += '\n';
                                lastWasSpace = false;
                                continue;
                            }
                        }
                        if (!lastWasSpace) { result += ' '; lastWasSpace = true; }
                    } else {
                        result += c;
                        lastWasSpace = false;
                    }
                }
                return result;
            };

            auto replaceInKernelSigs = [&](const string& type) {
                // Match ", TYPE name" and "( TYPE name" in kernel signatures.
                // Also match "\n TYPE name" for params after #else/#endif lines.
                // Signatures may span multiple lines, so we normalize whitespace
                // before matching and replace with the normalized form.
                // For "const TYPE" inputs, strip the const since "constant" already implies it.
                string baseType = type;
                if (baseType.substr(0, 6) == "const ") baseType = baseType.substr(6);
                string pattern1 = ", " + type + " ";
                string replace1 = ", constant " + baseType + "& ";
                string pattern2 = "(" + type + " ";
                string replace2 = "(constant " + baseType + "& ";
                // Pattern for params after preprocessor directives: "\n TYPE name"
                string pattern3 = "\n " + type + " ";
                string replace3 = "\n constant " + baseType + "& ";

                auto findKernel = [&](size_t startPos) -> size_t {
                    size_t p1 = finalSource.find("kernel void", startPos);
                    size_t p2 = finalSource.find("KERNEL void", startPos);
                    size_t p3 = finalSource.find("__kernel void", startPos);
                    return min({p1, p2, p3});
                };
                size_t pos = 0;
                while ((pos = findKernel(pos)) != string::npos) {
                    // Extract kernel name for logging
                    size_t nameStart = pos;
                    // Skip past "kernel void" / "KERNEL void" / "__kernel void"
                    if (finalSource.compare(pos, 13, "__kernel void") == 0) nameStart = pos + 13;
                    else if (finalSource.compare(pos, 11, "kernel void") == 0) nameStart = pos + 11;
                    else if (finalSource.compare(pos, 11, "KERNEL void") == 0) nameStart = pos + 11;
                    while (nameStart < finalSource.size() && isspace(finalSource[nameStart])) nameStart++;
                    size_t nameEnd = nameStart;
                    while (nameEnd < finalSource.size() && (isalnum(finalSource[nameEnd]) || finalSource[nameEnd] == '_')) nameEnd++;
                    string kernelName = finalSource.substr(nameStart, nameEnd - nameStart);

                    size_t sigStart = finalSource.find('(', pos);
                    size_t sigEnd = finalSource.find(')', sigStart);
                    if (sigStart == string::npos || sigEnd == string::npos) break;

                    // Extract and transform the signature, but STOP before MM_THREAD_ARGS
                    string sig = finalSource.substr(sigStart, sigEnd - sigStart + 1);
                    size_t threadArgsPos = sig.find("MM_THREAD_ARGS");
                    size_t transformEnd = (threadArgsPos != string::npos) ? threadArgsPos : sig.size();
                    string sigBefore = sig.substr(0, transformEnd);
                    string sigAfter = sig.substr(transformEnd);

                    // Normalize whitespace for reliable matching
                    string origNormalized = collapseWhitespace(sigBefore);
                    string normalized = origNormalized;

                    // Track which params get wrapped
                    vector<string> wrappedParams;
                    size_t spos = 0;
                    while ((spos = normalized.find(pattern1, spos)) != string::npos) {
                        // Extract param name that follows the type
                        size_t pnStart = spos + pattern1.size();
                        size_t pnEnd = pnStart;
                        while (pnEnd < normalized.size() && (isalnum(normalized[pnEnd]) || normalized[pnEnd] == '_')) pnEnd++;
                        wrappedParams.push_back(normalized.substr(pnStart, pnEnd - pnStart));
                        normalized.replace(spos, pattern1.length(), replace1);
                        spos += replace1.length();
                    }
                    spos = 0;
                    while ((spos = normalized.find(pattern2, spos)) != string::npos) {
                        size_t pnStart = spos + pattern2.size();
                        size_t pnEnd = pnStart;
                        while (pnEnd < normalized.size() && (isalnum(normalized[pnEnd]) || normalized[pnEnd] == '_')) pnEnd++;
                        wrappedParams.push_back(normalized.substr(pnStart, pnEnd - pnStart));
                        normalized.replace(spos, pattern2.length(), replace2);
                        spos += replace2.length();
                    }
                    // Pattern 3: params after #else/#endif lines
                    spos = 0;
                    while ((spos = normalized.find(pattern3, spos)) != string::npos) {
                        size_t pnStart = spos + pattern3.size();
                        size_t pnEnd = pnStart;
                        while (pnEnd < normalized.size() && (isalnum(normalized[pnEnd]) || normalized[pnEnd] == '_')) pnEnd++;
                        wrappedParams.push_back(normalized.substr(pnStart, pnEnd - pnStart));
                        normalized.replace(spos, pattern3.length(), replace3);
                        spos += replace3.length();
                    }

                    string newSig = normalized + sigAfter;
                    if (normalized != origNormalized) {
                        for (const auto& pn : wrappedParams) {
                            fprintf(stderr, "[Metal Transform STEP 1] kernel '%s': wrapped '%s %s' → 'constant %s& %s'\n",
                                    kernelName.c_str(), type.c_str(), pn.c_str(), type.c_str(), pn.c_str());
                        }
                    }
                    finalSource.replace(sigStart, sigEnd - sigStart + 1, newSig);
                    pos = sigStart + newSig.length();
                }
            };

            // Transform common scalar AND vector types used in OpenMM kernel signatures
            // Order matters! Longer types first to avoid partial matches.
            // Vector types (real4, float4, etc.) before their scalar base types.
            // "unsigned int" before "int", "unsigned long" before "long"
            // "const TYPE" before plain "TYPE" to catch qualified params (e.g. "const mixed energy")
            replaceInKernelSigs("const unsigned long");
            replaceInKernelSigs("const unsigned int");
            replaceInKernelSigs("const mm_ulong");
            replaceInKernelSigs("const mm_long");
            replaceInKernelSigs("const mixed4");
            replaceInKernelSigs("const mixed3");
            replaceInKernelSigs("const mixed2");
            replaceInKernelSigs("const mixed");
            replaceInKernelSigs("const float4");
            replaceInKernelSigs("const float3");
            replaceInKernelSigs("const float2");
            replaceInKernelSigs("const float");
            replaceInKernelSigs("const real4");
            replaceInKernelSigs("const real3");
            replaceInKernelSigs("const real2");
            replaceInKernelSigs("const real");
            replaceInKernelSigs("const int4");
            replaceInKernelSigs("const int2");
            replaceInKernelSigs("const long");
            replaceInKernelSigs("const uint");
            replaceInKernelSigs("const int");
            replaceInKernelSigs("unsigned long");
            replaceInKernelSigs("unsigned int");
            replaceInKernelSigs("mm_ulong");
            replaceInKernelSigs("mm_long");
            replaceInKernelSigs("mixed4");
            replaceInKernelSigs("mixed3");
            replaceInKernelSigs("mixed2");
            replaceInKernelSigs("mixed");
            replaceInKernelSigs("float4");
            replaceInKernelSigs("float3");
            replaceInKernelSigs("float2");
            replaceInKernelSigs("float");
            replaceInKernelSigs("real4");
            replaceInKernelSigs("real3");
            replaceInKernelSigs("real2");
            replaceInKernelSigs("real");
            replaceInKernelSigs("int4");
            replaceInKernelSigs("int2");
            replaceInKernelSigs("long");
            replaceInKernelSigs("uint");
            replaceInKernelSigs("int");

            // STEP 1b: Fix mutable scalar params.
            // Some kernel parameters (e.g., randomIndex) are modified inside the body.
            // After STEP 1, they became "constant TYPE& name" which is read-only.
            // Fix: rename param to _carg_name and add "TYPE name = _carg_name;" after MM_INIT_THREAD_STATE.
            {
                auto findKernelStep1b = [&](size_t startPos) -> size_t {
                    size_t p1 = finalSource.find("kernel void", startPos);
                    size_t p2 = finalSource.find("KERNEL void", startPos);
                    size_t p3 = finalSource.find("__kernel void", startPos);
                    return min({p1, p2, p3});
                };
                // Helper: find matching brace
                auto findBrace = [&](size_t openPos) -> size_t {
                    int depth = 1;
                    for (size_t i = openPos + 1; i < finalSource.size(); i++) {
                        if (finalSource[i] == '{') depth++;
                        else if (finalSource[i] == '}') { depth--; if (depth == 0) return i; }
                    }
                    return string::npos;
                };
                size_t kpos = 0;
                while ((kpos = findKernelStep1b(kpos)) != string::npos) {
                    // Extract kernel name for logging
                    size_t knStart = kpos;
                    if (finalSource.compare(kpos, 13, "__kernel void") == 0) knStart = kpos + 13;
                    else if (finalSource.compare(kpos, 11, "kernel void") == 0) knStart = kpos + 11;
                    else if (finalSource.compare(kpos, 11, "KERNEL void") == 0) knStart = kpos + 11;
                    while (knStart < finalSource.size() && isspace(finalSource[knStart])) knStart++;
                    size_t knEnd = knStart;
                    while (knEnd < finalSource.size() && (isalnum(finalSource[knEnd]) || finalSource[knEnd] == '_')) knEnd++;
                    string kernelName1b = finalSource.substr(knStart, knEnd - knStart);

                    size_t sigStart = finalSource.find('(', kpos);
                    size_t sigEnd = finalSource.find(')', sigStart);
                    if (sigStart == string::npos || sigEnd == string::npos) break;

                    // Find body
                    size_t bodyOpen = finalSource.find('{', sigEnd);
                    if (bodyOpen == string::npos) break;
                    size_t bodyClose = findBrace(bodyOpen);
                    if (bodyClose == string::npos) break;

                    string body = finalSource.substr(bodyOpen, bodyClose - bodyOpen + 1);

                    // Find all "constant TYPE& name" params in the signature
                    string sig = finalSource.substr(sigStart, sigEnd - sigStart + 1);
                    // Look for pattern: "constant TYPE& name" where name is a C identifier
                    vector<pair<string, string>> mutableParams; // (type, name) pairs to fix
                    size_t sp = 0;
                    string constPrefix = "constant ";
                    while ((sp = sig.find(constPrefix, sp)) != string::npos) {
                        size_t typeStart = sp + constPrefix.size();
                        // Find the '&'
                        size_t ampPos = sig.find('&', typeStart);
                        if (ampPos == string::npos) { sp = typeStart; continue; }
                        string paramType = sig.substr(typeStart, ampPos - typeStart);
                        // Trim whitespace from type
                        while (!paramType.empty() && paramType.back() == ' ') paramType.pop_back();
                        // Get param name (after & and space)
                        size_t nameStart = ampPos + 1;
                        while (nameStart < sig.size() && sig[nameStart] == ' ') nameStart++;
                        size_t nameEnd = nameStart;
                        while (nameEnd < sig.size() && (isalnum(sig[nameEnd]) || sig[nameEnd] == '_')) nameEnd++;
                        string paramName = sig.substr(nameStart, nameEnd - nameStart);

                        if (paramName.empty() || paramName.substr(0, 4) == "_mm_") {
                            sp = nameEnd;
                            continue;
                        }

                        // Check if this param is assigned to in the body (mutation, not redeclaration)
                        // Look for: paramName =, paramName +=, paramName -=, paramName++, paramName--
                        // Skip redeclarations like "TYPE paramName =" which are new variables
                        bool isMutated = false;
                        size_t bp = 0;
                        while ((bp = body.find(paramName, bp)) != string::npos) {
                            // Check it's a whole word
                            bool leftOk = (bp == 0 || (!isalnum(body[bp-1]) && body[bp-1] != '_'));
                            size_t afterName = bp + paramName.size();
                            bool rightOk = (afterName >= body.size() || (!isalnum(body[afterName]) && body[afterName] != '_'));
                            if (leftOk && rightOk) {
                                // Check what follows
                                size_t ch = afterName;
                                while (ch < body.size() && body[ch] == ' ') ch++;
                                if (ch < body.size()) {
                                    bool isAssign = false;
                                    if (body[ch] == '=' && (ch + 1 >= body.size() || body[ch+1] != '='))
                                        isAssign = true;
                                    else if (ch + 1 < body.size() && body[ch] == '+' && body[ch+1] == '=')
                                        isAssign = true;
                                    else if (ch + 1 < body.size() && body[ch] == '-' && body[ch+1] == '=')
                                        isAssign = true;
                                    else if (ch + 1 < body.size() && body[ch] == '+' && body[ch+1] == '+')
                                        isAssign = true;
                                    else if (ch + 1 < body.size() && body[ch] == '-' && body[ch+1] == '-')
                                        isAssign = true;

                                    if (isAssign) {
                                        // Check if this is a redeclaration (preceded by a type)
                                        // Look backwards past whitespace for a type keyword
                                        size_t bk = bp;
                                        while (bk > 0 && body[bk-1] == ' ') bk--;
                                        size_t tokEnd = bk;
                                        while (bk > 0 && (isalnum(body[bk-1]) || body[bk-1] == '_')) bk--;
                                        string prevTok = body.substr(bk, tokEnd - bk);
                                        bool isRedecl = (prevTok == "int" || prevTok == "uint" ||
                                                         prevTok == "float" || prevTok == "real" ||
                                                         prevTok == "mixed" || prevTok == "long" ||
                                                         prevTok == "double" || prevTok == "bool" ||
                                                         prevTok == "short" || prevTok == "char" ||
                                                         prevTok == "unsigned" || prevTok == "signed" ||
                                                         prevTok == "auto" || prevTok == "const");
                                        if (!isRedecl) {
                                            isMutated = true;
                                        }
                                    }
                                }
                            }
                            if (isMutated) break;
                            bp = afterName;
                        }

                        if (isMutated) {
                            mutableParams.push_back({paramType, paramName});
                        }
                        sp = nameEnd;
                    }

                    // Apply fixes for mutable params (work backwards to preserve positions)
                    if (!mutableParams.empty()) {
                        // Build the local copy declarations to insert after MM_INIT_THREAD_STATE
                        string localCopies;
                        for (const auto& [type, name] : mutableParams) {
                            localCopies += "    " + type + " " + name + " = _carg_" + name + ";\n";
                            fprintf(stderr, "[Metal Transform STEP 1b] kernel '%s': mutable param '%s %s' → local copy '_carg_%s'\n",
                                    kernelName1b.c_str(), type.c_str(), name.c_str(), name.c_str());
                        }

                        // Rename params in signature: "& name" → "& _carg_name"
                        string newSig = sig;
                        for (const auto& [type, name] : mutableParams) {
                            string oldPattern = "& " + name;
                            string newPattern = "& _carg_" + name;
                            size_t rpos = newSig.find(oldPattern);
                            if (rpos != string::npos) {
                                // Verify next char is not alnum (whole word)
                                size_t afterOld = rpos + oldPattern.size();
                                if (afterOld >= newSig.size() || (!isalnum(newSig[afterOld]) && newSig[afterOld] != '_')) {
                                    newSig.replace(rpos, oldPattern.size(), newPattern);
                                }
                            }
                        }

                        // Replace signature
                        finalSource.replace(sigStart, sigEnd - sigStart + 1, newSig);

                        // Find where to insert local copies — after MM_INIT_THREAD_STATE
                        size_t newBodyOpen = finalSource.find('{', sigStart + newSig.size());
                        if (newBodyOpen != string::npos) {
                            // Look for MM_INIT_THREAD_STATE
                            size_t initPos = finalSource.find("MM_INIT_THREAD_STATE", newBodyOpen);
                            if (initPos != string::npos && initPos < newBodyOpen + 200) {
                                // Insert after the line containing MM_INIT_THREAD_STATE
                                size_t lineEnd = finalSource.find('\n', initPos);
                                if (lineEnd != string::npos) {
                                    finalSource.insert(lineEnd + 1, localCopies);
                                }
                            } else {
                                // No MM_INIT_THREAD_STATE — insert right after '{'
                                finalSource.insert(newBodyOpen + 1, "\n" + localCopies);
                            }
                        }

                        kpos = sigStart + newSig.size() + localCopies.size() + 100;
                    } else {
                        kpos = bodyClose + 1;
                    }
                }
            }
        }

        {
            // STEP 2: Inject _mm_thread_state into DEVICE helper functions that use thread IDs.
            //
            // Common platform .cc kernels have DEVICE helper functions that reference
            // GLOBAL_ID, LOCAL_ID, etc. via the _mm_ts variable. But _mm_ts is only
            // created inside KERNEL bodies (by MM_INIT_THREAD_STATE). MSL type-checks
            // each function independently, so helpers can't see _mm_ts.
            //
            // Solution: find all DEVICE functions whose bodies use thread-ID macros,
            // add "_mm_thread_state _mm_ts" as a parameter, and pass "_mm_ts" at call sites.

            // Thread-ID macro names that indicate a function needs _mm_ts
            const vector<string> threadMacros = {
                "GLOBAL_ID", "LOCAL_ID", "GLOBAL_SIZE", "LOCAL_SIZE", "GROUP_ID", "NUM_GROUPS"
            };

            // Helper: find the matching closing brace for an opening brace
            auto findMatchingBrace = [&](const string& s, size_t openPos) -> size_t {
                int depth = 1;
                for (size_t i = openPos + 1; i < s.size(); i++) {
                    if (s[i] == '{') depth++;
                    else if (s[i] == '}') { depth--; if (depth == 0) return i; }
                }
                return string::npos;
            };

            // Helper: find matching closing paren for an opening paren
            auto findMatchingParen = [&](const string& s, size_t openPos) -> size_t {
                int depth = 1;
                for (size_t i = openPos + 1; i < s.size(); i++) {
                    if (s[i] == '(') depth++;
                    else if (s[i] == ')') { depth--; if (depth == 0) return i; }
                }
                return string::npos;
            };

            // Helper: check if a position is inside a comment or string literal
            // (simple heuristic — checks for // line comments and /* block comments)
            auto isInComment = [&](const string& s, size_t pos) -> bool {
                // Check for // line comment: scan backward to start of line
                size_t lineStart = s.rfind('\n', pos);
                if (lineStart == string::npos) lineStart = 0;
                size_t dslash = s.find("//", lineStart);
                if (dslash != string::npos && dslash < pos) return true;
                return false;
            };

            // Pass 1: Identify DEVICE functions whose bodies use thread-ID macros.
            // We look for patterns like:
            //   DEVICE void funcName(...)  { ... GLOBAL_ID ... }
            //   DEVICE int funcName(...)   { ... LOCAL_ID ... }
            //   inline DEVICE real funcName(...) { ... }
            //
            // We collect function names into a set.

            set<string> needsThreadState;

            // Regex-free scan: find "DEVICE" tokens not preceded by "KERNEL"
            // then extract the function name and check the body.
            struct DeviceFuncInfo {
                string name;
                size_t sigOpenParen;  // position of '(' in signature
                size_t bodyOpen;      // position of '{' for body
                size_t bodyClose;     // position of matching '}'
            };
            vector<DeviceFuncInfo> deviceFunctions;

            {
                size_t searchPos = 0;
                while (searchPos < finalSource.size()) {
                    // Find next "DEVICE " token (the macro expands to empty or inline)
                    // We need to find function definitions that have DEVICE in them
                    // Pattern: optional "inline" then "DEVICE" then return-type then name(
                    // But since DEVICE expands to empty string in our common.metal,
                    // the actual source after preprocessing won't have "DEVICE" anymore.
                    // Wait — DEVICE is #defined to empty, BUT the .cc source is included
                    // as raw text before preprocessing. The #define is in common.metal
                    // which is prepended. So at this point in createProgram(), the source
                    // string still has literal "DEVICE" tokens from the .cc files.

                    size_t devPos = finalSource.find("DEVICE ", searchPos);
                    if (devPos == string::npos) break;

                    // Skip if this is inside a comment
                    if (isInComment(finalSource, devPos)) {
                        searchPos = devPos + 7;
                        continue;
                    }

                    // Skip if preceded by "KERNEL" (i.e., this is a kernel, not a helper)
                    // Check the characters before DEVICE
                    bool isKernel = false;
                    if (devPos >= 7) {
                        string before = finalSource.substr(devPos - 7, 7);
                        if (before.find("KERNEL") != string::npos || before.find("kernel") != string::npos)
                            isKernel = true;
                    }
                    if (isKernel) {
                        searchPos = devPos + 7;
                        continue;
                    }

                    // Find the opening paren of the function signature
                    size_t parenOpen = finalSource.find('(', devPos);
                    if (parenOpen == string::npos) break;

                    // Make sure this paren is on the same or next line (not too far)
                    size_t nlCount = 0;
                    for (size_t i = devPos; i < parenOpen; i++)
                        if (finalSource[i] == '\n') nlCount++;
                    if (nlCount > 2) { searchPos = devPos + 7; continue; }

                    // Extract function name: word immediately before '('
                    size_t nameEnd = parenOpen;
                    while (nameEnd > devPos && finalSource[nameEnd-1] == ' ') nameEnd--;
                    size_t nameStart = nameEnd;
                    while (nameStart > devPos && (isalnum(finalSource[nameStart-1]) || finalSource[nameStart-1] == '_'))
                        nameStart--;
                    string funcName = finalSource.substr(nameStart, nameEnd - nameStart);

                    if (funcName.empty()) { searchPos = devPos + 7; continue; }

                    // Find matching close paren
                    size_t parenClose = findMatchingParen(finalSource, parenOpen);
                    if (parenClose == string::npos) { searchPos = devPos + 7; continue; }

                    // Find the opening brace of the function body
                    size_t bodyOpen = finalSource.find('{', parenClose);
                    if (bodyOpen == string::npos) break;

                    // Make sure there's no semicolon between ) and { (that would be a declaration, not definition)
                    bool isDecl = false;
                    for (size_t i = parenClose + 1; i < bodyOpen; i++) {
                        if (finalSource[i] == ';') { isDecl = true; break; }
                    }
                    if (isDecl) { searchPos = parenClose + 1; continue; }

                    size_t bodyClose = findMatchingBrace(finalSource, bodyOpen);
                    if (bodyClose == string::npos) break;

                    // Check if body contains any thread-ID macros
                    string body = finalSource.substr(bodyOpen, bodyClose - bodyOpen + 1);
                    bool usesThreadIDs = false;
                    for (const auto& macro : threadMacros) {
                        size_t mpos = 0;
                        while ((mpos = body.find(macro, mpos)) != string::npos) {
                            // Ensure it's a whole word (not part of a longer identifier)
                            bool leftOk = (mpos == 0 || !isalnum(body[mpos-1]) && body[mpos-1] != '_');
                            size_t endPos = mpos + macro.size();
                            bool rightOk = (endPos >= body.size() || !isalnum(body[endPos]) && body[endPos] != '_');
                            if (leftOk && rightOk) {
                                usesThreadIDs = true;
                                break;
                            }
                            mpos = endPos;
                        }
                        if (usesThreadIDs) break;
                    }

                    if (usesThreadIDs) {
                        needsThreadState.insert(funcName);
                    }

                    DeviceFuncInfo info;
                    info.name = funcName;
                    info.sigOpenParen = parenOpen;
                    info.bodyOpen = bodyOpen;
                    info.bodyClose = bodyClose;
                    deviceFunctions.push_back(info);

                    searchPos = bodyClose + 1;
                }
            }

            // Transitive propagation: if function A calls function B that needs thread state,
            // then A also needs thread state (so it can pass _mm_ts).
            {
                bool changed = true;
                while (changed) {
                    changed = false;
                    for (const auto& func : deviceFunctions) {
                        if (needsThreadState.count(func.name)) continue;
                        string body = finalSource.substr(func.bodyOpen, func.bodyClose - func.bodyOpen + 1);
                        for (const auto& needed : needsThreadState) {
                            // Check if this function calls any function that needs thread state
                            size_t callPos = body.find(needed);
                            if (callPos != string::npos) {
                                // Verify it's a function call (followed by '(')
                                size_t afterName = callPos + needed.size();
                                while (afterName < body.size() && body[afterName] == ' ') afterName++;
                                if (afterName < body.size() && body[afterName] == '(') {
                                    needsThreadState.insert(func.name);
                                    changed = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            if (!needsThreadState.empty()) {
                fprintf(stderr, "[Metal Transform STEP 2] Functions needing _mm_thread_state: ");
                for (const auto& name : needsThreadState) fprintf(stderr, "%s ", name.c_str());
                fprintf(stderr, "\n");

                // Pass 2: Rewrite signatures and call sites.
                // We must process from END to START of the source to preserve positions.
                // Collect all edit operations, sort by position descending, then apply.

                struct Edit {
                    size_t pos;
                    size_t len;    // chars to remove (0 for pure insert)
                    string replacement;
                    bool operator>(const Edit& o) const { return pos > o.pos; }
                };
                vector<Edit> edits;

                // 2a: Add "_mm_thread_state _mm_ts" as last parameter to each function that needs it.
                // Re-scan because positions may have shifted from STEP 0/1. We re-find each function.
                {
                    size_t searchPos = 0;
                    while (searchPos < finalSource.size()) {
                        size_t devPos = finalSource.find("DEVICE ", searchPos);
                        if (devPos == string::npos) break;
                        if (isInComment(finalSource, devPos)) { searchPos = devPos + 7; continue; }

                        // Skip KERNEL DEVICE
                        bool isKernel = false;
                        if (devPos >= 7) {
                            string before = finalSource.substr(devPos - 7, 7);
                            if (before.find("KERNEL") != string::npos || before.find("kernel") != string::npos)
                                isKernel = true;
                        }
                        if (isKernel) { searchPos = devPos + 7; continue; }

                        size_t parenOpen = finalSource.find('(', devPos);
                        if (parenOpen == string::npos) break;

                        // Extract function name
                        size_t nameEnd = parenOpen;
                        while (nameEnd > devPos && finalSource[nameEnd-1] == ' ') nameEnd--;
                        size_t nameStart = nameEnd;
                        while (nameStart > devPos && (isalnum(finalSource[nameStart-1]) || finalSource[nameStart-1] == '_'))
                            nameStart--;
                        string funcName = finalSource.substr(nameStart, nameEnd - nameStart);

                        if (!needsThreadState.count(funcName)) {
                            searchPos = devPos + 7;
                            continue;
                        }

                        // Find matching close paren — must handle #ifdef in param lists
                        size_t parenClose = findMatchingParen(finalSource, parenOpen);
                        if (parenClose == string::npos) { searchPos = devPos + 7; continue; }

                        // Check if already has _mm_thread_state
                        string params = finalSource.substr(parenOpen, parenClose - parenOpen + 1);
                        if (params.find("_mm_thread_state") != string::npos) {
                            searchPos = parenClose + 1;
                            continue;
                        }

                        // Find where to insert the parameter.
                        // If there's an #ifdef block before ')', insert before #ifdef.
                        // Otherwise insert before ')'.
                        size_t insertPos = parenClose;

                        // Check for #ifdef / #endif pattern near closing paren
                        // Look backward from ')' for #endif
                        size_t searchBack = parenClose;
                        while (searchBack > parenOpen && finalSource[searchBack] != '#') searchBack--;
                        if (searchBack > parenOpen) {
                            // Check if this is an #endif
                            string directive = finalSource.substr(searchBack, min((size_t)6, finalSource.size() - searchBack));
                            if (directive.find("#endif") == 0 || directive.find("#else") == 0 || directive.find("#ifdef") == 0) {
                                // Find the matching #ifdef above
                                size_t ifdefPos = finalSource.rfind("#ifdef", searchBack - 1);
                                if (ifdefPos != string::npos && ifdefPos > parenOpen) {
                                    // Insert before the #ifdef line
                                    size_t lineStart = finalSource.rfind('\n', ifdefPos);
                                    if (lineStart != string::npos && lineStart > parenOpen) {
                                        insertPos = lineStart; // insert before the newline preceding #ifdef
                                        Edit e;
                                        e.pos = insertPos;
                                        e.len = 0;
                                        e.replacement = ",\n    _mm_thread_state _mm_ts";
                                        edits.push_back(e);
                                        fprintf(stderr, "[Metal Transform STEP 2] Added _mm_ts param to %s (before #ifdef)\n", funcName.c_str());
                                        searchPos = parenClose + 1;
                                        continue;
                                    }
                                }
                            }
                        }

                        // Normal case: insert before ')'
                        // Check if there are existing params
                        string trimmedParams = params.substr(1, params.size() - 2);
                        bool hasParams = false;
                        for (char c : trimmedParams) { if (!isspace(c)) { hasParams = true; break; } }

                        Edit e;
                        e.pos = insertPos;
                        e.len = 0;
                        if (hasParams) {
                            e.replacement = ", _mm_thread_state _mm_ts";
                        } else {
                            e.replacement = "_mm_thread_state _mm_ts";
                        }
                        edits.push_back(e);
                        fprintf(stderr, "[Metal Transform STEP 2] Added _mm_ts param to %s\n", funcName.c_str());

                        searchPos = parenClose + 1;
                    }
                }

                // 2b: Add "_mm_ts" argument at every call site of functions needing thread state.
                // Scan for funcName( and add _mm_ts before the matching ).
                for (const auto& funcName : needsThreadState) {
                    size_t searchPos = 0;
                    while (searchPos < finalSource.size()) {
                        size_t callPos = finalSource.find(funcName, searchPos);
                        if (callPos == string::npos) break;

                        // Verify: must be followed by '(' (allowing whitespace)
                        size_t afterName = callPos + funcName.size();
                        size_t parenPos = afterName;
                        while (parenPos < finalSource.size() && finalSource[parenPos] == ' ') parenPos++;
                        if (parenPos >= finalSource.size() || finalSource[parenPos] != '(') {
                            searchPos = afterName;
                            continue;
                        }

                        // Verify: must be a whole-word match (not part of a longer identifier)
                        if (callPos > 0 && (isalnum(finalSource[callPos-1]) || finalSource[callPos-1] == '_')) {
                            searchPos = afterName;
                            continue;
                        }

                        // Skip if this is the function DEFINITION (preceded by DEVICE and a return type)
                        // Check: is this inside a DEVICE function signature?
                        // Heuristic: if the preceding non-whitespace token before funcName is a return
                        // type (void, int, real, float, mixed, etc.) AND further back is DEVICE,
                        // then this is the definition — skip it.
                        bool isDefinition = false;
                        {
                            // Look backward past spaces/newlines for keywords
                            size_t bk = callPos;
                            while (bk > 0 && (finalSource[bk-1] == ' ' || finalSource[bk-1] == '\n' || finalSource[bk-1] == '\r')) bk--;
                            // Now bk-1 is last char of previous token — extract it
                            size_t tokEnd = bk;
                            while (bk > 0 && (isalnum(finalSource[bk-1]) || finalSource[bk-1] == '_')) bk--;
                            string prevToken = finalSource.substr(bk, tokEnd - bk);
                            if (prevToken == "void" || prevToken == "int" || prevToken == "real" ||
                                prevToken == "float" || prevToken == "mixed" || prevToken == "uint" ||
                                prevToken == "unsigned" || prevToken == "long" || prevToken == "short" ||
                                prevToken == "bool" || prevToken == "double") {
                                // Check if further back we find "DEVICE" or "inline"
                                size_t bk2 = bk;
                                while (bk2 > 0 && (finalSource[bk2-1] == ' ' || finalSource[bk2-1] == '\n')) bk2--;
                                size_t tok2End = bk2;
                                while (bk2 > 0 && (isalnum(finalSource[bk2-1]) || finalSource[bk2-1] == '_')) bk2--;
                                string prevToken2 = finalSource.substr(bk2, tok2End - bk2);
                                if (prevToken2 == "DEVICE" || prevToken2 == "inline") {
                                    isDefinition = true;
                                }
                                // Also check one more level back (for "inline DEVICE void funcName")
                                if (prevToken2 == "DEVICE" || prevToken2 == "inline") {
                                    size_t bk3 = bk2;
                                    while (bk3 > 0 && (finalSource[bk3-1] == ' ' || finalSource[bk3-1] == '\n')) bk3--;
                                    size_t tok3End = bk3;
                                    while (bk3 > 0 && (isalnum(finalSource[bk3-1]) || finalSource[bk3-1] == '_')) bk3--;
                                    string prevToken3 = finalSource.substr(bk3, tok3End - bk3);
                                    if (prevToken3 == "DEVICE" || prevToken3 == "inline")
                                        isDefinition = true;
                                }
                            }
                        }
                        if (isDefinition) {
                            searchPos = afterName;
                            continue;
                        }

                        // This is a call site. Find the matching ')' and insert _mm_ts before it.
                        size_t callParenClose = findMatchingParen(finalSource, parenPos);
                        if (callParenClose == string::npos) { searchPos = afterName; continue; }

                        // Check if _mm_ts is already an argument
                        string args = finalSource.substr(parenPos, callParenClose - parenPos + 1);
                        if (args.find("_mm_ts") != string::npos) {
                            searchPos = callParenClose + 1;
                            continue;
                        }

                        // Check if there are existing arguments
                        string trimmedArgs = args.substr(1, args.size() - 2);
                        bool hasArgs = false;
                        for (char c : trimmedArgs) { if (!isspace(c)) { hasArgs = true; break; } }

                        Edit e;
                        e.pos = callParenClose;
                        e.len = 0;
                        if (hasArgs) {
                            e.replacement = ", _mm_ts";
                        } else {
                            e.replacement = "_mm_ts";
                        }
                        edits.push_back(e);

                        searchPos = callParenClose + 1;
                    }
                }

                // Sort edits by position descending so insertions don't shift later positions
                sort(edits.begin(), edits.end(), [](const Edit& a, const Edit& b) { return a.pos > b.pos; });

                // Apply edits
                for (const auto& e : edits) {
                    finalSource.replace(e.pos, e.len, e.replacement);
                }

                fprintf(stderr, "[Metal Transform STEP 2] Applied %d edits\n", (int)edits.size());
            }
        }

        {
            // STEP 3: Rename MSL reserved words used as variable names.
            // "thread" is an MSL address-space keyword but some .cc kernels use it
            // as a local variable (e.g., rmsd.cc: "const int thread = LOCAL_ID;").
            // Replace whole-word "thread" used as an identifier (not as a keyword).
            // We specifically target: "int thread", "[thread]", "thread+", "thread%",
            // "thread-", "(thread)", "thread)", "thread," patterns that indicate
            // variable usage rather than address-space qualification.
            // STEP 2a: Rewrite OpenCL-style vector constructor casts into MSL constructors.
            // OpenCL commonly emits "(float2) (a, b)". In Metal/C++, that parses as a
            // comma-expression cast, not a 2-component constructor. Rewrite it to "float2(a, b)".
            {
                vector<string> vectorTypes = {
                    "float2", "float3", "float4",
                    "double2", "double3", "double4",
                    "int2", "int3", "int4",
                    "uint2", "uint3", "uint4",
                    "long2", "long3", "long4",
                    "ulong2", "ulong3", "ulong4",
                    "real2", "real3", "real4",
                    "mixed2", "mixed3", "mixed4"
                };
                int vectorCtorFixes = 0;
                for (const auto& type : vectorTypes) {
                    for (const string& pattern : {string("(") + type + ") (", string("(") + type + ")("}) {
                        string replacement = type + "(";
                        size_t pos = 0;
                        while ((pos = finalSource.find(pattern, pos)) != string::npos) {
                            finalSource.replace(pos, pattern.size(), replacement);
                            vectorCtorFixes++;
                            pos += replacement.size();
                        }
                    }
                }
                if (vectorCtorFixes > 0) {
                    fprintf(stderr, "[Metal Transform STEP 2a] Rewrote %d OpenCL vector constructor casts\n", vectorCtorFixes);
                }
            }

            {
                // STEP 2b: Replace OpenCL uint8 vectors with a plain struct that Metal can compile.
                // Apple Metal headers reserve uint8 as an incomplete type, but CMAP kernels use it
                // as a packed group of eight atom indices with .s0 ... .s7 field access.
                auto replaceWholeWord = [&](const string& needle, const string& replacement) {
                    size_t pos = 0;
                    int count = 0;
                    while ((pos = finalSource.find(needle, pos)) != string::npos) {
                        bool leftOk = (pos == 0 || (!isalnum(finalSource[pos-1]) && finalSource[pos-1] != '_'));
                        size_t endPos = pos + needle.size();
                        bool rightOk = (endPos >= finalSource.size() || (!isalnum(finalSource[endPos]) && finalSource[endPos] != '_'));
                        if (leftOk && rightOk) {
                            finalSource.replace(pos, needle.size(), replacement);
                            count++;
                            pos += replacement.size();
                        }
                        else
                            pos = endPos;
                    }
                    return count;
                };
                int uint8Replacements = replaceWholeWord("uint8", "mm_uint8");
                if (uint8Replacements > 0) {
                    finalSource.insert(0, "typedef struct { uint s0, s1, s2, s3, s4, s5, s6, s7; } mm_uint8;\n");
                    fprintf(stderr, "[Metal Transform STEP 2b] Rewrote %d uint8 occurrences\n", uint8Replacements);
                }
            }

            auto replaceReservedVar = [&](const string& reserved, const string& replacement) {
                size_t pos = 0;
                int count = 0;
                while ((pos = finalSource.find(reserved, pos)) != string::npos) {
                    // Check it's a whole-word match
                    bool leftOk = (pos == 0 || (!isalnum(finalSource[pos-1]) && finalSource[pos-1] != '_'));
                    size_t endPos = pos + reserved.size();
                    bool rightOk = (endPos >= finalSource.size() || (!isalnum(finalSource[endPos]) && finalSource[endPos] != '_'));
                    if (!leftOk || !rightOk) { pos = endPos; continue; }

                    // Skip if this is an MSL address-space usage:
                    // "thread " followed by a type is address-space qualifier
                    // Check what follows: if it's a type keyword, skip
                    char after = (endPos < finalSource.size()) ? finalSource[endPos] : 0;
                    size_t afterNonSpacePos = endPos;
                    while (afterNonSpacePos < finalSource.size() && isspace((unsigned char) finalSource[afterNonSpacePos]))
                        afterNonSpacePos++;
                    char afterNonSpace = (afterNonSpacePos < finalSource.size()) ? finalSource[afterNonSpacePos] : 0;

                    // Variable usage patterns: after 'thread' we see [, +, -, %, ), ,, ;, =, <, >
                    // Address-space usage: "thread TYPE" where TYPE is a C type
                    auto isVarOperator = [&](char c) {
                        return (c == '[' || c == '+' || c == '-' || c == '%' ||
                                c == ')' || c == ',' || c == ';' || c == '=' ||
                                c == '<' || c == '>' || c == '|' || c == '&' ||
                                c == ']' || c == '!');
                    };
                    bool isVarUsage = false;
                    if (isVarOperator(after) || (isspace((unsigned char) after) && isVarOperator(afterNonSpace))) {
                        isVarUsage = true;
                    }
                    // "int thread" or "int thread " = declaration
                    if (pos >= 4) {
                        string before4 = finalSource.substr(pos - 4, 4);
                        if (before4 == "int " || before4 == "nt &") {
                            isVarUsage = true;
                        }
                    }
                    // Check for "thread = " (assignment)
                    if (after == ' ' && afterNonSpace == '=') {
                        isVarUsage = true;
                    }

                    if (isVarUsage) {
                        finalSource.replace(pos, reserved.size(), replacement);
                        count++;
                        pos += replacement.size();
                    } else {
                        pos = endPos;
                    }
                }
                if (count > 0) {
                    fprintf(stderr, "[Metal Transform STEP 3] Renamed '%s' → '%s' (%d occurrences)\n",
                            reserved.c_str(), replacement.c_str(), count);
                }
            };
            replaceReservedVar("thread", "_thread_idx");

            // STEP 3a: Add 'thread' to bare pointer-to-array declarations in function bodies.
            // Pattern: "TYPE (*name)[N]" without address space → "thread TYPE (*name)[N]"
            // This handles local variables like "real (*a)[3] = data->a;"
            {
                vector<string> ptrArrayTypes = {"real", "float", "double", "int", "mixed"};
                int count3a = 0;
                for (const auto& type : ptrArrayTypes) {
                    string pattern = type + " (*";
                    size_t pos = 0;
                    while ((pos = finalSource.find(pattern, pos)) != string::npos) {
                        // Check it's not already qualified
                        bool alreadyQualified = false;
                        if (pos >= 7) {
                            string before = finalSource.substr(pos - 7, 7);
                            if (before.find("thread") != string::npos ||
                                before.find("device") != string::npos ||
                                before.find("GLOBAL") != string::npos) {
                                alreadyQualified = true;
                            }
                        }
                        if (!alreadyQualified) {
                            finalSource.insert(pos, "thread ");
                            count3a++;
                            pos += 7 + pattern.size();
                        } else {
                            pos += pattern.size();
                        }
                    }
                }
                if (count3a > 0) {
                    fprintf(stderr, "[Metal Transform STEP 3a] Added 'thread' to %d bare pointer-to-array declarations\n", count3a);
                }
            }

            // STEP 3b: Add 'thread' address-space qualifier to bare pointer params
            // in DEVICE functions. MSL requires all pointers to have explicit address space.
            // DEVICE function params with bare "TYPE*" or "TYPE (*name)[N]" need "thread TYPE*".
            // Skip params that already have GLOBAL, LOCAL, device, threadgroup, constant qualifiers.
            {
                size_t pos = 0;
                int fixCount = 0;
                while (pos < finalSource.size()) {
                    size_t devPos = finalSource.find("DEVICE ", pos);
                    if (devPos == string::npos) break;

                    // Skip KERNEL DEVICE
                    if (devPos >= 7) {
                        string before = finalSource.substr(devPos - 7, 7);
                        if (before.find("KERNEL") != string::npos || before.find("kernel") != string::npos) {
                            pos = devPos + 7; continue;
                        }
                    }

                    // Find the opening paren
                    size_t parenOpen = finalSource.find('(', devPos);
                    if (parenOpen == string::npos) break;
                    size_t nlCount = 0;
                    for (size_t i = devPos; i < parenOpen; i++)
                        if (finalSource[i] == '\n') nlCount++;
                    if (nlCount > 2) { pos = devPos + 7; continue; }

                    // Find matching close paren
                    int depth = 1;
                    size_t parenClose = parenOpen + 1;
                    while (parenClose < finalSource.size() && depth > 0) {
                        if (finalSource[parenClose] == '(') depth++;
                        else if (finalSource[parenClose] == ')') depth--;
                        if (depth > 0) parenClose++;
                    }
                    if (depth != 0) { pos = devPos + 7; continue; }

                    // Scan params for bare pointer types (TYPE* without address-space)
                    // We look for '*' in the param list and check the token before it
                    string sig = finalSource.substr(parenOpen, parenClose - parenOpen + 1);
                    string newSig = sig;
                    bool changed = false;

                    // Find each '*' in the signature (skip those inside "(*" patterns — handled separately)
                    for (size_t sp = 0; sp < newSig.size(); sp++) {
                        if (newSig[sp] != '*') continue;
                        // Skip if this is part of "(*name)" — pointer-to-array syntax
                        if (sp > 0 && newSig[sp-1] == '(') continue;

                        // Check if preceded by address-space qualifier
                        // Look backwards past whitespace for the previous token
                        size_t bk = sp;
                        while (bk > 0 && newSig[bk-1] == ' ') bk--;
                        size_t tokEnd = bk;
                        while (bk > 0 && (isalnum(newSig[bk-1]) || newSig[bk-1] == '_')) bk--;
                        string prevToken = newSig.substr(bk, tokEnd - bk);

                        // Already has address space qualifier — skip
                        if (prevToken == "GLOBAL" || prevToken == "device" ||
                            prevToken == "LOCAL" || prevToken == "LOCAL_ARG" ||
                            prevToken == "threadgroup" || prevToken == "constant" ||
                            prevToken == "RESTRICT" || prevToken == "restrict" ||
                            prevToken == "__global" || prevToken == "__local" ||
                            prevToken == "__constant" || prevToken == "thread") {
                            continue;
                        }

                        // Also check for "const TYPE*" — back up one more token
                        if (prevToken == "const") {
                            size_t bk2 = bk;
                            while (bk2 > 0 && newSig[bk2-1] == ' ') bk2--;
                            size_t tok2End = bk2;
                            while (bk2 > 0 && (isalnum(newSig[bk2-1]) || newSig[bk2-1] == '_')) bk2--;
                            string prevToken2 = newSig.substr(bk2, tok2End - bk2);
                            if (prevToken2 == "GLOBAL" || prevToken2 == "device" ||
                                prevToken2 == "LOCAL" || prevToken2 == "threadgroup" ||
                                prevToken2 == "constant" || prevToken2 == "__global" ||
                                prevToken2 == "thread") {
                                continue;
                            }
                        }

                        // This is a bare pointer — insert "thread " before the type
                        // Find the start of this param (after ',' or '(')
                        size_t paramStart = bk;
                        // Back up past "const " if present
                        if (prevToken == "const") {
                            paramStart = bk;
                            while (paramStart > 0 && newSig[paramStart-1] == ' ') paramStart--;
                            // Back up past the type before const
                            while (paramStart > 0 && (isalnum(newSig[paramStart-1]) || newSig[paramStart-1] == '_')) paramStart--;
                        }
                        // Now back up past whitespace to find ',' or '('
                        size_t insertAt = paramStart;
                        while (insertAt > 0 && newSig[insertAt-1] == ' ') insertAt--;
                        if (insertAt > 0 && (newSig[insertAt-1] == ',' || newSig[insertAt-1] == '(')) {
                            // Insert "thread " right after the comma/paren + space
                            newSig.insert(paramStart, "thread ");
                            sp += 7; // account for inserted text
                            changed = true;
                            fixCount++;
                        }
                    }

                    // Also handle "TYPE (*name)[N]" — pointer-to-array syntax
                    // Pattern: "TYPE (*" where TYPE has no address-space qualifier
                    // Fix: insert "thread " before TYPE → "thread TYPE (*name)[N]"
                    size_t fpPos = 0;
                    while ((fpPos = newSig.find("(*", fpPos)) != string::npos) {
                        // Check if preceded by address-space qualifier
                        size_t bk = fpPos;
                        while (bk > 0 && newSig[bk-1] == ' ') bk--;
                        size_t tokEnd = bk;
                        while (bk > 0 && (isalnum(newSig[bk-1]) || newSig[bk-1] == '_')) bk--;
                        string prevToken = newSig.substr(bk, tokEnd - bk);
                        if (prevToken != "GLOBAL" && prevToken != "device" &&
                            prevToken != "LOCAL" && prevToken != "threadgroup" &&
                            prevToken != "constant" && prevToken != "thread" &&
                            !prevToken.empty()) {
                            // Insert "thread " before the type name (at bk position)
                            newSig.insert(bk, "thread ");
                            fpPos += 7 + 2;
                            changed = true;
                            fixCount++;
                        } else {
                            fpPos += 2;
                        }
                    }

                    if (changed) {
                        finalSource.replace(parenOpen, parenClose - parenOpen + 1, newSig);
                        pos = parenOpen + newSig.size();
                    } else {
                        pos = parenClose + 1;
                    }
                }
                if (fixCount > 0) {
                    fprintf(stderr, "[Metal Transform STEP 3b] Added 'thread' address-space to %d bare pointer params\n", fixCount);
                }
            }
        }

        // Debug: dump post-transform source
        if (finalSource.find("generateRandomNumbers") != string::npos) {
            FILE* dbg = fopen("/tmp/metal_source_post_transform.metal", "w");
            if (dbg) { fwrite(finalSource.c_str(), 1, finalSource.size(), dbg); fclose(dbg); }
            fprintf(stderr, "[Metal Debug] Wrote post-transform source to /tmp/metal_source_post_transform.metal\n");
        }

        NSError* error = nil;
        NSString* sourceStr = @(finalSource.c_str());
        id<MTLLibrary> library = [device newLibraryWithSource:sourceStr
                                                      options:options
                                                        error:&error];
        if (library == nil || error != nil) {
            string errorMsg = "Error compiling Metal kernel";
            if (error != nil) {
                errorMsg += ": " + string([[error localizedDescription] UTF8String]);
                if ([error localizedFailureReason] != nil)
                    errorMsg += "\nReason: " + string([[error localizedFailureReason] UTF8String]);
            }
            // Always log MSL compilation errors to stderr — exceptions may be swallowed
            fprintf(stderr, "[Metal] MSL compilation FAILED:\n%s\n", errorMsg.c_str());

            // Always dump failing source to a temp file — stderr output gets lost in noise
            char failPath[256];
            snprintf(failPath, sizeof(failPath), "/tmp/metal_kernel_FAILED_%d.metal", dumpSourceCounter++);
            std::ofstream failOut(failPath);
            if (failOut.is_open()) {
                failOut << finalSource;
                failOut.close();
                fprintf(stderr, "[Metal] Dumped FAILED source to %s (%zu bytes) — open this file and jump to the error line\n",
                        failPath, finalSource.size());
            }

            // Also dump the first 600 lines to stderr for inline visibility
            string srcStr = finalSource;
            size_t lineCount = 0;
            size_t pos = 0;
            fprintf(stderr, "[Metal] --- Source (first 600 lines) ---\n");
            while (pos < srcStr.size() && lineCount < 600) {
                size_t end = srcStr.find('\n', pos);
                if (end == string::npos) end = srcStr.size();
                fprintf(stderr, "%4d: %s\n", (int)(lineCount+1), srcStr.substr(pos, end-pos).c_str());
                pos = end + 1;
                lineCount++;
            }
            fprintf(stderr, "[Metal] --- End source ---\n");
            throw OpenMMException(errorMsg);
        }

        // Log successful compilation
        NSArray<NSString*>* functionNames = [library functionNames];
        fprintf(stderr, "[Metal] Compiled kernel library (%d functions)\n", (int)[functionNames count]);

        // Dump MSL source to /tmp when OPENMM_METAL_DUMP_SOURCE=1
        if (enableDumpSource) {
            char path[256];
            snprintf(path, sizeof(path), "/tmp/metal_kernel_%d.metal", dumpSourceCounter++);
            std::ofstream out(path);
            if (out.is_open()) {
                out << finalSource;
                out.close();
                fprintf(stderr, "[Metal] Dumped MSL source to %s (%d functions: ",
                        path, (int)[functionNames count]);
                for (NSUInteger i = 0; i < [functionNames count]; i++) {
                    if (i > 0) fprintf(stderr, ", ");
                    fprintf(stderr, "%s", [[functionNames objectAtIndex:i] UTF8String]);
                }
                fprintf(stderr, ")\n");
            }
        }

        // Transfer ownership to void* (retained)
        void* result = (__bridge void*)library;
        CFRetain(result);
        return result;
    }
}

void MetalContext::createUtilityKernel(void* library, const std::string& name, void*& pso) {
    @autoreleasepool {
        id<MTLLibrary> lib = (__bridge id<MTLLibrary>)library;
        id<MTLFunction> function = [lib newFunctionWithName:@(name.c_str())];
        if (function == nil) {
            throw OpenMMException("Utility kernel function not found: " + name);
        }
        NSError* error = nil;
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil || error != nil) {
            string errorMsg = "Error creating pipeline for utility kernel '" + name + "'";
            if (error != nil) {
                errorMsg += ": " + string([[error localizedDescription] UTF8String]);
                if ([error localizedFailureReason] != nil)
                    errorMsg += "\nReason: " + string([[error localizedFailureReason] UTF8String]);
            }
            fprintf(stderr, "[Metal] Pipeline creation FAILED for '%s': %s\n", name.c_str(), errorMsg.c_str());
            throw OpenMMException(errorMsg);
        }
        pso = (__bridge void*)pipeline;
        CFRetain(pso);
    }
}

// getMTLDevice() and getMTLCommandQueue() are defined inline in MetalContext.h

MetalArray* MetalContext::createArray() {
    return new MetalArray();
}

ComputeEvent MetalContext::createEvent() {
    return shared_ptr<ComputeEventImpl>(new MetalEvent(*this));
}

ComputeProgram MetalContext::compileProgram(const std::string source, const std::map<std::string, std::string>& defines) {
    void* library = createProgram(source, defines);
    return shared_ptr<ComputeProgramImpl>(new MetalProgram(*this, library));
}

MetalArray& MetalContext::unwrap(ArrayInterface& array) const {
    MetalArray* mtlarray;
    ComputeArray* wrapper = dynamic_cast<ComputeArray*>(&array);
    if (wrapper != NULL)
        mtlarray = dynamic_cast<MetalArray*>(&wrapper->getArray());
    else
        mtlarray = dynamic_cast<MetalArray*>(&array);
    if (mtlarray == NULL)
        throw OpenMMException("Array argument is not a MetalArray");
    return *mtlarray;
}

void MetalContext::executeKernel(void* pipelineState, const std::string& kernelName,
                                  int workUnits, int blockSize,
                                  const std::vector<MetalArray*>& arrayArgs,
                                  const std::vector<std::vector<uint8_t>>& primitiveArgs,
                                  const std::map<int, int>& threadgroupMemorySizes) {
    if (blockSize == -1)
        blockSize = ThreadBlockSize;
    int size = std::min((workUnits+blockSize-1)/blockSize, numThreadBlocks)*blockSize;
    if (size <= 0)
        return;

    // When profiling, fall back to per-dispatch sync for accurate per-kernel timing.
    if (enableKernelProfiling) {
        @autoreleasepool {
            id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mtlCommandQueue;
            // Flush any pending batch first
            flushQueue();
            id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pipelineState;
            [encoder setComputePipelineState:pso];

            int maxArgs = (int)arrayArgs.size();
            if ((int)primitiveArgs.size() > maxArgs) maxArgs = (int)primitiveArgs.size();
            for (auto& pair : threadgroupMemorySizes) { if (pair.first + 1 > maxArgs) maxArgs = pair.first + 1; }
            int bufferTableIndex = 0, threadgroupTableIndex = 0;
            for (int i = 0; i < maxArgs; i++) {
                auto tgIt = threadgroupMemorySizes.find(i);
                if (tgIt != threadgroupMemorySizes.end()) { [encoder setThreadgroupMemoryLength:(NSUInteger)tgIt->second atIndex:threadgroupTableIndex]; threadgroupTableIndex++; continue; }
                if (i < (int)arrayArgs.size() && arrayArgs[i] != NULL) { [encoder setBuffer:(__bridge id<MTLBuffer>)arrayArgs[i]->getDeviceBuffer() offset:0 atIndex:bufferTableIndex]; bufferTableIndex++; continue; }
                if (i < (int)primitiveArgs.size() && !primitiveArgs[i].empty()) { [encoder setBytes:primitiveArgs[i].data() length:(NSUInteger)primitiveArgs[i].size() atIndex:bufferTableIndex]; bufferTableIndex++; continue; }
                if (i < (int)arrayArgs.size()) { [encoder setBuffer:nil offset:0 atIndex:bufferTableIndex]; bufferTableIndex++; continue; }
            }
            [encoder dispatchThreads:MTLSizeMake((NSUInteger)size, 1, 1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)blockSize, 1, 1)];
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            if ([commandBuffer status] == MTLCommandBufferStatusError) {
                NSError* error = [commandBuffer error];
                string errorMsg = "Error executing kernel '" + kernelName + "'";
                if (error != nil) errorMsg += ": " + string([[error localizedDescription] UTF8String]);
                fprintf(stderr, "[Metal] GPU execution FAILED for '%s': %s\n", kernelName.c_str(), errorMsg.c_str());
                throw OpenMMException(errorMsg);
            }
            double gpuStart = [commandBuffer GPUStartTime];
            double gpuEnd = [commandBuffer GPUEndTime];
            if (profileStartTime == 0) { profileStartTime = (uint64_t)(gpuStart * 1e9); } else { printf(",\n"); }
            printf("{ \"pid\":1, \"tid\":1, \"ts\":%.6g, \"dur\":%g, \"ph\":\"X\", \"name\":\"%s\" }",
                   0.001 * (gpuStart * 1e9 - (double)profileStartTime), 0.001 * ((gpuEnd - gpuStart) * 1e9), kernelName.c_str());
            profilingEventCount++;
            if (profilingEventCount >= 500) printProfilingEvents();
        }
        return;
    }

    // ---- Batched dispatch (normal path) ----
    // Encode into a persistent command buffer. Multiple kernel dispatches are batched
    // into a single command buffer and only committed when flushQueue() is called
    // (before data reads, blit operations, or FFT). This eliminates the per-dispatch
    // CPU-GPU round-trip that was causing a 13x slowdown.

    try {
        // Create or reuse the current command buffer and compute encoder.
        // After blit operations, the command buffer may exist without an encoder.
        ensureCommandBuffer();
        if (currentComputeEncoder == NULL) {
            id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)currentCommandBuffer;
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            if (enc == nil)
                throw OpenMMException("Failed to create compute encoder for kernel " + kernelName);
            CFRetain((__bridge CFTypeRef)enc);
            currentComputeEncoder = (__bridge void*)enc;
        }

        id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)currentComputeEncoder;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pipelineState;
        [encoder setComputePipelineState:pso];

        // Set all arguments on the encoder.
        // Buffer/bytes and threadgroup memory use SEPARATE Metal index spaces.
        int maxArgs = (int)arrayArgs.size();
        if ((int)primitiveArgs.size() > maxArgs)
            maxArgs = (int)primitiveArgs.size();
        for (auto& pair : threadgroupMemorySizes) {
            if (pair.first + 1 > maxArgs)
                maxArgs = pair.first + 1;
        }

        int bufferTableIndex = 0;
        int threadgroupTableIndex = 0;

        for (int i = 0; i < maxArgs; i++) {
            auto tgIt = threadgroupMemorySizes.find(i);
            if (tgIt != threadgroupMemorySizes.end()) {
                [encoder setThreadgroupMemoryLength:(NSUInteger)tgIt->second
                                           atIndex:threadgroupTableIndex];
                threadgroupTableIndex++;
                continue;
            }
            if (i < (int)arrayArgs.size() && arrayArgs[i] != NULL) {
                id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)arrayArgs[i]->getDeviceBuffer();
                [encoder setBuffer:mtlBuffer offset:0 atIndex:bufferTableIndex];
                bufferTableIndex++;
                continue;
            }
            if (i < (int)primitiveArgs.size() && !primitiveArgs[i].empty()) {
                [encoder setBytes:primitiveArgs[i].data()
                           length:(NSUInteger)primitiveArgs[i].size()
                          atIndex:bufferTableIndex];
                bufferTableIndex++;
                continue;
            }
            if (i < (int)arrayArgs.size()) {
                [encoder setBuffer:nil offset:0 atIndex:bufferTableIndex];
                bufferTableIndex++;
                continue;
            }
        }

        // Log argument bindings when OPENMM_METAL_LOG_ARGS=1
        if (enableLogArgs) {
            fprintf(stderr, "[Metal] executeKernel '%s': grid=%d threadgroup=%d args=[",
                    kernelName.c_str(), size, blockSize);
            for (int i = 0; i < maxArgs; i++) {
                if (i > 0) fprintf(stderr, ", ");
                auto tgIt = threadgroupMemorySizes.find(i);
                if (tgIt != threadgroupMemorySizes.end()) {
                    fprintf(stderr, "%d:tgmem(%d bytes)", i, tgIt->second);
                } else if (i < (int)arrayArgs.size() && arrayArgs[i] != NULL) {
                    fprintf(stderr, "%d:buf('%s' %zux%d)",
                            i, arrayArgs[i]->getName().c_str(),
                            arrayArgs[i]->getSize(), arrayArgs[i]->getElementSize());
                } else if (i < (int)primitiveArgs.size() && !primitiveArgs[i].empty()) {
                    fprintf(stderr, "%d:val(%d bytes)", i, (int)primitiveArgs[i].size());
                } else {
                    fprintf(stderr, "%d:NULL", i);
                }
            }
            fprintf(stderr, "]\n");
        }

        // Dispatch threads
        MTLSize gridSize = MTLSizeMake((NSUInteger)size, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake((NSUInteger)blockSize, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        // Memory barrier between dispatches. Verified zero-cost on Apple Silicon (M4 benchmark:
        // 1065 us with vs 1065 us without), but required by Metal spec for correctness on
        // future hardware that may not guarantee coherence within a compute encoder.
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    catch (OpenMMException&) {
        throw;
    }
    catch (...) {
        throw OpenMMException("Error invoking kernel " + kernelName);
    }
}

void MetalContext::executeKernel(void* pipelineState, int workUnits, int blockSize) {
    // Simple overload with no arguments: dispatch with the pipeline state only.
    // This is used by FFT kernels that have their arguments pre-bound.
    std::vector<MetalArray*> emptyArrayArgs;
    std::vector<std::vector<uint8_t>> emptyPrimArgs;
    executeKernel(pipelineState, "kernel", workUnits, blockSize, emptyArrayArgs, emptyPrimArgs);
}

void MetalContext::executeUtilityKernel(void* pso, const std::string& name,
                                         int workUnits, int blockSize,
                                         const std::vector<MetalArray*>& bufferArgs,
                                         const std::vector<std::pair<const void*, int>>& primitiveArgValues,
                                         const std::map<int, int>& threadgroupMemorySizes) {
    // Build the argument vectors expected by executeKernel.
    // Utility kernels use a sequential argument layout: all buffer args first
    // (in order), followed by all primitive args (in order).
    // For example:
    //   clearBuffer: buffer(0), int(1)
    //   clearTwoBuffers: buffer(0), int(1), buffer(2), int(3)
    //   reduceForces: buffer(0), buffer(1), int(2), int(3)
    //   setCharges: buffer(0), buffer(1), buffer(2), int(3)

    int totalArgs = (int)bufferArgs.size() + (int)primitiveArgValues.size();
    // Also account for threadgroup memory args
    for (auto& tg : threadgroupMemorySizes) {
        if (tg.first >= totalArgs)
            totalArgs = tg.first + 1;
    }

    std::vector<MetalArray*> arrayArgsFull(totalArgs, NULL);
    std::vector<std::vector<uint8_t>> primArgsFull(totalArgs);

    // Place buffer args at indices 0..N-1
    for (int i = 0; i < (int)bufferArgs.size(); i++) {
        arrayArgsFull[i] = bufferArgs[i];
    }

    // Place primitive args at indices N..N+M-1
    int primStart = (int)bufferArgs.size();
    for (int i = 0; i < (int)primitiveArgValues.size(); i++) {
        auto& prim = primitiveArgValues[i];
        primArgsFull[primStart + i].resize(prim.second);
        memcpy(primArgsFull[primStart + i].data(), prim.first, prim.second);
    }

    executeKernel(pso, name, workUnits, blockSize, arrayArgsFull, primArgsFull, threadgroupMemorySizes);
}

void MetalContext::printProfilingEvents() {
    // In the Metal implementation, profiling data is printed inline during executeKernel.
    // This method exists for compatibility with the periodic flush pattern.
    profilingEventCount = 0;
}

int MetalContext::computeThreadBlockSize(double memory) const {
    // Apple Silicon GPUs have 32KB of threadgroup memory per threadgroup
    int maxShared = 32768;
    // On some implementations, more local memory gets used than we calculate by
    // adding up the sizes of the fields. To be safe, include a factor of 0.5.
    int max = (int) (0.5*maxShared/memory);
    if (max < 64)
        return 32;
    int threads = 64;
    while (threads+64 < max)
        threads += 64;
    return threads;
}

void MetalContext::clearBuffer(ArrayInterface& array) {
    clearBuffer(unwrap(array).getDeviceBuffer(), array.getSize()*array.getElementSize());
}

void MetalContext::clearBuffer(void* buffer, int size) {
    // Fold blit into persistent command buffer — no flush/wait needed.
    // End compute encoder → blit encoder on same CB → next executeKernel resumes compute.
    int words = size/4;
    endComputeEncoder();
    ensureCommandBuffer();
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)currentCommandBuffer;
        id<MTLBlitCommandEncoder> blitEncoder = [cb blitCommandEncoder];
        if (blitEncoder == nil)
            throw OpenMMException("Failed to create blit encoder in clearBuffer");
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
        [blitEncoder fillBuffer:mtlBuffer range:NSMakeRange(0, (NSUInteger)(words * 4)) value:0];
        [blitEncoder endEncoding];
    }
}

void MetalContext::addAutoclearBuffer(ArrayInterface& array) {
    addAutoclearBuffer(unwrap(array).getDeviceBuffer(), array.getSize()*array.getElementSize());
}

void MetalContext::addAutoclearBuffer(void* buffer, int size) {
    autoclearBuffers.push_back(buffer);
    autoclearBufferSizes.push_back(size/4);
}

void MetalContext::clearAutoclearBuffers() {
    // Fold blit into persistent command buffer — no flush/wait needed.
    // End compute encoder → blit encoder on same CB → next executeKernel resumes compute.
    endComputeEncoder();
    ensureCommandBuffer();
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)currentCommandBuffer;
        id<MTLBlitCommandEncoder> blitEncoder = [cb blitCommandEncoder];
        if (blitEncoder == nil)
            throw OpenMMException("Failed to create blit encoder in clearAutoclearBuffers");
        for (int i = 0; i < (int)autoclearBuffers.size(); i++) {
            id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)autoclearBuffers[i];
            NSUInteger clearSize = (NSUInteger)(autoclearBufferSizes[i] * 4);
            [blitEncoder fillBuffer:mtlBuffer range:NSMakeRange(0, clearSize) value:0];
        }
        [blitEncoder endEncoding];
    }
}

void MetalContext::reduceForces() {
    // Execute the reduceForces utility kernel
    int32_t pna = (int32_t)paddedNumAtoms;
    int32_t nfb = (int32_t)numForceBuffers;
    std::vector<MetalArray*> arrayArgs = {&longForceBuffer, &atomicForceBuffer, &forceBuffers};
    std::vector<std::pair<const void*, int>> primArgs = {
        {&pna, sizeof(int32_t)},
        {&nfb, sizeof(int32_t)}
    };
    executeUtilityKernel(reduceForcesPipeline, "reduceForces", paddedNumAtoms, 128,
                          arrayArgs, primArgs);
}

void MetalContext::reduceBuffer(MetalArray& array, MetalArray& longBuffer, int numBuffers) {
    int bufferSize = array.getSize()/numBuffers;
    std::vector<MetalArray*> arrayArgs = {&array, &longBuffer};
    std::vector<std::pair<const void*, int>> primArgs = {
        {&bufferSize, sizeof(int32_t)},
        {&numBuffers, sizeof(int32_t)}
    };
    executeUtilityKernel(reduceReal4Pipeline, "reduceReal4Buffer", bufferSize, 128,
                          arrayArgs, primArgs);
}

double MetalContext::reduceEnergy() {
    // Encode the reduce kernel into the persistent batch — all force/energy kernels
    // and the reduction share one command buffer, committed only at download time.
    int workGroupSize = 512;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)reduceEnergyPipeline;
    int maxThreads = (int)[pso maxTotalThreadsPerThreadgroup];
    if (workGroupSize > maxThreads)
        workGroupSize = maxThreads;
    if (workGroupSize > 512)
        workGroupSize = 512;

    int energyBufSize = (int)energyBuffer.getSize();
    int threadgroupMemSize = workGroupSize * energyBuffer.getElementSize();

    // Ensure we have a command buffer and compute encoder
    ensureCommandBuffer();
    if (currentComputeEncoder == NULL) {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)currentCommandBuffer;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil)
            throw OpenMMException("Failed to create compute encoder in reduceEnergy");
        CFRetain((__bridge CFTypeRef)enc);
        currentComputeEncoder = (__bridge void*)enc;
    }

    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)currentComputeEncoder;
    [encoder setComputePipelineState:pso];
    id<MTLBuffer> energyBuf = (__bridge id<MTLBuffer>)energyBuffer.getDeviceBuffer();
    id<MTLBuffer> energySumBuf = (__bridge id<MTLBuffer>)energySum.getDeviceBuffer();
    [encoder setBuffer:energyBuf offset:0 atIndex:0];
    [encoder setBuffer:energySumBuf offset:0 atIndex:1];
    [encoder setBytes:&energyBufSize length:sizeof(int32_t) atIndex:2];
    [encoder setBytes:&workGroupSize length:sizeof(int32_t) atIndex:3];
    [encoder setThreadgroupMemoryLength:(NSUInteger)threadgroupMemSize atIndex:0];

    int totalThreads = workGroupSize * (int)energySum.getSize();
    int clampedThreads = std::min((totalThreads + workGroupSize - 1) / workGroupSize, numThreadBlocks) * workGroupSize;
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)clampedThreads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake((NSUInteger)workGroupSize, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // download() calls flushQueue() internally — commits entire batch + waits
    energySum.download(pinnedMemory);

    double energy64 = 0;
    for (int i = 0; i < reduceEnergyThreadgroups; ++i) {
        float energy32 = ((float*)pinnedMemory)[i];
        energy64 += (double)energy32;
    }
    return energy64;
}

void MetalContext::setCharges(const vector<double>& charges) {
    if (!chargeBuffer.isInitialized())
        chargeBuffer.initialize(*this, numAtoms, useDoublePrecision ? sizeof(double) : sizeof(float), "chargeBuffer");
    vector<double> c(numAtoms);
    for (int i = 0; i < numAtoms; i++)
        c[i] = charges[i];
    chargeBuffer.upload(c, true);

    std::vector<MetalArray*> arrayArgs = {&chargeBuffer, &posq, &atomIndexDevice};
    std::vector<std::pair<const void*, int>> primArgs = {
        {&numAtoms, sizeof(int32_t)}
    };
    executeUtilityKernel(setChargesPipeline, "setCharges", numAtoms, -1,
                          arrayArgs, primArgs);
}

bool MetalContext::requestPosqCharges() {
    bool allow = !hasAssignedPosqCharges;
    hasAssignedPosqCharges = true;
    return allow;
}

void MetalContext::addEnergyParameterDerivative(const string& param) {
    // See if this parameter has already been registered.

    for (int i = 0; i < (int)energyParamDerivNames.size(); i++)
        if (param == energyParamDerivNames[i])
            return;
    energyParamDerivNames.push_back(param);
}

void MetalContext::flushQueue() {
    // Commit the current batched command buffer and wait for GPU completion.
    // Must be called before any CPU-side read of GPU data (download, reduceEnergy,
    // blit operations, FFT, etc.).
    if (currentComputeEncoder != NULL) {
        id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)currentComputeEncoder;
        [encoder endEncoding];
        CFRelease(currentComputeEncoder);
        currentComputeEncoder = NULL;
    }
    if (currentCommandBuffer != NULL) {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)currentCommandBuffer;
        [cb commit];
        [cb waitUntilCompleted];

        if ([cb status] == MTLCommandBufferStatusError) {
            NSError* error = [cb error];
            string errorMsg = "GPU batch execution failed";
            if (error != nil) {
                errorMsg += ": " + string([[error localizedDescription] UTF8String]);
                if ([error localizedFailureReason] != nil)
                    errorMsg += "\nReason: " + string([[error localizedFailureReason] UTF8String]);
            }
            fprintf(stderr, "[Metal] %s\n", errorMsg.c_str());
            CFRelease(currentCommandBuffer);
            currentCommandBuffer = NULL;
            throw OpenMMException(errorMsg);
        }

        CFRelease(currentCommandBuffer);
        currentCommandBuffer = NULL;
    }
}

void MetalContext::ensureCommandBuffer() {
    if (currentCommandBuffer == NULL) {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mtlCommandQueue;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        if (cb == nil)
            throw OpenMMException("Failed to create Metal command buffer");
        CFRetain((__bridge CFTypeRef)cb);
        currentCommandBuffer = (__bridge void*)cb;
    }
}

void* MetalContext::getCommandBufferForBlit() {
    endComputeEncoder();
    ensureCommandBuffer();
    return currentCommandBuffer;
}

void MetalContext::endComputeEncoder() {
    if (currentComputeEncoder != NULL) {
        id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)currentComputeEncoder;
        [encoder endEncoding];
        CFRelease(currentComputeEncoder);
        currentComputeEncoder = NULL;
    }
}

// getMaxThreadBlockSize() and getIsCPU() are defined inline in MetalContext.h
