/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2011-2019 Stanford University and the Authors.      *
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
#include "MetalParallelKernels.h"

using namespace OpenMM;
using namespace std;

/**
 * Get the current clock time, measured in microseconds.
 */
#ifdef _MSC_VER
    #include <Windows.h>
    static long long getTime() {
        FILETIME ft;
        GetSystemTimeAsFileTime(&ft);
        ULARGE_INTEGER result;
        result.LowPart = ft.dwLowDateTime;
        result.HighPart = ft.dwHighDateTime;
        return result.QuadPart/10;
    }
#else
    #include <sys/time.h>
    static long long getTime() {
        struct timeval tod;
        gettimeofday(&tod, 0);
        return 1000000*tod.tv_sec+tod.tv_usec;
    }
#endif

class MetalParallelCalcForcesAndEnergyKernel::BeginComputationTask : public MetalContext::WorkTask {
public:
    BeginComputationTask(ContextImpl& context, MetalContext& cl, MetalCalcForcesAndEnergyKernel& kernel,
            bool includeForce, bool includeEnergy, int groups, void* pinnedMemory, int& numTiles) : context(context), cl(cl), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), groups(groups), pinnedMemory(pinnedMemory), numTiles(numTiles) {
    }
    void execute() {
        // Copy coordinates over to this device and execute the kernel.
        if (cl.getContextIndex() > 0) {
            // Upload position data to this device's posq buffer
            cl.getPosq().upload(pinnedMemory);
        }
        kernel.beginComputation(context, includeForce, includeEnergy, groups);
        if (cl.getNonbondedUtilities().getUsePeriodic())
            cl.getNonbondedUtilities().getInteractionCount().download(&numTiles, false);
    }
private:
    ContextImpl& context;
    MetalContext& cl;
    MetalCalcForcesAndEnergyKernel& kernel;
    bool includeForce, includeEnergy;
    int groups;
    void* pinnedMemory;
    int& numTiles;
};

class MetalParallelCalcForcesAndEnergyKernel::FinishComputationTask : public MetalContext::WorkTask {
public:
    FinishComputationTask(ContextImpl& context, MetalContext& cl, MetalCalcForcesAndEnergyKernel& kernel,
            bool includeForce, bool includeEnergy, int groups, double& energy, long long& completionTime, void* pinnedMemory, bool& valid, int& numTiles) :
            context(context), cl(cl), kernel(kernel), includeForce(includeForce), includeEnergy(includeEnergy), groups(groups), energy(energy),
            completionTime(completionTime), pinnedMemory(pinnedMemory), valid(valid), numTiles(numTiles) {
    }
    void execute() {
        energy += kernel.finishComputation(context, includeForce, includeEnergy, groups, valid);
        if (includeForce) {
            if (cl.getContextIndex() > 0) {
                int numAtoms = cl.getPaddedNumAtoms();
                void* dest = (cl.getUseDoublePrecision() ? (void*) &((mm_double4*) pinnedMemory)[(cl.getContextIndex()-1)*numAtoms] : (void*) &((mm_float4*) pinnedMemory)[(cl.getContextIndex()-1)*numAtoms]);
                cl.getForce().download(dest);
            }
            // Metal kernel execution is synchronous, no need to call finish()
        }
        completionTime = getTime();
        if (cl.getNonbondedUtilities().getUsePeriodic() && numTiles > cl.getNonbondedUtilities().getInteractingTiles().getSize()) {
            valid = false;
            cl.getNonbondedUtilities().updateNeighborListSize();
        }
    }
private:
    ContextImpl& context;
    MetalContext& cl;
    MetalCalcForcesAndEnergyKernel& kernel;
    bool includeForce, includeEnergy;
    int groups;
    double& energy;
    long long& completionTime;
    void* pinnedMemory;
    bool& valid;
    int& numTiles;
};

MetalParallelCalcForcesAndEnergyKernel::MetalParallelCalcForcesAndEnergyKernel(string name, const Platform& platform, MetalPlatform::PlatformData& data) :
        CalcForcesAndEnergyKernel(name, platform), data(data), completionTimes(data.contexts.size()), contextNonbondedFractions(data.contexts.size()),
        tileCounts(data.contexts.size()), pinnedPositionBuffer(NULL), pinnedPositionMemory(NULL), pinnedForceBuffer(NULL), pinnedForceMemory(NULL) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new MetalCalcForcesAndEnergyKernel(name, platform, *data.contexts[i])));
}

MetalParallelCalcForcesAndEnergyKernel::~MetalParallelCalcForcesAndEnergyKernel() {
    if (pinnedPositionBuffer != NULL)
        CFRelease(pinnedPositionBuffer);
    if (pinnedForceBuffer != NULL)
        CFRelease(pinnedForceBuffer);
}

void MetalParallelCalcForcesAndEnergyKernel::initialize(const System& system) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).initialize(system);
    for (int i = 0; i < (int) contextNonbondedFractions.size(); i++)
        contextNonbondedFractions[i] = 1/(double) contextNonbondedFractions.size();
}

void MetalParallelCalcForcesAndEnergyKernel::beginComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups) {
    MetalContext& cl0 = *data.contexts[0];
    int elementSize = (cl0.getUseDoublePrecision() ? sizeof(mm_double4) : sizeof(mm_float4));
    if (!contextForces.isInitialized()) {
        contextForces.initialize<mm_float4>(cl0, cl0.getForceBuffers().getDeviceBuffer(),
                data.contexts.size()*cl0.getPaddedNumAtoms(), "contextForces");
        int bufferBytes = (data.contexts.size()-1)*cl0.getPaddedNumAtoms()*elementSize;
        if (bufferBytes > 0) {
            // Allocate shared memory buffers for multi-device data transfer
            @autoreleasepool {
                id<MTLDevice> device = (__bridge id<MTLDevice>)cl0.getMTLDevice();
                id<MTLBuffer> posBuf = [device newBufferWithLength:bufferBytes options:MTLResourceStorageModeShared];
                pinnedPositionBuffer = (__bridge void*)posBuf;
                CFRetain(pinnedPositionBuffer);
                pinnedPositionMemory = [posBuf contents];
                id<MTLBuffer> forceBuf = [device newBufferWithLength:bufferBytes options:MTLResourceStorageModeShared];
                pinnedForceBuffer = (__bridge void*)forceBuf;
                CFRetain(pinnedForceBuffer);
                pinnedForceMemory = [forceBuf contents];
            }
        }
    }

    // Copy coordinates over to each device and execute the kernel.
    if (pinnedPositionMemory != NULL)
        cl0.getPosq().download(pinnedPositionMemory);
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        data.contextEnergy[i] = 0.0;
        MetalContext& cl = *data.contexts[i];
        ComputeContext::WorkThread& thread = cl.getWorkThread();
        thread.addTask(new BeginComputationTask(context, cl, getKernel(i), includeForce, includeEnergy, groups, pinnedPositionMemory, tileCounts[i]));
    }
}

double MetalParallelCalcForcesAndEnergyKernel::finishComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups, bool& valid) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        ComputeContext::WorkThread& thread = cl.getWorkThread();
        thread.addTask(new FinishComputationTask(context, cl, getKernel(i), includeForce, includeEnergy, groups, data.contextEnergy[i], completionTimes[i], pinnedForceMemory, valid, tileCounts[i]));
    }
    data.syncContexts();
    double energy = 0.0;
    for (int i = 0; i < (int) data.contextEnergy.size(); i++)
        energy += data.contextEnergy[i];
    if (includeForce && valid) {
        MetalContext& cl = *data.contexts[0];
        int numAtoms = cl.getPaddedNumAtoms();
        int elementSize = (cl.getUseDoublePrecision() ? sizeof(mm_double4) : sizeof(mm_float4));
        // Upload forces from other contexts
        if (data.contexts.size() > 1 && pinnedForceMemory != NULL) {
            contextForces.uploadSubArray(pinnedForceMemory, numAtoms, numAtoms*(data.contexts.size()-1));
        }
        cl.reduceBuffer(contextForces, cl.getLongForceBuffer(), data.contexts.size());

        // Balance work between the contexts
        if (cl.getComputeForceCount() < 200) {
            int firstIndex = 0, lastIndex = 0;
            for (int i = 0; i < (int) completionTimes.size(); i++) {
                if (completionTimes[i] < completionTimes[firstIndex])
                    firstIndex = i;
                if (completionTimes[i] > completionTimes[lastIndex])
                    lastIndex = i;
            }
            double fractionToTransfer = min(0.001, contextNonbondedFractions[lastIndex]);
            contextNonbondedFractions[firstIndex] += fractionToTransfer;
            contextNonbondedFractions[lastIndex] -= fractionToTransfer;
            double startFraction = 0.0;
            for (int i = 0; i < (int) contextNonbondedFractions.size(); i++) {
                double endFraction = startFraction+contextNonbondedFractions[i];
                if (i == contextNonbondedFractions.size()-1)
                    endFraction = 1.0;
                data.contexts[i]->getNonbondedUtilities().setAtomBlockRange(startFraction, endFraction);
                startFraction = endFraction;
            }
        }
    }
    return energy;
}

// The rest of the parallel kernel classes are identical to the original --
// they don't use any cl:: types directly, only MetalContext and ComputeContext references.

class MetalParallelCalcHarmonicBondForceKernel::Task : public MetalContext::WorkTask {
public:
    Task(ContextImpl& context, CommonCalcHarmonicBondForceKernel& kernel, bool includeForce,
            bool includeEnergy, double& energy) : context(context), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), energy(energy) {
    }
    void execute() {
        energy += kernel.execute(context, includeForce, includeEnergy);
    }
private:
    ContextImpl& context;
    CommonCalcHarmonicBondForceKernel& kernel;
    bool includeForce, includeEnergy;
    double& energy;
};

MetalParallelCalcHarmonicBondForceKernel::MetalParallelCalcHarmonicBondForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcHarmonicBondForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcHarmonicBondForceKernel(name, platform, *data.contexts[i], system)));
}

void MetalParallelCalcHarmonicBondForceKernel::initialize(const System& system, const HarmonicBondForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).initialize(system, force);
}

double MetalParallelCalcHarmonicBondForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        ComputeContext::WorkThread& thread = cl.getWorkThread();
        thread.addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}

void MetalParallelCalcHarmonicBondForceKernel::copyParametersToContext(ContextImpl& context, const HarmonicBondForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).copyParametersToContext(context, force);
}

class MetalParallelCalcCustomBondForceKernel::Task : public MetalContext::WorkTask {
public:
    Task(ContextImpl& context, CommonCalcCustomBondForceKernel& kernel, bool includeForce,
            bool includeEnergy, double& energy) : context(context), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), energy(energy) {
    }
    void execute() {
        energy += kernel.execute(context, includeForce, includeEnergy);
    }
private:
    ContextImpl& context;
    CommonCalcCustomBondForceKernel& kernel;
    bool includeForce, includeEnergy;
    double& energy;
};

MetalParallelCalcCustomBondForceKernel::MetalParallelCalcCustomBondForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcCustomBondForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcCustomBondForceKernel(name, platform, *data.contexts[i], system)));
}

void MetalParallelCalcCustomBondForceKernel::initialize(const System& system, const CustomBondForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).initialize(system, force);
}

double MetalParallelCalcCustomBondForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        ComputeContext::WorkThread& thread = cl.getWorkThread();
        thread.addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}

void MetalParallelCalcCustomBondForceKernel::copyParametersToContext(ContextImpl& context, const CustomBondForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).copyParametersToContext(context, force);
}

// Remaining parallel kernel Task classes follow the same pattern -- no cl:: usage.
// They are identical to the original except for using MetalContext instead of cl:: types.

#define DEFINE_PARALLEL_KERNEL(ClassName, BaseKernel, ForceClass) \
class ClassName::Task : public MetalContext::WorkTask { \
public: \
    Task(ContextImpl& context, BaseKernel& kernel, bool includeForce, \
            bool includeEnergy, double& energy) : context(context), kernel(kernel), \
            includeForce(includeForce), includeEnergy(includeEnergy), energy(energy) { \
    } \
    void execute() { \
        energy += kernel.execute(context, includeForce, includeEnergy); \
    } \
private: \
    ContextImpl& context; \
    BaseKernel& kernel; \
    bool includeForce, includeEnergy; \
    double& energy; \
};

DEFINE_PARALLEL_KERNEL(MetalParallelCalcHarmonicAngleForceKernel, CommonCalcHarmonicAngleForceKernel, HarmonicAngleForce)

MetalParallelCalcHarmonicAngleForceKernel::MetalParallelCalcHarmonicAngleForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcHarmonicAngleForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcHarmonicAngleForceKernel(name, platform, *data.contexts[i], system)));
}
void MetalParallelCalcHarmonicAngleForceKernel::initialize(const System& system, const HarmonicAngleForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++) getKernel(i).initialize(system, force);
}
double MetalParallelCalcHarmonicAngleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        cl.getWorkThread().addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}
void MetalParallelCalcHarmonicAngleForceKernel::copyParametersToContext(ContextImpl& context, const HarmonicAngleForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++) getKernel(i).copyParametersToContext(context, force);
}

DEFINE_PARALLEL_KERNEL(MetalParallelCalcCustomAngleForceKernel, CommonCalcCustomAngleForceKernel, CustomAngleForce)

MetalParallelCalcCustomAngleForceKernel::MetalParallelCalcCustomAngleForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcCustomAngleForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcCustomAngleForceKernel(name, platform, *data.contexts[i], system)));
}
void MetalParallelCalcCustomAngleForceKernel::initialize(const System& system, const CustomAngleForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++) getKernel(i).initialize(system, force);
}
double MetalParallelCalcCustomAngleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        cl.getWorkThread().addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}
void MetalParallelCalcCustomAngleForceKernel::copyParametersToContext(ContextImpl& context, const CustomAngleForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++) getKernel(i).copyParametersToContext(context, force);
}

DEFINE_PARALLEL_KERNEL(MetalParallelCalcPeriodicTorsionForceKernel, CommonCalcPeriodicTorsionForceKernel, PeriodicTorsionForce)

MetalParallelCalcPeriodicTorsionForceKernel::MetalParallelCalcPeriodicTorsionForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcPeriodicTorsionForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcPeriodicTorsionForceKernel(name, platform, *data.contexts[i], system)));
}
void MetalParallelCalcPeriodicTorsionForceKernel::initialize(const System& system, const PeriodicTorsionForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++) getKernel(i).initialize(system, force);
}
double MetalParallelCalcPeriodicTorsionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        cl.getWorkThread().addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}
void MetalParallelCalcPeriodicTorsionForceKernel::copyParametersToContext(ContextImpl& context, const PeriodicTorsionForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++) getKernel(i).copyParametersToContext(context, force);
}

DEFINE_PARALLEL_KERNEL(MetalParallelCalcRBTorsionForceKernel, CommonCalcRBTorsionForceKernel, RBTorsionForce)

MetalParallelCalcRBTorsionForceKernel::MetalParallelCalcRBTorsionForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcRBTorsionForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcRBTorsionForceKernel(name, platform, *data.contexts[i], system)));
}
void MetalParallelCalcRBTorsionForceKernel::initialize(const System& system, const RBTorsionForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++) getKernel(i).initialize(system, force);
}
double MetalParallelCalcRBTorsionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        cl.getWorkThread().addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}
void MetalParallelCalcRBTorsionForceKernel::copyParametersToContext(ContextImpl& context, const RBTorsionForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++) getKernel(i).copyParametersToContext(context, force);
}

DEFINE_PARALLEL_KERNEL(MetalParallelCalcCMAPTorsionForceKernel, CommonCalcCMAPTorsionForceKernel, CMAPTorsionForce)

MetalParallelCalcCMAPTorsionForceKernel::MetalParallelCalcCMAPTorsionForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcCMAPTorsionForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcCMAPTorsionForceKernel(name, platform, *data.contexts[i], system)));
}
void MetalParallelCalcCMAPTorsionForceKernel::initialize(const System& system, const CMAPTorsionForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++) getKernel(i).initialize(system, force);
}
double MetalParallelCalcCMAPTorsionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        cl.getWorkThread().addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}
void MetalParallelCalcCMAPTorsionForceKernel::copyParametersToContext(ContextImpl& context, const CMAPTorsionForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++) getKernel(i).copyParametersToContext(context, force);
}

DEFINE_PARALLEL_KERNEL(MetalParallelCalcCustomTorsionForceKernel, CommonCalcCustomTorsionForceKernel, CustomTorsionForce)

MetalParallelCalcCustomTorsionForceKernel::MetalParallelCalcCustomTorsionForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcCustomTorsionForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcCustomTorsionForceKernel(name, platform, *data.contexts[i], system)));
}
void MetalParallelCalcCustomTorsionForceKernel::initialize(const System& system, const CustomTorsionForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++) getKernel(i).initialize(system, force);
}
double MetalParallelCalcCustomTorsionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        cl.getWorkThread().addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}
void MetalParallelCalcCustomTorsionForceKernel::copyParametersToContext(ContextImpl& context, const CustomTorsionForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++) getKernel(i).copyParametersToContext(context, force);
}

// NonbondedForce has a different Task signature
class MetalParallelCalcNonbondedForceKernel::Task : public MetalContext::WorkTask {
public:
    Task(ContextImpl& context, MetalCalcNonbondedForceKernel& kernel, bool includeForce,
            bool includeEnergy, bool includeDirect, bool includeReciprocal, double& energy) : context(context), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), includeDirect(includeDirect), includeReciprocal(includeReciprocal), energy(energy) {
    }
    void execute() {
        energy += kernel.execute(context, includeForce, includeEnergy, includeDirect, includeReciprocal);
    }
private:
    ContextImpl& context;
    MetalCalcNonbondedForceKernel& kernel;
    bool includeForce, includeEnergy, includeDirect, includeReciprocal;
    double& energy;
};

MetalParallelCalcNonbondedForceKernel::MetalParallelCalcNonbondedForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcNonbondedForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new MetalCalcNonbondedForceKernel(name, platform, *data.contexts[i], system)));
}

void MetalParallelCalcNonbondedForceKernel::initialize(const System& system, const NonbondedForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).initialize(system, force);
}

double MetalParallelCalcNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        cl.getWorkThread().addTask(new Task(context, getKernel(i), includeForces, includeEnergy, includeDirect, includeReciprocal, data.contextEnergy[i]));
    }
    return 0.0;
}

void MetalParallelCalcNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const NonbondedForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).copyParametersToContext(context, force);
}

void MetalParallelCalcNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const MetalCalcNonbondedForceKernel&>(kernels[0].getImpl()).getPMEParameters(alpha, nx, ny, nz);
}

void MetalParallelCalcNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const MetalCalcNonbondedForceKernel&>(kernels[0].getImpl()).getLJPMEParameters(alpha, nx, ny, nz);
}

DEFINE_PARALLEL_KERNEL(MetalParallelCalcCustomNonbondedForceKernel, CommonCalcCustomNonbondedForceKernel, CustomNonbondedForce)

MetalParallelCalcCustomNonbondedForceKernel::MetalParallelCalcCustomNonbondedForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcCustomNonbondedForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcCustomNonbondedForceKernel(name, platform, *data.contexts[i], system)));
}
void MetalParallelCalcCustomNonbondedForceKernel::initialize(const System& system, const CustomNonbondedForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++) getKernel(i).initialize(system, force);
}
double MetalParallelCalcCustomNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        cl.getWorkThread().addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}
void MetalParallelCalcCustomNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const CustomNonbondedForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++) getKernel(i).copyParametersToContext(context, force);
}

DEFINE_PARALLEL_KERNEL(MetalParallelCalcCustomExternalForceKernel, CommonCalcCustomExternalForceKernel, CustomExternalForce)

MetalParallelCalcCustomExternalForceKernel::MetalParallelCalcCustomExternalForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcCustomExternalForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcCustomExternalForceKernel(name, platform, *data.contexts[i], system)));
}
void MetalParallelCalcCustomExternalForceKernel::initialize(const System& system, const CustomExternalForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++) getKernel(i).initialize(system, force);
}
double MetalParallelCalcCustomExternalForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        cl.getWorkThread().addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}
void MetalParallelCalcCustomExternalForceKernel::copyParametersToContext(ContextImpl& context, const CustomExternalForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++) getKernel(i).copyParametersToContext(context, force);
}

DEFINE_PARALLEL_KERNEL(MetalParallelCalcCustomHbondForceKernel, CommonCalcCustomHbondForceKernel, CustomHbondForce)

MetalParallelCalcCustomHbondForceKernel::MetalParallelCalcCustomHbondForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcCustomHbondForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcCustomHbondForceKernel(name, platform, *data.contexts[i], system)));
}
void MetalParallelCalcCustomHbondForceKernel::initialize(const System& system, const CustomHbondForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++) getKernel(i).initialize(system, force);
}
double MetalParallelCalcCustomHbondForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        cl.getWorkThread().addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}
void MetalParallelCalcCustomHbondForceKernel::copyParametersToContext(ContextImpl& context, const CustomHbondForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++) getKernel(i).copyParametersToContext(context, force);
}

DEFINE_PARALLEL_KERNEL(MetalParallelCalcCustomCompoundBondForceKernel, CommonCalcCustomCompoundBondForceKernel, CustomCompoundBondForce)

MetalParallelCalcCustomCompoundBondForceKernel::MetalParallelCalcCustomCompoundBondForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcCustomCompoundBondForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcCustomCompoundBondForceKernel(name, platform, *data.contexts[i], system)));
}
void MetalParallelCalcCustomCompoundBondForceKernel::initialize(const System& system, const CustomCompoundBondForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++) getKernel(i).initialize(system, force);
}
double MetalParallelCalcCustomCompoundBondForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        cl.getWorkThread().addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}
void MetalParallelCalcCustomCompoundBondForceKernel::copyParametersToContext(ContextImpl& context, const CustomCompoundBondForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++) getKernel(i).copyParametersToContext(context, force);
}
