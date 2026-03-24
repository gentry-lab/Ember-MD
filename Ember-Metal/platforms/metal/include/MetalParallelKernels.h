#ifndef OPENMM_OPENCLPARALLELKERNELS_H_
#define OPENMM_OPENCLPARALLELKERNELS_H_

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

#include "MetalPlatform.h"
#include "MetalContext.h"
#include "MetalKernels.h"
#include "openmm/common/CommonKernels.h"

namespace OpenMM {

class MetalParallelCalcForcesAndEnergyKernel : public CalcForcesAndEnergyKernel {
public:
    MetalParallelCalcForcesAndEnergyKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data);
    ~MetalParallelCalcForcesAndEnergyKernel();
    MetalCalcForcesAndEnergyKernel& getKernel(int index) {
        return dynamic_cast<MetalCalcForcesAndEnergyKernel&>(kernels[index].getImpl());
    }
    void initialize(const System& system);
    void beginComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups);
    double finishComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups, bool& valid);
private:
    class BeginComputationTask;
    class FinishComputationTask;
    MetalPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
    std::vector<long long> completionTimes;
    std::vector<double> contextNonbondedFractions;
    std::vector<int> tileCounts;
    MetalArray contextForces;
    void* pinnedPositionBuffer;     // id<MTLBuffer>
    void* pinnedForceBuffer;        // id<MTLBuffer>
    void* pinnedPositionMemory;
    void* pinnedForceMemory;
};

class MetalParallelCalcHarmonicBondForceKernel : public CalcHarmonicBondForceKernel {
public:
    MetalParallelCalcHarmonicBondForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system);
    CommonCalcHarmonicBondForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcHarmonicBondForceKernel&>(kernels[index].getImpl());
    }
    void initialize(const System& system, const HarmonicBondForce& force);
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(ContextImpl& context, const HarmonicBondForce& force);
private:
    class Task;
    MetalPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

class MetalParallelCalcCustomBondForceKernel : public CalcCustomBondForceKernel {
public:
    MetalParallelCalcCustomBondForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system);
    CommonCalcCustomBondForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcCustomBondForceKernel&>(kernels[index].getImpl());
    }
    void initialize(const System& system, const CustomBondForce& force);
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(ContextImpl& context, const CustomBondForce& force);
private:
    class Task;
    MetalPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

class MetalParallelCalcHarmonicAngleForceKernel : public CalcHarmonicAngleForceKernel {
public:
    MetalParallelCalcHarmonicAngleForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system);
    CommonCalcHarmonicAngleForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcHarmonicAngleForceKernel&>(kernels[index].getImpl());
    }
    void initialize(const System& system, const HarmonicAngleForce& force);
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(ContextImpl& context, const HarmonicAngleForce& force);
private:
    class Task;
    MetalPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

class MetalParallelCalcCustomAngleForceKernel : public CalcCustomAngleForceKernel {
public:
    MetalParallelCalcCustomAngleForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system);
    CommonCalcCustomAngleForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcCustomAngleForceKernel&>(kernels[index].getImpl());
    }
    void initialize(const System& system, const CustomAngleForce& force);
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(ContextImpl& context, const CustomAngleForce& force);
private:
    class Task;
    MetalPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

class MetalParallelCalcPeriodicTorsionForceKernel : public CalcPeriodicTorsionForceKernel {
public:
    MetalParallelCalcPeriodicTorsionForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system);
    CommonCalcPeriodicTorsionForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcPeriodicTorsionForceKernel&>(kernels[index].getImpl());
    }
    void initialize(const System& system, const PeriodicTorsionForce& force);
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    class Task;
    void copyParametersToContext(ContextImpl& context, const PeriodicTorsionForce& force);
private:
    MetalPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

class MetalParallelCalcRBTorsionForceKernel : public CalcRBTorsionForceKernel {
public:
    MetalParallelCalcRBTorsionForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system);
    CommonCalcRBTorsionForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcRBTorsionForceKernel&>(kernels[index].getImpl());
    }
    void initialize(const System& system, const RBTorsionForce& force);
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(ContextImpl& context, const RBTorsionForce& force);
private:
    class Task;
    MetalPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

class MetalParallelCalcCMAPTorsionForceKernel : public CalcCMAPTorsionForceKernel {
public:
    MetalParallelCalcCMAPTorsionForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system);
    CommonCalcCMAPTorsionForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcCMAPTorsionForceKernel&>(kernels[index].getImpl());
    }
    void initialize(const System& system, const CMAPTorsionForce& force);
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(ContextImpl& context, const CMAPTorsionForce& force);
private:
    class Task;
    MetalPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

class MetalParallelCalcCustomTorsionForceKernel : public CalcCustomTorsionForceKernel {
public:
    MetalParallelCalcCustomTorsionForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system);
    CommonCalcCustomTorsionForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcCustomTorsionForceKernel&>(kernels[index].getImpl());
    }
    void initialize(const System& system, const CustomTorsionForce& force);
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(ContextImpl& context, const CustomTorsionForce& force);
private:
    class Task;
    MetalPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

class MetalParallelCalcNonbondedForceKernel : public CalcNonbondedForceKernel {
public:
    MetalParallelCalcNonbondedForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system);
    MetalCalcNonbondedForceKernel& getKernel(int index) {
        return dynamic_cast<MetalCalcNonbondedForceKernel&>(kernels[index].getImpl());
    }
    void initialize(const System& system, const NonbondedForce& force);
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal);
    void copyParametersToContext(ContextImpl& context, const NonbondedForce& force);
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    void getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
private:
    class Task;
    MetalPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

class MetalParallelCalcCustomNonbondedForceKernel : public CalcCustomNonbondedForceKernel {
public:
    MetalParallelCalcCustomNonbondedForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system);
    CommonCalcCustomNonbondedForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcCustomNonbondedForceKernel&>(kernels[index].getImpl());
    }
    void initialize(const System& system, const CustomNonbondedForce& force);
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(ContextImpl& context, const CustomNonbondedForce& force);
private:
    class Task;
    MetalPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

class MetalParallelCalcCustomExternalForceKernel : public CalcCustomExternalForceKernel {
public:
    MetalParallelCalcCustomExternalForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system);
    CommonCalcCustomExternalForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcCustomExternalForceKernel&>(kernels[index].getImpl());
    }
    void initialize(const System& system, const CustomExternalForce& force);
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(ContextImpl& context, const CustomExternalForce& force);
private:
    class Task;
    MetalPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

class MetalParallelCalcCustomHbondForceKernel : public CalcCustomHbondForceKernel {
public:
    MetalParallelCalcCustomHbondForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system);
    CommonCalcCustomHbondForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcCustomHbondForceKernel&>(kernels[index].getImpl());
    }
    void initialize(const System& system, const CustomHbondForce& force);
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(ContextImpl& context, const CustomHbondForce& force);
private:
    class Task;
    MetalPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

class MetalParallelCalcCustomCompoundBondForceKernel : public CalcCustomCompoundBondForceKernel {
public:
    MetalParallelCalcCustomCompoundBondForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system);
    CommonCalcCustomCompoundBondForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcCustomCompoundBondForceKernel&>(kernels[index].getImpl());
    }
    void initialize(const System& system, const CustomCompoundBondForce& force);
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(ContextImpl& context, const CustomCompoundBondForce& force);
private:
    class Task;
    MetalPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

} // namespace OpenMM

#endif /*OPENMM_OPENCLPARALLELKERNELS_H_*/
