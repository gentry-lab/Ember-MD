#ifndef OPENMM_OPENCLKERNELS_H_
#define OPENMM_OPENCLKERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2022 Stanford University and the Authors.      *
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
#include "MetalArray.h"
#include "MetalContext.h"
#include "MetalFFT3D.h"
#include "MetalSort.h"
#include "openmm/kernels.h"
#include "openmm/System.h"
#include "openmm/common/CommonKernels.h"

namespace OpenMM {

/**
 * This kernel is invoked at the beginning and end of force and energy computations.
 */
class MetalCalcForcesAndEnergyKernel : public CalcForcesAndEnergyKernel {
public:
    MetalCalcForcesAndEnergyKernel(std::string name, const Platform& platform, MetalContext& cl) : CalcForcesAndEnergyKernel(name, platform), cl(cl) {
    }
    void initialize(const System& system);
    void beginComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups);
    double finishComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups, bool& valid);
private:
   MetalContext& cl;
};

/**
 * This kernel provides methods for setting and retrieving various state data.
 */
class MetalUpdateStateDataKernel : public UpdateStateDataKernel {
public:
    MetalUpdateStateDataKernel(std::string name, const Platform& platform, MetalContext& cl) : UpdateStateDataKernel(name, platform), cl(cl) {
    }
    void initialize(const System& system);
    double getTime(const ContextImpl& context) const;
    void setTime(ContextImpl& context, double time);
    long long getStepCount(const ContextImpl& context) const;
    void setStepCount(const ContextImpl& context, long long count);
    void getPositions(ContextImpl& context, std::vector<Vec3>& positions);
    void setPositions(ContextImpl& context, const std::vector<Vec3>& positions);
    void getVelocities(ContextImpl& context, std::vector<Vec3>& velocities);
    void setVelocities(ContextImpl& context, const std::vector<Vec3>& velocities);
    void computeShiftedVelocities(ContextImpl& context, double timeShift, std::vector<Vec3>& velocities);
    void getForces(ContextImpl& context, std::vector<Vec3>& forces);
    void getEnergyParameterDerivatives(ContextImpl& context, std::map<std::string, double>& derivs);
    void getPeriodicBoxVectors(ContextImpl& context, Vec3& a, Vec3& b, Vec3& c) const;
    void setPeriodicBoxVectors(ContextImpl& context, const Vec3& a, const Vec3& b, const Vec3& c);
    void createCheckpoint(ContextImpl& context, std::ostream& stream);
    void loadCheckpoint(ContextImpl& context, std::istream& stream);
private:
    MetalContext& cl;
};

/**
 * This kernel is invoked by NonbondedForce to calculate the forces acting on the system.
 */
class MetalCalcNonbondedForceKernel : public CalcNonbondedForceKernel {
public:
    MetalCalcNonbondedForceKernel(std::string name, const Platform& platform, MetalContext& cl, const System& system) : CalcNonbondedForceKernel(name, platform),
            hasInitializedKernel(false), cl(cl), sort(NULL), fft(NULL), dispersionFft(NULL), pmeio(NULL), usePmeQueue(false) {
    }
    ~MetalCalcNonbondedForceKernel();
    void initialize(const System& system, const NonbondedForce& force);
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal);
    void copyParametersToContext(ContextImpl& context, const NonbondedForce& force);
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    void getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
private:
    class SortTrait : public MetalSort::SortTrait {
        int getDataSize() const {return 8;}
        int getKeySize() const {return 4;}
        const char* getDataType() const {return "int2";}
        const char* getKeyType() const {return "int";}
        const char* getMinKey() const {return "INT_MIN";}
        const char* getMaxKey() const {return "INT_MAX";}
        const char* getMaxValue() const {return "(int2) (INT_MAX, INT_MAX)";}
        const char* getSortKey() const {return "value.y";}
    };
    class ForceInfo;
    class PmeIO;
    class PmePreComputation;
    class PmePostComputation;
    class SyncQueuePreComputation;
    class SyncQueuePostComputation;
    MetalContext& cl;
    ForceInfo* info;
    bool hasInitializedKernel;
    MetalArray charges;
    MetalArray sigmaEpsilon;
    MetalArray exceptionParams;
    MetalArray exclusionAtoms;
    MetalArray exclusionParams;
    MetalArray baseParticleParams;
    MetalArray baseExceptionParams;
    MetalArray particleParamOffsets;
    MetalArray exceptionParamOffsets;
    MetalArray particleOffsetIndices;
    MetalArray exceptionOffsetIndices;
    MetalArray globalParams;
    MetalArray cosSinSums;
    MetalArray pmeGrid1;
    MetalArray pmeGrid2;
    MetalArray pmeBsplineModuliX;
    MetalArray pmeBsplineModuliY;
    MetalArray pmeBsplineModuliZ;
    MetalArray pmeDispersionBsplineModuliX;
    MetalArray pmeDispersionBsplineModuliY;
    MetalArray pmeDispersionBsplineModuliZ;
    MetalArray pmeBsplineTheta;
    MetalArray pmeAtomRange;
    MetalArray pmeAtomGridIndex;
    MetalArray pmeEnergyBuffer;
    MetalSort* sort;
    MetalFFT3D* fft;
    MetalFFT3D* dispersionFft;
    Kernel cpuPme;
    PmeIO* pmeio;
    SyncQueuePostComputation* syncQueue;
    ComputeKernel computeParamsKernel, computeExclusionParamsKernel;
    ComputeKernel ewaldSumsKernel;
    ComputeKernel ewaldForcesKernel;
    ComputeKernel pmeAtomRangeKernel;
    ComputeKernel pmeDispersionAtomRangeKernel;
    ComputeKernel pmeZIndexKernel;
    ComputeKernel pmeDispersionZIndexKernel;
    ComputeKernel pmeGridIndexKernel;
    ComputeKernel pmeDispersionGridIndexKernel;
    ComputeKernel pmeSpreadChargeKernel;
    ComputeKernel pmeDispersionSpreadChargeKernel;
    ComputeKernel pmeFinishSpreadChargeKernel;
    ComputeKernel pmeDispersionFinishSpreadChargeKernel;
    ComputeKernel pmeConvolutionKernel;
    ComputeKernel pmeDispersionConvolutionKernel;
    ComputeKernel pmeEvalEnergyKernel;
    ComputeKernel pmeDispersionEvalEnergyKernel;
    ComputeKernel pmeInterpolateForceKernel;
    ComputeKernel pmeDispersionInterpolateForceKernel;
    std::map<std::string, std::string> pmeDefines;
    std::vector<std::pair<int, int> > exceptionAtoms;
    std::vector<std::string> paramNames;
    std::vector<double> paramValues;
    double ewaldSelfEnergy, dispersionCoefficient, alpha, dispersionAlpha;
    int gridSizeX, gridSizeY, gridSizeZ;
    int dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ;
    bool hasCoulomb, hasLJ, usePmeQueue, doLJPME, usePosqCharges, recomputeParams, hasOffsets;
    NonbondedMethod nonbondedMethod;
    static const int PmeOrder = 5;
};

/**
 * This kernel is invoked by CustomCVForce to calculate the forces acting on the system and the energy of the system.
 */
class MetalCalcCustomCVForceKernel : public CommonCalcCustomCVForceKernel {
public:
    MetalCalcCustomCVForceKernel(std::string name, const Platform& platform, ComputeContext& cc) : CommonCalcCustomCVForceKernel(name, platform, cc) {
    }
    ComputeContext& getInnerComputeContext(ContextImpl& innerContext) {
        return *reinterpret_cast<MetalPlatform::PlatformData*>(innerContext.getPlatformData())->contexts[0];
    }
};

/**
 * This kernel is invoked by ATMForce to calculate the forces acting on the system and the energy of the system.
 */
class MetalCalcATMForceKernel : public CommonCalcATMForceKernel {
public:
    MetalCalcATMForceKernel(std::string name, const Platform& platform, ComputeContext& cc) : CommonCalcATMForceKernel(name, platform, cc) {
    }
    ComputeContext& getInnerComputeContext(ContextImpl& innerContext) {
        return *reinterpret_cast<MetalPlatform::PlatformData*>(innerContext.getPlatformData())->contexts[0];
    }
};

} // namespace OpenMM

#endif /*OPENMM_OPENCLKERNELS_H_*/
