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

#import <Metal/Metal.h>
#include "MetalKernels.h"
#include "MetalForceInfo.h"
#include "openmm/Context.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include "CommonKernelSources.h"
#include "MetalBondedUtilities.h"
#include "MetalExpressionUtilities.h"
#include "MetalIntegrationUtilities.h"
#include "MetalNonbondedUtilities.h"
#include "MetalKernelSources.h"
#include "SimTKOpenMMRealType.h"
#include "SimTKOpenMMUtilities.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <iterator>
#include <set>

using namespace OpenMM;
using namespace std;

static void setPeriodicBoxSizeArg(MetalContext& cl, ComputeKernel& kernel, int index) {
    if (cl.getUseDoublePrecision())
        kernel->setArg(index, cl.getPeriodicBoxSizeDouble());
    else
        kernel->setArg(index, cl.getPeriodicBoxSize());
}

static void setPeriodicBoxArgs(MetalContext& cl, ComputeKernel& kernel, int index) {
    if (cl.getUseDoublePrecision()) {
        kernel->setArg(index++, cl.getPeriodicBoxSizeDouble());
        kernel->setArg(index++, cl.getInvPeriodicBoxSizeDouble());
        kernel->setArg(index++, cl.getPeriodicBoxVecXDouble());
        kernel->setArg(index++, cl.getPeriodicBoxVecYDouble());
        kernel->setArg(index, cl.getPeriodicBoxVecZDouble());
    }
    else {
        kernel->setArg(index++, cl.getPeriodicBoxSize());
        kernel->setArg(index++, cl.getInvPeriodicBoxSize());
        kernel->setArg(index++, cl.getPeriodicBoxVecX());
        kernel->setArg(index++, cl.getPeriodicBoxVecY());
        kernel->setArg(index, cl.getPeriodicBoxVecZ());
    }
}

void MetalCalcForcesAndEnergyKernel::initialize(const System& system) {
}

void MetalCalcForcesAndEnergyKernel::beginComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    cl.setForcesValid(true);
    cl.clearAutoclearBuffers();
    for (auto computation : cl.getPreComputations())
        computation->computeForceAndEnergy(includeForces, includeEnergy, groups);
    MetalNonbondedUtilities& nb = cl.getNonbondedUtilities();
    cl.setComputeForceCount(cl.getComputeForceCount()+1);
    nb.prepareInteractions(groups);
    map<string, double>& derivs = cl.getEnergyParamDerivWorkspace();
    for (auto& param : context.getParameters())
        derivs[param.first] = 0;
}

double MetalCalcForcesAndEnergyKernel::finishComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups, bool& valid) {
    cl.getBondedUtilities().computeInteractions(groups);
    cl.getNonbondedUtilities().computeInteractions(groups, includeForces, includeEnergy);
    double sum = 0.0;
    for (auto computation : cl.getPostComputations())
        sum += computation->computeForceAndEnergy(includeForces, includeEnergy, groups);
    cl.reduceForces();
    cl.getIntegrationUtilities().distributeForcesFromVirtualSites();
    if (includeEnergy)
        sum += cl.reduceEnergy();
    if (!cl.getForcesValid())
        valid = false;
    return sum;
}

void MetalUpdateStateDataKernel::initialize(const System& system) {
}

double MetalUpdateStateDataKernel::getTime(const ContextImpl& context) const {
    return cl.getTime();
}

void MetalUpdateStateDataKernel::setTime(ContextImpl& context, double time) {
    vector<MetalContext*>& contexts = cl.getPlatformData().contexts;
    for (auto ctx : contexts)
        ctx->setTime(time);
}

long long MetalUpdateStateDataKernel::getStepCount(const ContextImpl& context) const {
    return cl.getStepCount();
}

void MetalUpdateStateDataKernel::setStepCount(const ContextImpl& context, long long count) {
    vector<MetalContext*>& contexts = cl.getPlatformData().contexts;
    for (auto ctx : contexts)
        ctx->setStepCount(count);
}

void MetalUpdateStateDataKernel::getPositions(ContextImpl& context, vector<Vec3>& positions) {
    int numParticles = context.getSystem().getNumParticles();
    positions.resize(numParticles);
    vector<mm_float4> posCorrection;
    if (cl.getUseDoublePrecision()) {
        mm_double4* posq = (mm_double4*) cl.getPinnedBuffer();
        cl.getPosq().download(posq);
    }
    else if (cl.getUseMixedPrecision()) {
        mm_float4* posq = (mm_float4*) cl.getPinnedBuffer();
        cl.getPosq().download(posq, false);
        posCorrection.resize(numParticles);
        cl.getPosqCorrection().download(posCorrection);
    }
    else {
        mm_float4* posq = (mm_float4*) cl.getPinnedBuffer();
        cl.getPosq().download(posq);
    }
    
    // Filling in the output array is done in parallel for speed.
    
    cl.getPlatformData().threads.execute([&] (ThreadPool& threads, int threadIndex) {
        // Compute the position of each particle to return to the user.  This is done in parallel for speed.
        
        const vector<int>& order = cl.getAtomIndex();
        int numParticles = cl.getNumAtoms();
        Vec3 boxVectors[3];
        cl.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        int numThreads = threads.getNumThreads();
        int start = threadIndex*numParticles/numThreads;
        int end = (threadIndex+1)*numParticles/numThreads;
        if (cl.getUseDoublePrecision()) {
            mm_double4* posq = (mm_double4*) cl.getPinnedBuffer();
            for (int i = start; i < end; ++i) {
                mm_double4 pos = posq[i];
                mm_int4 offset = cl.getPosCellOffsets()[i];
                positions[order[i]] = Vec3(pos.x, pos.y, pos.z)-boxVectors[0]*offset.x-boxVectors[1]*offset.y-boxVectors[2]*offset.z;
            }
        }
        else if (cl.getUseMixedPrecision()) {
            mm_float4* posq = (mm_float4*) cl.getPinnedBuffer();
            for (int i = start; i < end; ++i) {
                mm_float4 pos1 = posq[i];
                mm_float4 pos2 = posCorrection[i];
                mm_int4 offset = cl.getPosCellOffsets()[i];
                positions[order[i]] = Vec3((double)pos1.x+(double)pos2.x, (double)pos1.y+(double)pos2.y, (double)pos1.z+(double)pos2.z)-boxVectors[0]*offset.x-boxVectors[1]*offset.y-boxVectors[2]*offset.z;
            }
        }
        else {
            mm_float4* posq = (mm_float4*) cl.getPinnedBuffer();
            for (int i = start; i < end; ++i) {
                mm_float4 pos = posq[i];
                mm_int4 offset = cl.getPosCellOffsets()[i];
                positions[order[i]] = Vec3(pos.x, pos.y, pos.z)-boxVectors[0]*offset.x-boxVectors[1]*offset.y-boxVectors[2]*offset.z;
            }
        }
    });
    cl.getPlatformData().threads.waitForThreads();
}

void MetalUpdateStateDataKernel::setPositions(ContextImpl& context, const vector<Vec3>& positions) {
    const vector<int>& order = cl.getAtomIndex();
    int numParticles = context.getSystem().getNumParticles();
    if (cl.getUseDoublePrecision()) {
        mm_double4* posq = (mm_double4*) cl.getPinnedBuffer();
        cl.getPosq().download(posq);
        for (int i = 0; i < numParticles; ++i) {
            mm_double4& pos = posq[i];
            const Vec3& p = positions[order[i]];
            pos.x = p[0];
            pos.y = p[1];
            pos.z = p[2];
        }
        for (int i = numParticles; i < cl.getPaddedNumAtoms(); i++)
            posq[i] = mm_double4(0.0, 0.0, 0.0, 0.0);
        cl.getPosq().upload(posq);
    }
    else {
        mm_float4* posq = (mm_float4*) cl.getPinnedBuffer();
        cl.getPosq().download(posq);
        for (int i = 0; i < numParticles; ++i) {
            mm_float4& pos = posq[i];
            const Vec3& p = positions[order[i]];
            pos.x = (float) p[0];
            pos.y = (float) p[1];
            pos.z = (float) p[2];
        }
        for (int i = numParticles; i < cl.getPaddedNumAtoms(); i++)
            posq[i] = mm_float4(0.0f, 0.0f, 0.0f, 0.0f);
        cl.getPosq().upload(posq);
    }
    if (cl.getUseMixedPrecision()) {
        mm_float4* posCorrection = (mm_float4*) cl.getPinnedBuffer();
        for (int i = 0; i < numParticles; ++i) {
            mm_float4& c = posCorrection[i];
            const Vec3& p = positions[order[i]];
            c.x = (float) (p[0]-(float)p[0]);
            c.y = (float) (p[1]-(float)p[1]);
            c.z = (float) (p[2]-(float)p[2]);
            c.w = 0;
        }
        for (int i = numParticles; i < cl.getPaddedNumAtoms(); i++)
            posCorrection[i] = mm_float4(0.0f, 0.0f, 0.0f, 0.0f);
        cl.getPosqCorrection().upload(posCorrection);
    }
    for (auto& offset : cl.getPosCellOffsets())
        offset = mm_int4(0, 0, 0, 0);
    cl.reorderAtoms();
}

void MetalUpdateStateDataKernel::getVelocities(ContextImpl& context, vector<Vec3>& velocities) {
    const vector<int>& order = cl.getAtomIndex();
    int numParticles = context.getSystem().getNumParticles();
    velocities.resize(numParticles);
    if (cl.getUseDoublePrecision() || cl.getUseMixedPrecision()) {
        mm_double4* velm = (mm_double4*) cl.getPinnedBuffer();
        cl.getVelm().download(velm);
        for (int i = 0; i < numParticles; ++i) {
            mm_double4 vel = velm[i];
            velocities[order[i]] = Vec3(vel.x, vel.y, vel.z);
        }
    }
    else {
        mm_float4* velm = (mm_float4*) cl.getPinnedBuffer();
        cl.getVelm().download(velm);
        for (int i = 0; i < numParticles; ++i) {
            mm_float4 vel = velm[i];
            velocities[order[i]] = Vec3(vel.x, vel.y, vel.z);
        }
    }
}

void MetalUpdateStateDataKernel::setVelocities(ContextImpl& context, const vector<Vec3>& velocities) {
    const vector<int>& order = cl.getAtomIndex();
    int numParticles = context.getSystem().getNumParticles();
    if (cl.getUseDoublePrecision() || cl.getUseMixedPrecision()) {
        mm_double4* velm = (mm_double4*) cl.getPinnedBuffer();
        cl.getVelm().download(velm);
        for (int i = 0; i < numParticles; ++i) {
            mm_double4& vel = velm[i];
            const Vec3& p = velocities[order[i]];
            vel.x = p[0];
            vel.y = p[1];
            vel.z = p[2];
        }
        for (int i = numParticles; i < cl.getPaddedNumAtoms(); i++)
            velm[i] = mm_double4(0.0, 0.0, 0.0, 0.0);
        cl.getVelm().upload(velm);
    }
    else {
        mm_float4* velm = (mm_float4*) cl.getPinnedBuffer();
        cl.getVelm().download(velm);
        for (int i = 0; i < numParticles; ++i) {
            mm_float4& vel = velm[i];
            const Vec3& p = velocities[order[i]];
            vel.x = p[0];
            vel.y = p[1];
            vel.z = p[2];
        }
        for (int i = numParticles; i < cl.getPaddedNumAtoms(); i++)
            velm[i] = mm_float4(0.0f, 0.0f, 0.0f, 0.0f);
        cl.getVelm().upload(velm);
    }
}

void MetalUpdateStateDataKernel::computeShiftedVelocities(ContextImpl& context, double timeShift, vector<Vec3>& velocities) {
    cl.getIntegrationUtilities().computeShiftedVelocities(timeShift, velocities);
}

void MetalUpdateStateDataKernel::getForces(ContextImpl& context, vector<Vec3>& forces) {
    const vector<int>& order = cl.getAtomIndex();
    int numParticles = context.getSystem().getNumParticles();
    forces.resize(numParticles);
    if (cl.getUseDoublePrecision()) {
        mm_double4* force = (mm_double4*) cl.getPinnedBuffer();
        cl.getForce().download(force);
        for (int i = 0; i < numParticles; ++i) {
            mm_double4 f = force[i];
            forces[order[i]] = Vec3(f.x, f.y, f.z);
        }
    }
    else {
        mm_float4* force = (mm_float4*) cl.getPinnedBuffer();
        cl.getForce().download(force);
        for (int i = 0; i < numParticles; ++i) {
            mm_float4 f = force[i];
            forces[order[i]] = Vec3(f.x, f.y, f.z);
        }
    }
}

void MetalUpdateStateDataKernel::getEnergyParameterDerivatives(ContextImpl& context, map<string, double>& derivs) {
    const vector<string>& paramDerivNames = cl.getEnergyParamDerivNames();
    int numDerivs = paramDerivNames.size();
    if (numDerivs == 0)
        return;
    derivs = cl.getEnergyParamDerivWorkspace();
    MetalArray& derivArray = cl.getEnergyParamDerivBuffer();
    if (cl.getUseDoublePrecision() || cl.getUseMixedPrecision()) {
        vector<double> derivBuffers;
        derivArray.download(derivBuffers);
        for (int i = numDerivs; i < derivArray.getSize(); i += numDerivs)
            for (int j = 0; j < numDerivs; j++)
                derivBuffers[j] += derivBuffers[i+j];
        for (int i = 0; i < numDerivs; i++)
            derivs[paramDerivNames[i]] += derivBuffers[i];
    }
    else {
        vector<float> derivBuffers;
        derivArray.download(derivBuffers);
        for (int i = numDerivs; i < derivArray.getSize(); i += numDerivs)
            for (int j = 0; j < numDerivs; j++)
                derivBuffers[j] += derivBuffers[i+j];
        for (int i = 0; i < numDerivs; i++)
            derivs[paramDerivNames[i]] += derivBuffers[i];
    }
}

void MetalUpdateStateDataKernel::getPeriodicBoxVectors(ContextImpl& context, Vec3& a, Vec3& b, Vec3& c) const {
    cl.getPeriodicBoxVectors(a, b, c);
}

void MetalUpdateStateDataKernel::setPeriodicBoxVectors(ContextImpl& context, const Vec3& a, const Vec3& b, const Vec3& c) {
    vector<MetalContext*>& contexts = cl.getPlatformData().contexts;

    // If any particles have been wrapped to the first periodic box, we need to unwrap them
    // to avoid changing their positions.

    vector<Vec3> positions;
    for (auto offset : cl.getPosCellOffsets()) {
        if (offset.x != 0 || offset.y != 0 || offset.z != 0) {
            getPositions(context, positions);
            break;
        }
    }
    
    // Update the vectors.

    for (auto ctx : contexts)
        ctx->setPeriodicBoxVectors(a, b, c);
    if (positions.size() > 0)
        setPositions(context, positions);
}

void MetalUpdateStateDataKernel::createCheckpoint(ContextImpl& context, ostream& stream) {
    int version = 3;
    stream.write((char*) &version, sizeof(int));
    int precision = (cl.getUseDoublePrecision() ? 2 : cl.getUseMixedPrecision() ? 1 : 0);
    stream.write((char*) &precision, sizeof(int));
    double time = cl.getTime();
    stream.write((char*) &time, sizeof(double));
    long long stepCount = cl.getStepCount();
    stream.write((char*) &stepCount, sizeof(long long));
    int stepsSinceReorder = cl.getStepsSinceReorder();
    stream.write((char*) &stepsSinceReorder, sizeof(int));
    char* buffer = (char*) cl.getPinnedBuffer();
    cl.getPosq().download(buffer);
    stream.write(buffer, cl.getPosq().getSize()*cl.getPosq().getElementSize());
    if (cl.getUseMixedPrecision()) {
        cl.getPosqCorrection().download(buffer);
        stream.write(buffer, cl.getPosqCorrection().getSize()*cl.getPosqCorrection().getElementSize());
    }
    cl.getVelm().download(buffer);
    stream.write(buffer, cl.getVelm().getSize()*cl.getVelm().getElementSize());
    stream.write((char*) &cl.getAtomIndex()[0], sizeof(int)*cl.getAtomIndex().size());
    stream.write((char*) &cl.getPosCellOffsets()[0], sizeof(mm_int4)*cl.getPosCellOffsets().size());
    Vec3 boxVectors[3];
    cl.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
    stream.write((char*) boxVectors, 3*sizeof(Vec3));
    cl.getIntegrationUtilities().createCheckpoint(stream);
    SimTKOpenMMUtilities::createCheckpoint(stream);
}

void MetalUpdateStateDataKernel::loadCheckpoint(ContextImpl& context, istream& stream) {
    int version;
    stream.read((char*) &version, sizeof(int));
    if (version != 3)
        throw OpenMMException("Checkpoint was created with a different version of OpenMM");
    int precision;
    stream.read((char*) &precision, sizeof(int));
    int expectedPrecision = (cl.getUseDoublePrecision() ? 2 : cl.getUseMixedPrecision() ? 1 : 0);
    if (precision != expectedPrecision)
        throw OpenMMException("Checkpoint was created with a different numeric precision");
    double time;
    stream.read((char*) &time, sizeof(double));
    long long stepCount;
    stream.read((char*) &stepCount, sizeof(long long));
    int stepsSinceReorder;
    stream.read((char*) &stepsSinceReorder, sizeof(int));
    vector<MetalContext*>& contexts = cl.getPlatformData().contexts;
    for (auto ctx : contexts) {
        ctx->setTime(time);
        ctx->setStepCount(stepCount);
        ctx->setStepsSinceReorder(stepsSinceReorder);
    }
    char* buffer = (char*) cl.getPinnedBuffer();
    stream.read(buffer, cl.getPosq().getSize()*cl.getPosq().getElementSize());
    cl.getPosq().upload(buffer);
    if (cl.getUseMixedPrecision()) {
        stream.read(buffer, cl.getPosqCorrection().getSize()*cl.getPosqCorrection().getElementSize());
        cl.getPosqCorrection().upload(buffer);
    }
    stream.read(buffer, cl.getVelm().getSize()*cl.getVelm().getElementSize());
    cl.getVelm().upload(buffer);
    stream.read((char*) &cl.getAtomIndex()[0], sizeof(int)*cl.getAtomIndex().size());
    cl.getAtomIndexArray().upload(cl.getAtomIndex());
    stream.read((char*) &cl.getPosCellOffsets()[0], sizeof(mm_int4)*cl.getPosCellOffsets().size());
    Vec3 boxVectors[3];
    stream.read((char*) &boxVectors, 3*sizeof(Vec3));
    for (auto ctx : contexts)
        ctx->setPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
    cl.getIntegrationUtilities().loadCheckpoint(stream);
    SimTKOpenMMUtilities::loadCheckpoint(stream);
    for (auto listener : cl.getReorderListeners())
        listener->execute();
    cl.validateAtomOrder();
}

class MetalCalcNonbondedForceKernel::ForceInfo : public MetalForceInfo {
public:
    ForceInfo(int requiredBuffers, const NonbondedForce& force) : MetalForceInfo(requiredBuffers), force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        double charge1, charge2, sigma1, sigma2, epsilon1, epsilon2;
        force.getParticleParameters(particle1, charge1, sigma1, epsilon1);
        force.getParticleParameters(particle2, charge2, sigma2, epsilon2);
        return (charge1 == charge2 && sigma1 == sigma2 && epsilon1 == epsilon2);
    }
    int getNumParticleGroups() {
        return force.getNumExceptions();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(index, particle1, particle2, chargeProd, sigma, epsilon);
        particles.resize(2);
        particles[0] = particle1;
        particles[1] = particle2;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int particle1, particle2;
        double chargeProd1, chargeProd2, sigma1, sigma2, epsilon1, epsilon2;
        force.getExceptionParameters(group1, particle1, particle2, chargeProd1, sigma1, epsilon1);
        force.getExceptionParameters(group2, particle1, particle2, chargeProd2, sigma2, epsilon2);
        return (chargeProd1 == chargeProd2 && sigma1 == sigma2 && epsilon1 == epsilon2);
    }
private:
    const NonbondedForce& force;
};

class MetalCalcNonbondedForceKernel::PmeIO : public CalcPmeReciprocalForceKernel::IO {
public:
    PmeIO(MetalContext& cl, ComputeKernel addForcesKernel) : cl(cl), addForcesKernel(addForcesKernel) {
        forceTemp.initialize<mm_float4>(cl, cl.getNumAtoms(), "PmeForce");
        addForcesKernel->setArg(0, forceTemp);
    }
    float* getPosq() {
        cl.getPosq().download(posq);
        return (float*) &posq[0];
    }
    void setForce(float* force) {
        forceTemp.upload(force);
        addForcesKernel->setArg(1, cl.getLongForceBuffer());
        addForcesKernel->execute(cl.getNumAtoms());
    }
private:
    MetalContext& cl;
    vector<mm_float4> posq;
    MetalArray forceTemp;
    ComputeKernel addForcesKernel;
};

class MetalCalcNonbondedForceKernel::PmePreComputation : public MetalContext::ForcePreComputation {
public:
    PmePreComputation(MetalContext& cl, Kernel& pme, CalcPmeReciprocalForceKernel::IO& io) : cl(cl), pme(pme), io(io) {
    }
    void computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        Vec3 boxVectors[3] = {Vec3(cl.getPeriodicBoxSize().x, 0, 0), Vec3(0, cl.getPeriodicBoxSize().y, 0), Vec3(0, 0, cl.getPeriodicBoxSize().z)};
        pme.getAs<CalcPmeReciprocalForceKernel>().beginComputation(io, boxVectors, includeEnergy);
    }
private:
    MetalContext& cl;
    Kernel pme;
    CalcPmeReciprocalForceKernel::IO& io;
};

class MetalCalcNonbondedForceKernel::PmePostComputation : public MetalContext::ForcePostComputation {
public:
    PmePostComputation(Kernel& pme, CalcPmeReciprocalForceKernel::IO& io) : pme(pme), io(io) {
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        return pme.getAs<CalcPmeReciprocalForceKernel>().finishComputation(io);
    }
private:
    Kernel pme;
    CalcPmeReciprocalForceKernel::IO& io;
};

class MetalCalcNonbondedForceKernel::SyncQueuePreComputation : public MetalContext::ForcePreComputation {
public:
    SyncQueuePreComputation(MetalContext& cl, int forceGroup) : cl(cl), forceGroup(forceGroup) {
    }
    void computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        // Metal executes synchronously -- no queue synchronization needed.
    }
private:
    MetalContext& cl;
    int forceGroup;
};

class MetalCalcNonbondedForceKernel::SyncQueuePostComputation : public MetalContext::ForcePostComputation {
public:
    SyncQueuePostComputation(MetalContext& cl, MetalArray& pmeEnergyBuffer, int forceGroup) : cl(cl),
            pmeEnergyBuffer(pmeEnergyBuffer), forceGroup(forceGroup) {
    }
    void setKernel(ComputeKernel kernel) {
        addEnergyKernel = kernel;
        addEnergyKernel->setArg(0, pmeEnergyBuffer);
        addEnergyKernel->setArg(1, cl.getEnergyBuffer());
        addEnergyKernel->setArg(2, (int)pmeEnergyBuffer.getSize());
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((groups&(1<<forceGroup)) != 0) {
            // Metal executes synchronously -- no event synchronization needed.
            if (includeEnergy)
                addEnergyKernel->execute(pmeEnergyBuffer.getSize());
        }
        return 0.0;
    }
private:
    MetalContext& cl;
    ComputeKernel addEnergyKernel;
    MetalArray& pmeEnergyBuffer;
    int forceGroup;
};

MetalCalcNonbondedForceKernel::~MetalCalcNonbondedForceKernel() {
    if (sort != NULL)
        delete sort;
    if (fft != NULL)
        delete fft;
    if (dispersionFft != NULL)
        delete dispersionFft;
    if (pmeio != NULL)
        delete pmeio;
}

void MetalCalcNonbondedForceKernel::initialize(const System& system, const NonbondedForce& force) {
    int forceIndex;
    for (forceIndex = 0; forceIndex < system.getNumForces() && &system.getForce(forceIndex) != &force; ++forceIndex)
        ;
    string prefix = "nonbonded"+cl.intToString(forceIndex)+"_";

    // Identify which exceptions are 1-4 interactions.

    set<int> exceptionsWithOffsets;
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge, sigma, epsilon;
        force.getExceptionParameterOffset(i, param, exception, charge, sigma, epsilon);
        exceptionsWithOffsets.insert(exception);
    }
    vector<pair<int, int> > exclusions;
    vector<int> exceptions;
    map<int, int> exceptionIndex;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        exclusions.push_back(pair<int, int>(particle1, particle2));
        if (chargeProd != 0.0 || epsilon != 0.0 || exceptionsWithOffsets.find(i) != exceptionsWithOffsets.end()) {
            exceptionIndex[i] = exceptions.size();
            exceptions.push_back(i);
        }
    }

    // Initialize nonbonded interactions.

    int numParticles = force.getNumParticles();
    vector<mm_float4> baseParticleParamVec(cl.getPaddedNumAtoms(), mm_float4(0, 0, 0, 0));
    vector<vector<int> > exclusionList(numParticles);
    hasCoulomb = false;
    hasLJ = false;
    for (int i = 0; i < numParticles; i++) {
        double charge, sigma, epsilon;
        force.getParticleParameters(i, charge, sigma, epsilon);
        baseParticleParamVec[i] = mm_float4(charge, sigma, epsilon, 0);
        exclusionList[i].push_back(i);
        if (charge != 0.0)
            hasCoulomb = true;
        if (epsilon != 0.0)
            hasLJ = true;
    }
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        string param;
        int particle;
        double charge, sigma, epsilon;
        force.getParticleParameterOffset(i, param, particle, charge, sigma, epsilon);
        if (charge != 0.0)
            hasCoulomb = true;
        if (epsilon != 0.0)
            hasLJ = true;
    }
    for (auto exclusion : exclusions) {
        exclusionList[exclusion.first].push_back(exclusion.second);
        exclusionList[exclusion.second].push_back(exclusion.first);
    }
    nonbondedMethod = CalcNonbondedForceKernel::NonbondedMethod(force.getNonbondedMethod());
    bool useCutoff = (nonbondedMethod != NoCutoff);
    bool usePeriodic = (nonbondedMethod != NoCutoff && nonbondedMethod != CutoffNonPeriodic);
    doLJPME = (nonbondedMethod == LJPME && hasLJ);
    usePosqCharges = hasCoulomb ? cl.requestPosqCharges() : false;
    map<string, string> defines;
    defines["HAS_COULOMB"] = (hasCoulomb ? "1" : "0");
    defines["HAS_LENNARD_JONES"] = (hasLJ ? "1" : "0");
    defines["USE_LJ_SWITCH"] = (useCutoff && force.getUseSwitchingFunction() ? "1" : "0");
    if (useCutoff) {
        // Compute the reaction field constants.

        double reactionFieldK = pow(force.getCutoffDistance(), -3.0)*(force.getReactionFieldDielectric()-1.0)/(2.0*force.getReactionFieldDielectric()+1.0);
        double reactionFieldC = (1.0 / force.getCutoffDistance())*(3.0*force.getReactionFieldDielectric())/(2.0*force.getReactionFieldDielectric()+1.0);
        defines["REACTION_FIELD_K"] = cl.doubleToString(reactionFieldK);
        defines["REACTION_FIELD_C"] = cl.doubleToString(reactionFieldC);
        
        // Compute the switching coefficients.
        
        if (force.getUseSwitchingFunction()) {
            defines["LJ_SWITCH_CUTOFF"] = cl.doubleToString(force.getSwitchingDistance());
            defines["LJ_SWITCH_C3"] = cl.doubleToString(10/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 3.0));
            defines["LJ_SWITCH_C4"] = cl.doubleToString(15/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 4.0));
            defines["LJ_SWITCH_C5"] = cl.doubleToString(6/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 5.0));
        }
    }
    if (force.getUseDispersionCorrection() && cl.getContextIndex() == 0 && !doLJPME)
        dispersionCoefficient = NonbondedForceImpl::calcDispersionCorrection(system, force);
    else
        dispersionCoefficient = 0.0;
    alpha = 0;
    ewaldSelfEnergy = 0.0;
    map<string, string> paramsDefines;
    paramsDefines["ONE_4PI_EPS0"] = cl.doubleToString(ONE_4PI_EPS0);
    hasOffsets = (force.getNumParticleParameterOffsets() > 0 || force.getNumExceptionParameterOffsets() > 0);
    if (hasOffsets)
        paramsDefines["HAS_OFFSETS"] = "1";
    if (force.getNumParticleParameterOffsets() > 0)
        paramsDefines["HAS_PARTICLE_OFFSETS"] = "1";
    if (force.getNumExceptionParameterOffsets() > 0)
        paramsDefines["HAS_EXCEPTION_OFFSETS"] = "1";
    if (usePosqCharges)
        paramsDefines["USE_POSQ_CHARGES"] = "1";
    if (doLJPME)
        paramsDefines["INCLUDE_LJPME_EXCEPTIONS"] = "1";
    if (nonbondedMethod == Ewald) {
        // Compute the Ewald parameters.

        int kmaxx, kmaxy, kmaxz;
        NonbondedForceImpl::calcEwaldParameters(system, force, alpha, kmaxx, kmaxy, kmaxz);
        defines["EWALD_ALPHA"] = cl.doubleToString(alpha);
        defines["TWO_OVER_SQRT_PI"] = cl.doubleToString(2.0/sqrt(M_PI));
        defines["USE_EWALD"] = "1";
        if (cl.getContextIndex() == 0) {
            paramsDefines["INCLUDE_EWALD"] = "1";
            paramsDefines["EWALD_SELF_ENERGY_SCALE"] = cl.doubleToString(ONE_4PI_EPS0*alpha/sqrt(M_PI));
            for (int i = 0; i < numParticles; i++)
                ewaldSelfEnergy -= baseParticleParamVec[i].x*baseParticleParamVec[i].x*ONE_4PI_EPS0*alpha/sqrt(M_PI);

            // Create the reciprocal space kernels.

            map<string, string> replacements;
            replacements["NUM_ATOMS"] = cl.intToString(numParticles);
            replacements["PADDED_NUM_ATOMS"] = cl.intToString(cl.getPaddedNumAtoms());
            replacements["KMAX_X"] = cl.intToString(kmaxx);
            replacements["KMAX_Y"] = cl.intToString(kmaxy);
            replacements["KMAX_Z"] = cl.intToString(kmaxz);
            replacements["EXP_COEFFICIENT"] = cl.doubleToString(-1.0/(4.0*alpha*alpha));
            replacements["ONE_4PI_EPS0"] = cl.doubleToString(ONE_4PI_EPS0);
            replacements["M_PI"] = cl.doubleToString(M_PI);
            ComputeProgram program;
            try {
                program = cl.compileProgram(MetalKernelSources::ewald, replacements);
            } catch (OpenMMException& e) {
                fprintf(stderr, "[Metal] Ewald kernel compilation failed (kmax=%d,%d,%d alpha=%g)\n",
                        kmaxx, kmaxy, kmaxz, alpha);
                throw;
            }
            ewaldSumsKernel = program->createKernel("calculateEwaldCosSinSums");
            ewaldForcesKernel = program->createKernel("calculateEwaldForces");
            int elementSize = (cl.getUseDoublePrecision() ? sizeof(mm_double2) : sizeof(mm_float2));
            cosSinSums.initialize(cl, (2*kmaxx-1)*(2*kmaxy-1)*(2*kmaxz-1), elementSize, "cosSinSums");
        }
    }
    else if (((nonbondedMethod == PME || nonbondedMethod == LJPME) && hasCoulomb) || doLJPME) {
        // Compute the PME parameters.

        NonbondedForceImpl::calcPMEParameters(system, force, alpha, gridSizeX, gridSizeY, gridSizeZ, false);
        gridSizeX = MetalFFT3D::findLegalDimension(gridSizeX);
        gridSizeY = MetalFFT3D::findLegalDimension(gridSizeY);
        gridSizeZ = MetalFFT3D::findLegalDimension(gridSizeZ);
        if (doLJPME) {
            NonbondedForceImpl::calcPMEParameters(system, force, dispersionAlpha, dispersionGridSizeX,
                                                  dispersionGridSizeY, dispersionGridSizeZ, true);
            dispersionGridSizeX = MetalFFT3D::findLegalDimension(dispersionGridSizeX);
            dispersionGridSizeY = MetalFFT3D::findLegalDimension(dispersionGridSizeY);
            dispersionGridSizeZ = MetalFFT3D::findLegalDimension(dispersionGridSizeZ);
        }
        defines["EWALD_ALPHA"] = cl.doubleToString(alpha);
        defines["TWO_OVER_SQRT_PI"] = cl.doubleToString(2.0/sqrt(M_PI));
        defines["USE_EWALD"] = "1";
        defines["DO_LJPME"] = doLJPME ? "1" : "0";
        if (doLJPME) {
            defines["EWALD_DISPERSION_ALPHA"] = cl.doubleToString(dispersionAlpha);
            double invRCut6 = pow(force.getCutoffDistance(), -6);
            double dalphaR = dispersionAlpha * force.getCutoffDistance();
            double dar2 = dalphaR*dalphaR;
            double dar4 = dar2*dar2;
            double multShift6 = -invRCut6*(1.0 - exp(-dar2) * (1.0 + dar2 + 0.5*dar4));
            defines["INVCUT6"] = cl.doubleToString(invRCut6);
            defines["MULTSHIFT6"] = cl.doubleToString(multShift6);
        }
        if (cl.getContextIndex() == 0) {
            paramsDefines["INCLUDE_EWALD"] = "1";
            paramsDefines["EWALD_SELF_ENERGY_SCALE"] = cl.doubleToString(ONE_4PI_EPS0*alpha/sqrt(M_PI));
            for (int i = 0; i < numParticles; i++)
                ewaldSelfEnergy -= baseParticleParamVec[i].x*baseParticleParamVec[i].x*ONE_4PI_EPS0*alpha/sqrt(M_PI);
            if (doLJPME) {
                paramsDefines["INCLUDE_LJPME"] = "1";
                paramsDefines["LJPME_SELF_ENERGY_SCALE"] = cl.doubleToString(pow(dispersionAlpha, 6)/3.0);
                for (int i = 0; i < numParticles; i++)
                    ewaldSelfEnergy += baseParticleParamVec[i].z*pow(baseParticleParamVec[i].y*dispersionAlpha, 6)/3.0;
            }
            pmeDefines["PME_ORDER"] = cl.intToString(PmeOrder);
            pmeDefines["NUM_ATOMS"] = cl.intToString(numParticles);
            pmeDefines["PADDED_NUM_ATOMS"] = cl.intToString(cl.getPaddedNumAtoms());
            pmeDefines["RECIP_EXP_FACTOR"] = cl.doubleToString(M_PI*M_PI/(alpha*alpha));
            pmeDefines["GRID_SIZE_X"] = cl.intToString(gridSizeX);
            pmeDefines["GRID_SIZE_Y"] = cl.intToString(gridSizeY);
            pmeDefines["GRID_SIZE_Z"] = cl.intToString(gridSizeZ);
            pmeDefines["EPSILON_FACTOR"] = cl.doubleToString(sqrt(ONE_4PI_EPS0));
            pmeDefines["M_PI"] = cl.doubleToString(M_PI);
            pmeDefines["USE_FIXED_POINT_CHARGE_SPREADING"] = "1";
            bool deviceIsCpu = false;
            if (deviceIsCpu)
                pmeDefines["DEVICE_IS_CPU"] = "1";
            if (cl.getPlatformData().useCpuPme && !doLJPME && usePosqCharges) {
                // Create the CPU PME kernel.

                try {
                    cpuPme = getPlatform().createKernel(CalcPmeReciprocalForceKernel::Name(), *cl.getPlatformData().context);
                    cpuPme.getAs<CalcPmeReciprocalForceKernel>().initialize(gridSizeX, gridSizeY, gridSizeZ, numParticles, alpha, false);
                    ComputeProgram program = cl.compileProgram(MetalKernelSources::pme, pmeDefines);
                    ComputeKernel addForcesKernel = program->createKernel("addForces");
                    pmeio = new PmeIO(cl, addForcesKernel);
                    cl.addPreComputation(new PmePreComputation(cl, cpuPme, *pmeio));
                    cl.addPostComputation(new PmePostComputation(cpuPme, *pmeio));
                }
                catch (OpenMMException& ex) {
                    // The CPU PME plugin isn't available — fall through to GPU PME.
                    fprintf(stderr, "[Metal] CPU PME not available (%s), using GPU PME path\n", ex.what());
                }
            }
            if (pmeio == NULL) {
                // Create required data structures.

                int elementSize = (cl.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
                int roundedZSize = PmeOrder*(int) ceil(gridSizeZ/(double) PmeOrder);
                int gridElements = gridSizeX*gridSizeY*roundedZSize;
                if (doLJPME) {
                    roundedZSize = PmeOrder*(int) ceil(dispersionGridSizeZ/(double) PmeOrder);
                    gridElements = max(gridElements, dispersionGridSizeX*dispersionGridSizeY*roundedZSize);
                }
                pmeGrid1.initialize(cl, gridElements, 2*elementSize, "pmeGrid1");
                pmeGrid2.initialize(cl, gridElements, 2*elementSize, "pmeGrid2");
                cl.addAutoclearBuffer(pmeGrid2);
                pmeBsplineModuliX.initialize(cl, gridSizeX, elementSize, "pmeBsplineModuliX");
                pmeBsplineModuliY.initialize(cl, gridSizeY, elementSize, "pmeBsplineModuliY");
                pmeBsplineModuliZ.initialize(cl, gridSizeZ, elementSize, "pmeBsplineModuliZ");
                if (doLJPME) {
                    pmeDispersionBsplineModuliX.initialize(cl, dispersionGridSizeX, elementSize, "pmeDispersionBsplineModuliX");
                    pmeDispersionBsplineModuliY.initialize(cl, dispersionGridSizeY, elementSize, "pmeDispersionBsplineModuliY");
                    pmeDispersionBsplineModuliZ.initialize(cl, dispersionGridSizeZ, elementSize, "pmeDispersionBsplineModuliZ");
                }
                pmeBsplineTheta.initialize(cl, PmeOrder*numParticles, 4*elementSize, "pmeBsplineTheta");
                pmeAtomRange.initialize<int>(cl, gridSizeX*gridSizeY*gridSizeZ+1, "pmeAtomRange");
                pmeAtomGridIndex.initialize<mm_int2>(cl, numParticles, "pmeAtomGridIndex");
                int energyElementSize = (cl.getUseDoublePrecision() || cl.getUseMixedPrecision() ? sizeof(double) : sizeof(float));
                pmeEnergyBuffer.initialize(cl, cl.getNumThreadBlocks()*MetalContext::ThreadBlockSize, energyElementSize, "pmeEnergyBuffer");
                cl.clearBuffer(pmeEnergyBuffer);
                sort = new MetalSort(cl, new SortTrait(), cl.getNumAtoms());
                fft = new MetalFFT3D(cl, gridSizeX, gridSizeY, gridSizeZ, true);
                if (doLJPME)
                    dispersionFft = new MetalFFT3D(cl, dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, true);
                // Metal doesn't support separate command queues for PME stream.
                // All GPU work is serialized through the single Metal command queue.
                usePmeQueue = false;

                // Initialize the b-spline moduli.

                for (int grid = 0; grid < 2; grid++) {
                    int xsize, ysize, zsize;
                    MetalArray *xmoduli, *ymoduli, *zmoduli;
                    if (grid == 0) {
                        xsize = gridSizeX;
                        ysize = gridSizeY;
                        zsize = gridSizeZ;
                        xmoduli = &pmeBsplineModuliX;
                        ymoduli = &pmeBsplineModuliY;
                        zmoduli = &pmeBsplineModuliZ;
                    }
                    else {
                        if (!doLJPME)
                            continue;
                        xsize = dispersionGridSizeX;
                        ysize = dispersionGridSizeY;
                        zsize = dispersionGridSizeZ;
                        xmoduli = &pmeDispersionBsplineModuliX;
                        ymoduli = &pmeDispersionBsplineModuliY;
                        zmoduli = &pmeDispersionBsplineModuliZ;
                    }
                    int maxSize = max(max(xsize, ysize), zsize);
                    vector<double> data(PmeOrder);
                    vector<double> ddata(PmeOrder);
                    vector<double> bsplines_data(maxSize);
                    data[PmeOrder-1] = 0.0;
                    data[1] = 0.0;
                    data[0] = 1.0;
                    for (int i = 3; i < PmeOrder; i++) {
                        double div = 1.0/(i-1.0);
                        data[i-1] = 0.0;
                        for (int j = 1; j < (i-1); j++)
                            data[i-j-1] = div*(j*data[i-j-2]+(i-j)*data[i-j-1]);
                        data[0] = div*data[0];
                    }

                    // Differentiate.

                    ddata[0] = -data[0];
                    for (int i = 1; i < PmeOrder; i++)
                        ddata[i] = data[i-1]-data[i];
                    double div = 1.0/(PmeOrder-1);
                    data[PmeOrder-1] = 0.0;
                    for (int i = 1; i < (PmeOrder-1); i++)
                        data[PmeOrder-i-1] = div*(i*data[PmeOrder-i-2]+(PmeOrder-i)*data[PmeOrder-i-1]);
                    data[0] = div*data[0];
                    for (int i = 0; i < maxSize; i++)
                        bsplines_data[i] = 0.0;
                    for (int i = 1; i <= PmeOrder; i++)
                        bsplines_data[i] = data[i-1];

                    // Evaluate the actual bspline moduli for X/Y/Z.

                    for (int dim = 0; dim < 3; dim++) {
                        int ndata = (dim == 0 ? xsize : dim == 1 ? ysize : zsize);
                        vector<double> moduli(ndata);
                        for (int i = 0; i < ndata; i++) {
                            double sc = 0.0;
                            double ss = 0.0;
                            for (int j = 0; j < ndata; j++) {
                                double arg = (2.0*M_PI*i*j)/ndata;
                                sc += bsplines_data[j]*cos(arg);
                                ss += bsplines_data[j]*sin(arg);
                            }
                            moduli[i] = sc*sc+ss*ss;
                        }
                        for (int i = 0; i < ndata; i++)
                        {
                            if (moduli[i] < 1.0e-7)
                                moduli[i] = (moduli[(i-1+ndata)%ndata]+moduli[(i+1)%ndata])*0.5;
                        }
                        if (dim == 0)
                            xmoduli->upload(moduli, true);
                        else if (dim == 1)
                            ymoduli->upload(moduli, true);
                        else
                            zmoduli->upload(moduli, true);
                    }
                }
            }
        }
    }

    // Add code to subtract off the reciprocal part of excluded interactions.

    if ((nonbondedMethod == Ewald || nonbondedMethod == PME || nonbondedMethod == LJPME) && pmeio == NULL) {
        int numContexts = cl.getPlatformData().contexts.size();
        int startIndex = cl.getContextIndex()*force.getNumExceptions()/numContexts;
        int endIndex = (cl.getContextIndex()+1)*force.getNumExceptions()/numContexts;
        int numExclusions = endIndex-startIndex;
        if (numExclusions > 0) {
            paramsDefines["HAS_EXCLUSIONS"] = "1";
            vector<vector<int> > atoms(numExclusions, vector<int>(2));
            exclusionAtoms.initialize<mm_int2>(cl, numExclusions, "exclusionAtoms");
            exclusionParams.initialize<mm_float4>(cl, numExclusions, "exclusionParams");
            vector<mm_int2> exclusionAtomsVec(numExclusions);
            for (int i = 0; i < numExclusions; i++) {
                int j = i+startIndex;
                exclusionAtomsVec[i] = mm_int2(exclusions[j].first, exclusions[j].second);
                atoms[i][0] = exclusions[j].first;
                atoms[i][1] = exclusions[j].second;
            }
            exclusionAtoms.upload(exclusionAtomsVec);
            map<string, string> replacements;
            replacements["PARAMS"] = cl.getBondedUtilities().addArgument(exclusionParams, "float4");
            replacements["EWALD_ALPHA"] = cl.doubleToString(alpha);
            replacements["TWO_OVER_SQRT_PI"] = cl.doubleToString(2.0/sqrt(M_PI));
            replacements["DO_LJPME"] = doLJPME ? "1" : "0";
            replacements["USE_PERIODIC"] = force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0";
            if (doLJPME)
                replacements["EWALD_DISPERSION_ALPHA"] = cl.doubleToString(dispersionAlpha);
            if (force.getIncludeDirectSpace())
                cl.getBondedUtilities().addInteraction(atoms, cl.replaceStrings(MetalKernelSources::pmeExclusions, replacements), force.getForceGroup());
        }
    }

    // Add the interaction to the default nonbonded kernel.
    
    string source = cl.replaceStrings(MetalKernelSources::coulombLennardJones, defines);
    charges.initialize(cl, cl.getPaddedNumAtoms(), cl.getUseDoublePrecision() ? sizeof(double) : sizeof(float), "charges");
    baseParticleParams.initialize<mm_float4>(cl, cl.getPaddedNumAtoms(), "baseParticleParams");
    baseParticleParams.upload(baseParticleParamVec);
    map<string, string> replacements;
    replacements["ONE_4PI_EPS0"] = cl.doubleToString(ONE_4PI_EPS0);
    if (usePosqCharges) {
        replacements["CHARGE1"] = "posq1.w";
        replacements["CHARGE2"] = "posq2.w";
    }
    else {
        replacements["CHARGE1"] = prefix+"charge1";
        replacements["CHARGE2"] = prefix+"charge2";
    }
    if (hasCoulomb && !usePosqCharges)
        cl.getNonbondedUtilities().addParameter(ComputeParameterInfo(charges, prefix+"charge", "real", 1));
    sigmaEpsilon.initialize<mm_float2>(cl, cl.getPaddedNumAtoms(), "sigmaEpsilon");
    if (hasLJ) {
        replacements["SIGMA_EPSILON1"] = prefix+"sigmaEpsilon1";
        replacements["SIGMA_EPSILON2"] = prefix+"sigmaEpsilon2";
        cl.getNonbondedUtilities().addParameter(ComputeParameterInfo(sigmaEpsilon, prefix+"sigmaEpsilon", "float", 2));
    }
    source = cl.replaceStrings(source, replacements);
    if (force.getIncludeDirectSpace())
        cl.getNonbondedUtilities().addInteraction(useCutoff, usePeriodic, true, force.getCutoffDistance(), exclusionList, source, force.getForceGroup(), numParticles > 3000);

    // Initialize the exceptions.

    int numContexts = cl.getPlatformData().contexts.size();
    int startIndex = cl.getContextIndex()*exceptions.size()/numContexts;
    int endIndex = (cl.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    if (numExceptions > 0) {
        paramsDefines["HAS_EXCEPTIONS"] = "1";
        exceptionAtoms.resize(numExceptions);
        vector<vector<int> > atoms(numExceptions, vector<int>(2));
        exceptionParams.initialize<mm_float4>(cl, numExceptions, "exceptionParams");
        baseExceptionParams.initialize<mm_float4>(cl, numExceptions, "baseExceptionParams");
        vector<mm_float4> baseExceptionParamsVec(numExceptions);
        for (int i = 0; i < numExceptions; i++) {
            double chargeProd, sigma, epsilon;
            force.getExceptionParameters(exceptions[startIndex+i], atoms[i][0], atoms[i][1], chargeProd, sigma, epsilon);
            baseExceptionParamsVec[i] = mm_float4(chargeProd, sigma, epsilon, 0);
            exceptionAtoms[i] = make_pair(atoms[i][0], atoms[i][1]);
        }
        baseExceptionParams.upload(baseExceptionParamsVec);
        map<string, string> replacements;
        replacements["APPLY_PERIODIC"] = (usePeriodic && force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0");
        replacements["PARAMS"] = cl.getBondedUtilities().addArgument(exceptionParams, "float4");
        if (force.getIncludeDirectSpace())
            cl.getBondedUtilities().addInteraction(atoms, cl.replaceStrings(MetalKernelSources::nonbondedExceptions, replacements), force.getForceGroup());
    }
    
    // Initialize parameter offsets.

    vector<vector<mm_float4> > particleOffsetVec(force.getNumParticles());
    vector<vector<mm_float4> > exceptionOffsetVec(numExceptions);
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        string param;
        int particle;
        double charge, sigma, epsilon;
        force.getParticleParameterOffset(i, param, particle, charge, sigma, epsilon);
        auto paramPos = find(paramNames.begin(), paramNames.end(), param);
        int paramIndex;
        if (paramPos == paramNames.end()) {
            paramIndex = paramNames.size();
            paramNames.push_back(param);
        }
        else
            paramIndex = paramPos-paramNames.begin();
        particleOffsetVec[particle].push_back(mm_float4(charge, sigma, epsilon, paramIndex));
    }
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge, sigma, epsilon;
        force.getExceptionParameterOffset(i, param, exception, charge, sigma, epsilon);
        int index = exceptionIndex[exception];
        if (index < startIndex || index >= endIndex)
            continue;
        auto paramPos = find(paramNames.begin(), paramNames.end(), param);
        int paramIndex;
        if (paramPos == paramNames.end()) {
            paramIndex = paramNames.size();
            paramNames.push_back(param);
        }
        else
            paramIndex = paramPos-paramNames.begin();
        exceptionOffsetVec[index-startIndex].push_back(mm_float4(charge, sigma, epsilon, paramIndex));
    }
    paramValues.resize(paramNames.size(), 0.0);
    particleParamOffsets.initialize<mm_float4>(cl, max(force.getNumParticleParameterOffsets(), 1), "particleParamOffsets");
    particleOffsetIndices.initialize<int>(cl, cl.getPaddedNumAtoms()+1, "particleOffsetIndices");
    vector<int> particleOffsetIndicesVec, exceptionOffsetIndicesVec;
    vector<mm_float4> p, e;
    for (int i = 0; i < particleOffsetVec.size(); i++) {
        particleOffsetIndicesVec.push_back(p.size());
        for (int j = 0; j < particleOffsetVec[i].size(); j++)
            p.push_back(particleOffsetVec[i][j]);
    }
    while (particleOffsetIndicesVec.size() < particleOffsetIndices.getSize())
        particleOffsetIndicesVec.push_back(p.size());
    for (int i = 0; i < exceptionOffsetVec.size(); i++) {
        exceptionOffsetIndicesVec.push_back(e.size());
        for (int j = 0; j < exceptionOffsetVec[i].size(); j++)
            e.push_back(exceptionOffsetVec[i][j]);
    }
    exceptionOffsetIndicesVec.push_back(e.size());
    if (force.getNumParticleParameterOffsets() > 0) {
        particleParamOffsets.upload(p);
        particleOffsetIndices.upload(particleOffsetIndicesVec);
    }
    exceptionParamOffsets.initialize<mm_float4>(cl, max((int) e.size(), 1), "exceptionParamOffsets");
    exceptionOffsetIndices.initialize<int>(cl, exceptionOffsetIndicesVec.size(), "exceptionOffsetIndices");
    if (e.size() > 0) {
        exceptionParamOffsets.upload(e);
        exceptionOffsetIndices.upload(exceptionOffsetIndicesVec);
    }
    globalParams.initialize(cl, max((int) paramValues.size(), 1), cl.getUseDoublePrecision() ? sizeof(double) : sizeof(float), "globalParams");
    if (paramValues.size() > 0)
        globalParams.upload(paramValues, true);
    recomputeParams = true;
    
    // Initialize the kernel for updating parameters.
    
    ComputeProgram program;
    try {
        program = cl.compileProgram(MetalKernelSources::nonbondedParameters, paramsDefines);
    } catch (OpenMMException& e) {
        fprintf(stderr, "[Metal] Nonbonded parameter kernel compilation failed (%d defines)\n",
                (int)paramsDefines.size());
        throw;
    }
    computeParamsKernel = program->createKernel("computeParameters");
    computeExclusionParamsKernel = program->createKernel("computeExclusionParameters");
    info = new ForceInfo(0, force);
    cl.addForce(info);
}

double MetalCalcNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    bool deviceIsCpu = false;
    if (!hasInitializedKernel) {
        hasInitializedKernel = true;
        int index = 0;
        computeParamsKernel->setArg(index++, cl.getEnergyBuffer());
        index++;
        computeParamsKernel->setArg(index++, globalParams);
        computeParamsKernel->setArg(index++, cl.getPaddedNumAtoms());
        computeParamsKernel->setArg(index++, baseParticleParams);
        computeParamsKernel->setArg(index++, cl.getPosq());
        computeParamsKernel->setArg(index++, charges);
        computeParamsKernel->setArg(index++, sigmaEpsilon);
        computeParamsKernel->setArg(index++, particleParamOffsets);
        computeParamsKernel->setArg(index++, particleOffsetIndices);
        if (exceptionParams.isInitialized()) {
            computeParamsKernel->setArg(index++, exceptionParams.getSize());
            computeParamsKernel->setArg(index++, baseExceptionParams);
            computeParamsKernel->setArg(index++, exceptionParams);
            computeParamsKernel->setArg(index++, exceptionParamOffsets);
            computeParamsKernel->setArg(index++, exceptionOffsetIndices);
        }
        if (exclusionParams.isInitialized()) {
            computeExclusionParamsKernel->setArg(0, cl.getPosq());
            computeExclusionParamsKernel->setArg(1, charges);
            computeExclusionParamsKernel->setArg(2, sigmaEpsilon);
            computeExclusionParamsKernel->setArg(3, exclusionParams.getSize());
            computeExclusionParamsKernel->setArg(4, exclusionAtoms);
            computeExclusionParamsKernel->setArg(5, exclusionParams);
        }
        if (cosSinSums.isInitialized()) {
            ewaldSumsKernel->setArg(0, cl.getEnergyBuffer());
            ewaldSumsKernel->setArg(1, cl.getPosq());
            ewaldSumsKernel->setArg(2, cosSinSums);
            ewaldForcesKernel->setArg(0, cl.getLongForceBuffer());
            ewaldForcesKernel->setArg(1, cl.getPosq());
            ewaldForcesKernel->setArg(2, cosSinSums);
        }
        if (pmeGrid1.isInitialized()) {
            // Create kernels for Coulomb PME.
            
            map<string, string> replacements;
            replacements["CHARGE"] = (usePosqCharges ? "pos.w" : "charges[atom]");
            ComputeProgram program;
            try {
                program = cl.compileProgram(cl.replaceStrings(MetalKernelSources::pme, replacements), pmeDefines);
            } catch (OpenMMException& e) {
                fprintf(stderr, "[Metal] Coulomb PME kernel compilation failed (grid=%dx%dx%d, alpha=%g)\n",
                        gridSizeX, gridSizeY, gridSizeZ, alpha);
                throw;
            }
            pmeGridIndexKernel = program->createKernel("findAtomGridIndex");
            pmeSpreadChargeKernel = program->createKernel("gridSpreadCharge");
            pmeConvolutionKernel = program->createKernel("reciprocalConvolution");
            pmeEvalEnergyKernel = program->createKernel("gridEvaluateEnergy");
            pmeInterpolateForceKernel = program->createKernel("gridInterpolateForce");
            int elementSize = (cl.getUseDoublePrecision() ? sizeof(mm_double4) : sizeof(mm_float4));
            pmeGridIndexKernel->setArg(0, cl.getPosq());
            pmeGridIndexKernel->setArg(1, pmeAtomGridIndex);
            pmeSpreadChargeKernel->setArg(0, cl.getPosq());
            pmeSpreadChargeKernel->setArg(1, pmeGrid2);
            pmeSpreadChargeKernel->setArg(10, pmeAtomGridIndex);
            pmeSpreadChargeKernel->setArg(11, charges);
            pmeConvolutionKernel->setArg(0, pmeGrid2);
            pmeConvolutionKernel->setArg(1, pmeBsplineModuliX);
            pmeConvolutionKernel->setArg(2, pmeBsplineModuliY);
            pmeConvolutionKernel->setArg(3, pmeBsplineModuliZ);
            pmeEvalEnergyKernel->setArg(0, pmeGrid2);
            pmeEvalEnergyKernel->setArg(1, usePmeQueue ? pmeEnergyBuffer : cl.getEnergyBuffer());
            pmeEvalEnergyKernel->setArg(2, pmeBsplineModuliX);
            pmeEvalEnergyKernel->setArg(3, pmeBsplineModuliY);
            pmeEvalEnergyKernel->setArg(4, pmeBsplineModuliZ);
            pmeInterpolateForceKernel->setArg(0, cl.getPosq());
            pmeInterpolateForceKernel->setArg(1, cl.getLongForceBuffer());
            pmeInterpolateForceKernel->setArg(2, pmeGrid1);
            pmeInterpolateForceKernel->setArg(11, pmeAtomGridIndex);
            pmeInterpolateForceKernel->setArg(12, charges);
            pmeFinishSpreadChargeKernel = program->createKernel("finishSpreadCharge");
            pmeFinishSpreadChargeKernel->setArg(0, pmeGrid2);
            pmeFinishSpreadChargeKernel->setArg(1, pmeGrid1);
            if (usePmeQueue)
                if (syncQueue)
                    syncQueue->setKernel(program->createKernel("addEnergy"));

            if (doLJPME) {
                // Create kernels for LJ PME.

                pmeDefines["EWALD_ALPHA"] = cl.doubleToString(dispersionAlpha);
                pmeDefines["GRID_SIZE_X"] = cl.intToString(dispersionGridSizeX);
                pmeDefines["GRID_SIZE_Y"] = cl.intToString(dispersionGridSizeY);
                pmeDefines["GRID_SIZE_Z"] = cl.intToString(dispersionGridSizeZ);
                pmeDefines["EPSILON_FACTOR"] = "1";
                pmeDefines["RECIP_EXP_FACTOR"] = cl.doubleToString(M_PI*M_PI/(dispersionAlpha*dispersionAlpha));
                pmeDefines["USE_LJPME"] = "1";
                pmeDefines["CHARGE_FROM_SIGEPS"] = "1";
                try {
                    program = cl.compileProgram(MetalKernelSources::pme, pmeDefines);
                } catch (OpenMMException& e) {
                    fprintf(stderr, "[Metal] LJPME dispersion kernel compilation failed (grid=%dx%dx%d, dispersionAlpha=%g)\n",
                            dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, dispersionAlpha);
                    throw;
                }
                pmeDispersionGridIndexKernel = program->createKernel("findAtomGridIndex");
                pmeDispersionSpreadChargeKernel = program->createKernel("gridSpreadCharge");
                pmeDispersionConvolutionKernel = program->createKernel("reciprocalConvolution");
                pmeDispersionEvalEnergyKernel = program->createKernel("gridEvaluateEnergy");
                pmeDispersionInterpolateForceKernel = program->createKernel("gridInterpolateForce");
                pmeDispersionGridIndexKernel->setArg(0, cl.getPosq());
                pmeDispersionGridIndexKernel->setArg(1, pmeAtomGridIndex);
                pmeDispersionSpreadChargeKernel->setArg(0, cl.getPosq());
                pmeDispersionSpreadChargeKernel->setArg(1, pmeGrid2);
                pmeDispersionSpreadChargeKernel->setArg(10, pmeAtomGridIndex);
                pmeDispersionSpreadChargeKernel->setArg(11, sigmaEpsilon);
                pmeDispersionConvolutionKernel->setArg(0, pmeGrid2);
                pmeDispersionConvolutionKernel->setArg(1, pmeDispersionBsplineModuliX);
                pmeDispersionConvolutionKernel->setArg(2, pmeDispersionBsplineModuliY);
                pmeDispersionConvolutionKernel->setArg(3, pmeDispersionBsplineModuliZ);
                pmeDispersionEvalEnergyKernel->setArg(0, pmeGrid2);
                pmeDispersionEvalEnergyKernel->setArg(1, usePmeQueue ? pmeEnergyBuffer : cl.getEnergyBuffer());
                pmeDispersionEvalEnergyKernel->setArg(2, pmeDispersionBsplineModuliX);
                pmeDispersionEvalEnergyKernel->setArg(3, pmeDispersionBsplineModuliY);
                pmeDispersionEvalEnergyKernel->setArg(4, pmeDispersionBsplineModuliZ);
                pmeDispersionInterpolateForceKernel->setArg(0, cl.getPosq());
                pmeDispersionInterpolateForceKernel->setArg(1, cl.getLongForceBuffer());
                pmeDispersionInterpolateForceKernel->setArg(2, pmeGrid1);
                pmeDispersionInterpolateForceKernel->setArg(11, pmeAtomGridIndex);
                pmeDispersionInterpolateForceKernel->setArg(12, sigmaEpsilon);
                pmeDispersionFinishSpreadChargeKernel = program->createKernel("finishSpreadCharge");
                pmeDispersionFinishSpreadChargeKernel->setArg(0, pmeGrid2);
                pmeDispersionFinishSpreadChargeKernel->setArg(1, pmeGrid1);
            }
       }
    }
    
    // Update particle and exception parameters.

    bool paramChanged = false;
    for (int i = 0; i < paramNames.size(); i++) {
        double value = context.getParameter(paramNames[i]);
        if (value != paramValues[i]) {
            paramValues[i] = value;;
            paramChanged = true;
        }
    }
    if (paramChanged) {
        recomputeParams = true;
        globalParams.upload(paramValues, true);
    }
    double energy = (includeReciprocal ? ewaldSelfEnergy : 0.0);
    if (recomputeParams || hasOffsets) {
        computeParamsKernel->setArg(1, includeEnergy && includeReciprocal);
        computeParamsKernel->execute(cl.getPaddedNumAtoms());
        if (exclusionParams.isInitialized())
            computeExclusionParamsKernel->execute(exclusionParams.getSize());
        // Metal: no PME queue synchronization needed (all work is synchronous).
        if (hasOffsets)
            energy = 0.0; // The Ewald self energy was computed in the kernel.
        recomputeParams = false;
    }
    
    // Do reciprocal space calculations.
    
    if (cosSinSums.isInitialized() && includeReciprocal) {
        mm_double4 boxSize = cl.getPeriodicBoxSizeDouble();
        if (cl.getUseDoublePrecision()) {
            ewaldSumsKernel->setArg(3, boxSize);
            ewaldForcesKernel->setArg(3, boxSize);
        }
        else {
            ewaldSumsKernel->setArg(3, mm_float4((float) boxSize.x, (float) boxSize.y, (float) boxSize.z, 0));
            ewaldForcesKernel->setArg(3, mm_float4((float) boxSize.x, (float) boxSize.y, (float) boxSize.z, 0));
        }
        ewaldSumsKernel->execute(cosSinSums.getSize());
        ewaldForcesKernel->execute(cl.getNumAtoms());
    }
    if (pmeGrid1.isInitialized() && includeReciprocal) {
        // Metal: single command queue, no separate PME stream needed.
        // Invert the periodic box vectors.
        
        Vec3 boxVectors[3];
        cl.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double determinant = boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2];
        double scale = 1.0/determinant;
        mm_double4 recipBoxVectors[3];
        recipBoxVectors[0] = mm_double4(boxVectors[1][1]*boxVectors[2][2]*scale, 0, 0, 0);
        recipBoxVectors[1] = mm_double4(-boxVectors[1][0]*boxVectors[2][2]*scale, boxVectors[0][0]*boxVectors[2][2]*scale, 0, 0);
        recipBoxVectors[2] = mm_double4((boxVectors[1][0]*boxVectors[2][1]-boxVectors[1][1]*boxVectors[2][0])*scale, -boxVectors[0][0]*boxVectors[2][1]*scale, boxVectors[0][0]*boxVectors[1][1]*scale, 0);
        mm_float4 recipBoxVectorsFloat[3];
        for (int i = 0; i < 3; i++)
            recipBoxVectorsFloat[i] = mm_float4((float) recipBoxVectors[i].x, (float) recipBoxVectors[i].y, (float) recipBoxVectors[i].z, 0);
        
        // Execute the reciprocal space kernels.

        if (hasCoulomb) {
            setPeriodicBoxArgs(cl, pmeGridIndexKernel, 2);
            if (cl.getUseDoublePrecision()) {
                pmeGridIndexKernel->setArg(7, recipBoxVectors[0]);
                pmeGridIndexKernel->setArg(8, recipBoxVectors[1]);
                pmeGridIndexKernel->setArg(9, recipBoxVectors[2]);
            }
            else {
                pmeGridIndexKernel->setArg(7, recipBoxVectorsFloat[0]);
                pmeGridIndexKernel->setArg(8, recipBoxVectorsFloat[1]);
                pmeGridIndexKernel->setArg(9, recipBoxVectorsFloat[2]);
            }
            pmeGridIndexKernel->execute(cl.getNumAtoms());
            sort->sort(pmeAtomGridIndex);
            setPeriodicBoxArgs(cl, pmeSpreadChargeKernel, 2);
            if (cl.getUseDoublePrecision()) {
                pmeSpreadChargeKernel->setArg(7, recipBoxVectors[0]);
                pmeSpreadChargeKernel->setArg(8, recipBoxVectors[1]);
                pmeSpreadChargeKernel->setArg(9, recipBoxVectors[2]);
            }
            else {
                pmeSpreadChargeKernel->setArg(7, recipBoxVectorsFloat[0]);
                pmeSpreadChargeKernel->setArg(8, recipBoxVectorsFloat[1]);
                pmeSpreadChargeKernel->setArg(9, recipBoxVectorsFloat[2]);
            }
            pmeSpreadChargeKernel->execute(cl.getNumAtoms());
            pmeFinishSpreadChargeKernel->execute(gridSizeX*gridSizeY*gridSizeZ);
            fft->execFFT(pmeGrid1, pmeGrid2, true);
            mm_double4 boxSize = cl.getPeriodicBoxSizeDouble();
            if (cl.getUseDoublePrecision()) {
                pmeConvolutionKernel->setArg(4, recipBoxVectors[0]);
                pmeConvolutionKernel->setArg(5, recipBoxVectors[1]);
                pmeConvolutionKernel->setArg(6, recipBoxVectors[2]);
                pmeEvalEnergyKernel->setArg(5, recipBoxVectors[0]);
                pmeEvalEnergyKernel->setArg(6, recipBoxVectors[1]);
                pmeEvalEnergyKernel->setArg(7, recipBoxVectors[2]);
            }
            else {
                pmeConvolutionKernel->setArg(4, recipBoxVectorsFloat[0]);
                pmeConvolutionKernel->setArg(5, recipBoxVectorsFloat[1]);
                pmeConvolutionKernel->setArg(6, recipBoxVectorsFloat[2]);
                pmeEvalEnergyKernel->setArg(5, recipBoxVectorsFloat[0]);
                pmeEvalEnergyKernel->setArg(6, recipBoxVectorsFloat[1]);
                pmeEvalEnergyKernel->setArg(7, recipBoxVectorsFloat[2]);
            }
            if (includeEnergy)
                pmeEvalEnergyKernel->execute(gridSizeX*gridSizeY*gridSizeZ);
            pmeConvolutionKernel->execute(gridSizeX*gridSizeY*gridSizeZ);
            fft->execFFT(pmeGrid2, pmeGrid1, false);
            setPeriodicBoxArgs(cl, pmeInterpolateForceKernel, 3);
            if (cl.getUseDoublePrecision()) {
                pmeInterpolateForceKernel->setArg(8, recipBoxVectors[0]);
                pmeInterpolateForceKernel->setArg(9, recipBoxVectors[1]);
                pmeInterpolateForceKernel->setArg(10, recipBoxVectors[2]);
            }
            else {
                pmeInterpolateForceKernel->setArg(8, recipBoxVectorsFloat[0]);
                pmeInterpolateForceKernel->setArg(9, recipBoxVectorsFloat[1]);
                pmeInterpolateForceKernel->setArg(10, recipBoxVectorsFloat[2]);
            }
            if (deviceIsCpu)
                pmeInterpolateForceKernel->execute(2*10, 1);
            else
                pmeInterpolateForceKernel->execute(cl.getNumAtoms());
        }
        
        if (doLJPME && hasLJ) {
            setPeriodicBoxArgs(cl, pmeDispersionGridIndexKernel, 2);
            if (cl.getUseDoublePrecision()) {
                pmeDispersionGridIndexKernel->setArg(7, recipBoxVectors[0]);
                pmeDispersionGridIndexKernel->setArg(8, recipBoxVectors[1]);
                pmeDispersionGridIndexKernel->setArg(9, recipBoxVectors[2]);
            }
            else {
                pmeDispersionGridIndexKernel->setArg(7, recipBoxVectorsFloat[0]);
                pmeDispersionGridIndexKernel->setArg(8, recipBoxVectorsFloat[1]);
                pmeDispersionGridIndexKernel->setArg(9, recipBoxVectorsFloat[2]);
            }
            pmeDispersionGridIndexKernel->execute(cl.getNumAtoms());
            if (!hasCoulomb)
                sort->sort(pmeAtomGridIndex);
            cl.clearBuffer(pmeGrid2);
            setPeriodicBoxArgs(cl, pmeDispersionSpreadChargeKernel, 2);
            if (cl.getUseDoublePrecision()) {
                pmeDispersionSpreadChargeKernel->setArg(7, recipBoxVectors[0]);
                pmeDispersionSpreadChargeKernel->setArg(8, recipBoxVectors[1]);
                pmeDispersionSpreadChargeKernel->setArg(9, recipBoxVectors[2]);
            }
            else {
                pmeDispersionSpreadChargeKernel->setArg(7, recipBoxVectorsFloat[0]);
                pmeDispersionSpreadChargeKernel->setArg(8, recipBoxVectorsFloat[1]);
                pmeDispersionSpreadChargeKernel->setArg(9, recipBoxVectorsFloat[2]);
            }
            pmeDispersionSpreadChargeKernel->execute(cl.getNumAtoms());
            pmeDispersionFinishSpreadChargeKernel->execute(gridSizeX*gridSizeY*gridSizeZ);
            dispersionFft->execFFT(pmeGrid1, pmeGrid2, true);
            if (cl.getUseDoublePrecision()) {
                pmeDispersionConvolutionKernel->setArg(4, recipBoxVectors[0]);
                pmeDispersionConvolutionKernel->setArg(5, recipBoxVectors[1]);
                pmeDispersionConvolutionKernel->setArg(6, recipBoxVectors[2]);
                pmeDispersionEvalEnergyKernel->setArg(5, recipBoxVectors[0]);
                pmeDispersionEvalEnergyKernel->setArg(6, recipBoxVectors[1]);
                pmeDispersionEvalEnergyKernel->setArg(7, recipBoxVectors[2]);
            }
            else {
                pmeDispersionConvolutionKernel->setArg(4, recipBoxVectorsFloat[0]);
                pmeDispersionConvolutionKernel->setArg(5, recipBoxVectorsFloat[1]);
                pmeDispersionConvolutionKernel->setArg(6, recipBoxVectorsFloat[2]);
                pmeDispersionEvalEnergyKernel->setArg(5, recipBoxVectorsFloat[0]);
                pmeDispersionEvalEnergyKernel->setArg(6, recipBoxVectorsFloat[1]);
                pmeDispersionEvalEnergyKernel->setArg(7, recipBoxVectorsFloat[2]);
            }
            if (!hasCoulomb) cl.clearBuffer(pmeEnergyBuffer);
            if (includeEnergy)
                pmeDispersionEvalEnergyKernel->execute(gridSizeX*gridSizeY*gridSizeZ);
            pmeDispersionConvolutionKernel->execute(gridSizeX*gridSizeY*gridSizeZ);
            dispersionFft->execFFT(pmeGrid2, pmeGrid1, false);
            setPeriodicBoxArgs(cl, pmeDispersionInterpolateForceKernel, 3);
            if (cl.getUseDoublePrecision()) {
                pmeDispersionInterpolateForceKernel->setArg(8, recipBoxVectors[0]);
                pmeDispersionInterpolateForceKernel->setArg(9, recipBoxVectors[1]);
                pmeDispersionInterpolateForceKernel->setArg(10, recipBoxVectors[2]);
            }
            else {
                pmeDispersionInterpolateForceKernel->setArg(8, recipBoxVectorsFloat[0]);
                pmeDispersionInterpolateForceKernel->setArg(9, recipBoxVectorsFloat[1]);
                pmeDispersionInterpolateForceKernel->setArg(10, recipBoxVectorsFloat[2]);
            }
            if (deviceIsCpu)
                pmeDispersionInterpolateForceKernel->execute(2*10, 1);
            else
                pmeDispersionInterpolateForceKernel->execute(cl.getNumAtoms());
        }
        if (usePmeQueue) {
            // Metal: single command queue
        }
    }
    if (dispersionCoefficient != 0.0 && includeDirect) {
        mm_double4 boxSize = cl.getPeriodicBoxSizeDouble();
        energy += dispersionCoefficient/(boxSize.x*boxSize.y*boxSize.z);
    }
    return energy;
}

void MetalCalcNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const NonbondedForce& force) {
    // Make sure the new parameters are acceptable.

    if (force.getNumParticles() != cl.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    if (!hasCoulomb || !hasLJ) {
        for (int i = 0; i < force.getNumParticles(); i++) {
            double charge, sigma, epsilon;
            force.getParticleParameters(i, charge, sigma, epsilon);
            if (!hasCoulomb && charge != 0.0)
                throw OpenMMException("updateParametersInContext: The nonbonded force kernel does not include Coulomb interactions, because all charges were originally 0");
            if (!hasLJ && epsilon != 0.0)
                throw OpenMMException("updateParametersInContext: The nonbonded force kernel does not include Lennard-Jones interactions, because all epsilons were originally 0");
        }
    }
    set<int> exceptionsWithOffsets;
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge, sigma, epsilon;
        force.getExceptionParameterOffset(i, param, exception, charge, sigma, epsilon);
        exceptionsWithOffsets.insert(exception);
    }
    vector<int> exceptions;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        if (chargeProd != 0.0 || epsilon != 0.0 || exceptionsWithOffsets.find(i) != exceptionsWithOffsets.end())
            exceptions.push_back(i);
    }
    int numContexts = cl.getPlatformData().contexts.size();
    int startIndex = cl.getContextIndex()*exceptions.size()/numContexts;
    int endIndex = (cl.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    if (numExceptions != exceptionAtoms.size())
        throw OpenMMException("updateParametersInContext: The set of non-excluded exceptions has changed");

    // Record the per-particle parameters.

    vector<mm_float4> baseParticleParamVec(cl.getPaddedNumAtoms(), mm_float4(0, 0, 0, 0));
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge, sigma, epsilon;
        force.getParticleParameters(i, charge, sigma, epsilon);
        baseParticleParamVec[i] = mm_float4(charge, sigma, epsilon, 0);
    }
    baseParticleParams.upload(baseParticleParamVec);
    
    // Record the exceptions.
    
    if (numExceptions > 0) {
        vector<mm_float4> baseExceptionParamsVec(numExceptions);
        for (int i = 0; i < numExceptions; i++) {
            int particle1, particle2;
            double chargeProd, sigma, epsilon;
            force.getExceptionParameters(exceptions[startIndex+i], particle1, particle2, chargeProd, sigma, epsilon);
            if (make_pair(particle1, particle2) != exceptionAtoms[i])
                throw OpenMMException("updateParametersInContext: The set of non-excluded exceptions has changed");
            baseExceptionParamsVec[i] = mm_float4(chargeProd, sigma, epsilon, 0);
        }
        baseExceptionParams.upload(baseExceptionParamsVec);
    }
    
    // Compute other values.
    
    ewaldSelfEnergy = 0.0;
    if (nonbondedMethod == Ewald || nonbondedMethod == PME || nonbondedMethod == LJPME) {
        if (cl.getContextIndex() == 0) {
            for (int i = 0; i < force.getNumParticles(); i++) {
                ewaldSelfEnergy -= baseParticleParamVec[i].x*baseParticleParamVec[i].x*ONE_4PI_EPS0*alpha/sqrt(M_PI);
                if (doLJPME)
                    ewaldSelfEnergy += baseParticleParamVec[i].z*pow(baseParticleParamVec[i].y*dispersionAlpha, 6)/3.0;
            }
        }
    }
    if (force.getUseDispersionCorrection() && cl.getContextIndex() == 0 && (nonbondedMethod == CutoffPeriodic || nonbondedMethod == Ewald || nonbondedMethod == PME))
        dispersionCoefficient = NonbondedForceImpl::calcDispersionCorrection(context.getSystem(), force);
    cl.invalidateMolecules(info);
    recomputeParams = true;
}

void MetalCalcNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (nonbondedMethod != PME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    if (cl.getPlatformData().useCpuPme)
        cpuPme.getAs<CalcPmeReciprocalForceKernel>().getPMEParameters(alpha, nx, ny, nz);
    else {
        alpha = this->alpha;
        nx = gridSizeX;
        ny = gridSizeY;
        nz = gridSizeZ;
    }
}

void MetalCalcNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (nonbondedMethod != LJPME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    if (cl.getPlatformData().useCpuPme)
        //cpuPme.getAs<CalcPmeReciprocalForceKernel>().getLJPMEParameters(alpha, nx, ny, nz);
        throw OpenMMException("getPMEParametersInContext: CPUPME has not been implemented for LJPME yet.");
    else {
        alpha = this->dispersionAlpha;
        nx = dispersionGridSizeX;
        ny = dispersionGridSizeY;
        nz = dispersionGridSizeZ;
    }
}
