#ifndef OPENMM_METALCONTEXT_H_
#define OPENMM_METALCONTEXT_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2009-2019 Stanford University and the Authors.      *
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

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <pthread.h>
#include "openmm/common/windowsExportCommon.h"
#include "MetalArray.h"
#include "MetalBondedUtilities.h"
#include "MetalExpressionUtilities.h"
#include "MetalIntegrationUtilities.h"
#include "MetalLogging.h"
#include "MetalNonbondedUtilities.h"
#include "MetalPlatform.h"
#include "openmm/common/ComputeContext.h"

namespace OpenMM {

class MetalForceInfo;

/**
 * Extra vector types beyond the ones in ComputeVectorTypes.h.
 * These use plain C++ types (no OpenCL dependency).
 */

struct mm_float8 {
    float s0, s1, s2, s3, s4, s5, s6, s7;
    mm_float8() {
    }
    mm_float8(float s0, float s1, float s2, float s3, float s4, float s5, float s6, float s7) :
        s0(s0), s1(s1), s2(s2), s3(s3), s4(s4), s5(s5), s6(s6), s7(s7) {
    }
};
struct mm_float16 {
    float s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15;
    mm_float16() {
    }
    mm_float16(float s0, float s1, float s2, float s3, float s4, float s5, float s6, float s7,
            float s8, float s9, float s10, float s11, float s12, float s13, float s14, float s15) :
        s0(s0), s1(s1), s2(s2), s3(s3), s4(s4), s5(s5), s6(s6), s7(s7),
        s8(s8), s9(s9), s10(s10), s11(s11), s12(s12), s13(s13), s14(s14), s15(s15) {
    }
};
struct mm_ushort2 {
    unsigned short x, y;
    mm_ushort2() {
    }
    mm_ushort2(unsigned short x, unsigned short y) : x(x), y(y) {
    }
};
struct mm_int8 {
    int s0, s1, s2, s3, s4, s5, s6, s7;
    mm_int8() {
    }
    mm_int8(int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7) :
        s0(s0), s1(s1), s2(s2), s3(s3), s4(s4), s5(s5), s6(s6), s7(s7) {
    }
};
struct mm_int16 {
    int s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15;
    mm_int16() {
    }
    mm_int16(int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7,
            int s8, int s9, int s10, int s11, int s12, int s13, int s14, int s15) :
        s0(s0), s1(s1), s2(s2), s3(s3), s4(s4), s5(s5), s6(s6), s7(s7),
        s8(s8), s9(s9), s10(s10), s11(s11), s12(s12), s13(s13), s14(s14), s15(s15) {
    }
};

/**
 * This class contains the information associated with a Context by the Metal Platform.  Each MetalContext is
 * specific to a particular device, and manages data structures and kernels for that device.  When running a simulation
 * in parallel on multiple devices, there is a separate MetalContext for each one.  The list of all contexts is
 * stored in the MetalPlatform::PlatformData.
 * <p>
 * In addition, a worker thread is created for each MetalContext.  This is used for parallel computations, so that
 * blocking calls to one device will not block other devices.  When only a single device is being used, the worker
 * thread is not used and calculations are performed on the main application thread.
 */

class OPENMM_EXPORT_COMMON MetalContext : public ComputeContext {
public:
    class WorkTask;
    class ReorderListener;
    class ForcePreComputation;
    class ForcePostComputation;
    static const int ThreadBlockSize;
    static const int TileSize;
    MetalContext(const System& system, int platformIndex, int deviceIndex, const std::string& precision, MetalPlatform::PlatformData& platformData,
        MetalContext* originalContext);
    ~MetalContext();
    /**
     * This is called to initialize internal data structures after all Forces in the system
     * have been initialized.
     */
    void initialize();
    /**
     * Add an ComputeForceInfo to this context.
     */
    void addForce(ComputeForceInfo* force);
    /**
     * Request that the context provide at least a particular number of force buffers.
     * Force kernels should call this during initialization.
     */
    void requestForceBuffers(int minBuffers);
    /**
     * Get the Metal device (id<MTLDevice>) associated with this object.
     * In .mm files, cast the return value to id<MTLDevice>.
     */
    void* getMTLDevice() const {
        return mtlDevice;
    }
    /**
     * Get the index of the device associated with this object.
     */
    int getDeviceIndex() {
        return deviceIndex;
    }
    /**
     * Get the index of the platform associated with this object.
     */
    int getPlatformIndex() {
        return platformIndex;
    }
    /**
     * Get the PlatformData object this context is part of.
     */
    MetalPlatform::PlatformData& getPlatformData() {
        return platformData;
    }
    /**
     * Get the number of contexts being used for the current simulation.
     * This is relevant when a simulation is parallelized across multiple devices.  In that case,
     * one MetalContext is created for each device.
     */
    int getNumContexts() const {
        return platformData.contexts.size();
    }
    /**
     * Get the index of this context in the list stored in the PlatformData.
     */
    int getContextIndex() const {
        return contextIndex;
    }
    /**
     * Get the Metal command queue (id<MTLCommandQueue>) currently being used for execution.
     * In .mm files, cast the return value to id<MTLCommandQueue>.
     */
    void* getMTLCommandQueue() const {
        return mtlCommandQueue;
    }
    /**
     * Construct an uninitialized array of the appropriate class for this platform.  The returned
     * value should be created on the heap with the "new" operator.
     */
    MetalArray* createArray();
    /**
     * Construct a ComputeEvent object of the appropriate class for this platform.
     */
    ComputeEvent createEvent();
    /**
     * Compile source code to create a ComputeProgram.
     *
     * @param source             the source code of the program
     * @param defines            a set of preprocessor definitions (name, value) to define when compiling the program
     */
    ComputeProgram compileProgram(const std::string source, const std::map<std::string, std::string>& defines=std::map<std::string, std::string>());
    /**
     * Convert an array to an MetalArray.  If the argument is already an MetalArray, this simply casts it.
     * If the argument is a ComputeArray that wraps an MetalArray, this returns the wrapped array.  For any
     * other argument, this throws an exception.
     */
    MetalArray& unwrap(ArrayInterface& array) const;
    /**
     * Get the array which contains the position (the xyz components) and charge (the w component) of each atom.
     */
    MetalArray& getPosq() {
        return posq;
    }
    /**
     * Get the array which contains a correction to the position of each atom.  This only exists if getUseMixedPrecision() returns true.
     */
    MetalArray& getPosqCorrection() {
        return posqCorrection;
    }
    /**
     * Get the array which contains the velocity (the xyz components) and inverse mass (the w component) of each atom.
     */
    MetalArray& getVelm() {
        return velm;
    }
    /**
     * Get the array which contains the force on each atom.
     */
    MetalArray& getForce() {
        return force;
    }
    /**
     * Get the array which contains the buffers in which forces are computed.
     */
    MetalArray& getForceBuffers() {
        return forceBuffers;
    }
    /**
     * Get the array which contains a contribution to each force represented as a real4.
     * This is a synonym for getForce().  It exists to satisfy the ComputeContext interface.
     */
    ArrayInterface& getFloatForceBuffer() {
        return force;
    }
    /**
     * Get the array which contains a contribution to each force represented as 64 bit fixed point.
     */
    MetalArray& getLongForceBuffer() {
        return longForceBuffer;
    }
    /**
     * Get the array which contains nonbonded forces accumulated as 32-bit float atomics.
     */
    MetalArray& getAtomicForceBuffer() {
        return atomicForceBuffer;
    }
    /**
     * Get the array which contains the buffer in which energy is computed.
     */
    MetalArray& getEnergyBuffer() {
        return energyBuffer;
    }
    /**
     * Get the array which contains the buffer in which derivatives of the energy with respect to parameters are computed.
     */
    MetalArray& getEnergyParamDerivBuffer() {
        return energyParamDerivBuffer;
    }
    /**
     * Get a pointer to a block of pinned memory that can be used for efficient transfers between host and device.
     * This is guaranteed to be at least as large as any of the arrays returned by methods of this class.
     */
    void* getPinnedBuffer() {
        return pinnedMemory;
    }
    /**
     * Get a shared ThreadPool that code can use to parallelize operations.
     *
     * Because this object is freely available to all code, care is needed to avoid conflicts.  Only use it
     * from the main thread, and make sure all operations are complete before you invoke any other code that
     * might make use of it
     */
    ThreadPool& getThreadPool() {
        return getPlatformData().threads;
    }
    /**
     * Get the array which contains the index of each atom.
     */
    MetalArray& getAtomIndexArray() {
        return atomIndexDevice;
    }
    /**
     * Create a Metal library (id<MTLLibrary>) from source code.
     * In .mm files, cast the return value to id<MTLLibrary>.
     *
     * @param source             the source code of the program
     * @param optimizationFlags  the optimization flags to pass to the Metal compiler.  If this is
     *                           omitted, a default set of options will be used
     */
    void* createProgram(const std::string source, const char* optimizationFlags = NULL);
    /**
     * Create a Metal library (id<MTLLibrary>) from source code.
     * In .mm files, cast the return value to id<MTLLibrary>.
     *
     * @param source             the source code of the program
     * @param defines            a set of preprocessor definitions (name, value) to define when compiling the program
     * @param optimizationFlags  the optimization flags to pass to the Metal compiler.  If this is
     *                           omitted, a default set of options will be used
     */
    void* createProgram(const std::string source, const std::map<std::string, std::string>& defines, const char* optimizationFlags = NULL);
    /**
     * Execute a kernel (compute pipeline state) with full argument lists.
     * This is the primary dispatch method used by MetalKernel.
     *
     * @param pipelineState          the Metal compute pipeline state (id<MTLComputePipelineState>)
     * @param kernelName             the name of the kernel for debugging
     * @param workUnits              the maximum number of work units that should be used
     * @param blockSize              the size of each thread block to use
     * @param arrayArgs              array arguments (MetalArray pointers, NULL for non-array slots)
     * @param primitiveArgs          primitive argument bytes for each slot
     * @param threadgroupMemorySizes threadgroup memory sizes keyed by argument index
     */
    void executeKernel(void* pipelineState, const std::string& kernelName,
                       int workUnits, int blockSize,
                       const std::vector<MetalArray*>& arrayArgs,
                       const std::vector<std::vector<uint8_t>>& primitiveArgs,
                       const std::map<int, int>& threadgroupMemorySizes = std::map<int, int>());
    /**
     * Execute a kernel (compute pipeline state) with simple arguments.
     *
     * @param pipelineState  the Metal compute pipeline state (id<MTLComputePipelineState>)
     * @param workUnits      the maximum number of work units that should be used
     * @param blockSize      the size of each thread block to use
     */
    void executeKernel(void* pipelineState, int workUnits, int blockSize = -1);
    /**
     * Compute the largest thread block size that can be used for a kernel that requires a particular amount of
     * shared memory per thread.
     *
     * @param memory        the number of bytes of shared memory per thread
     */
    int computeThreadBlockSize(double memory) const;
    /**
     * Set all elements of an array to 0.
     */
    void clearBuffer(ArrayInterface& array);
    /**
     * Set all elements of an array to 0.
     *
     * @param buffer   the Metal buffer (id<MTLBuffer>) to clear
     * @param size     the size of the buffer in bytes
     */
    void clearBuffer(void* buffer, int size);
    /**
     * Register a buffer that should be automatically cleared (all elements set to 0) at the start of each force or energy computation.
     */
    void addAutoclearBuffer(ArrayInterface& array);
    /**
     * Register a buffer that should be automatically cleared (all elements set to 0) at the start of each force or energy computation.
     *
     * @param buffer   the Metal buffer (id<MTLBuffer>) to clear
     * @param size     the size of the buffer in bytes
     */
    void addAutoclearBuffer(void* buffer, int size);
    /**
     * Clear all buffers that have been registered with addAutoclearBuffer().
     */
    void clearAutoclearBuffers();
    /**
     * Given a collection of floating point buffers packed into an array, sum them and store
     * the sum in the first buffer.
     * Also, write the result into a 64-bit fixed point buffer (overwriting its contents).
     *
     * @param array       the array containing the buffers to reduce
     * @param longBuffer  the 64-bit fixed point buffer to write the result into
     * @param numBuffers  the number of buffers packed into the array
     */
    void reduceBuffer(MetalArray& array, MetalArray& longBuffer, int numBuffers);
    /**
     * Sum the buffers containing forces.
     */
    void reduceForces();
    /**
     * Sum the buffer containing energy.
     */
    double reduceEnergy();
    /**
     * Get the current simulation time.
     */
    double getTime() {
        return time;
    }
    /**
     * Set the current simulation time.
     */
    void setTime(double t) {
        time = t;
    }
    /**
     * Get the number of integration steps that have been taken.
     */
    long long getStepCount() {
        return stepCount;
    }
    /**
     * Set the number of integration steps that have been taken.
     */
    void setStepCount(long long steps) {
        stepCount = steps;
    }
    /**
     * Get the number of times forces or energy has been computed.
     */
    int getComputeForceCount() {
        return computeForceCount;
    }
    /**
     * Set the number of times forces or energy has been computed.
     */
    void setComputeForceCount(int count) {
        computeForceCount = count;
    }
    /**
     * Get the number of time steps since the atoms were reordered.
     */
    int getStepsSinceReorder() const {
        return stepsSinceReorder;
    }
    /**
     * Set the number of time steps since the atoms were reordered.
     */
    void setStepsSinceReorder(int steps) {
        stepsSinceReorder = steps;
    }
    /**
     * Get the flag that marks whether the current force evaluation is valid.
     */
    bool getForcesValid() const {
        return forcesValid;
    }
    /**
     * Get the flag that marks whether the current force evaluation is valid.
     */
    void setForcesValid(bool valid) {
        forcesValid = valid;
    }
    /**
     * Get the number of blocks of TileSize atoms.
     */
    int getNumAtomBlocks() const {
        return numAtomBlocks;
    }
    /**
     * Get the standard number of thread blocks to use when executing kernels.
     */
    int getNumThreadBlocks() const {
        return numThreadBlocks;
    }
    /**
     * Get the maximum number of threads in a thread block supported by this device.
     */
    int getMaxThreadBlockSize() const {
        return maxThreadBlockSize;
    }
    /**
     * Get the number of force buffers.
     */
    int getNumForceBuffers() const {
        return numForceBuffers;
    }
    /**
     * Get whether the device being used is a CPU.
     * Metal always runs on GPU, so this always returns false.
     */
    bool getIsCPU() const {
        return false;
    }
    /**
     * Get the SIMD width of the device being used.
     */
    int getSIMDWidth() const {
        return simdWidth;
    }
    /**
     * Get whether the device being used supports 64 bit atomic operations on global memory.
     */
    bool getSupports64BitGlobalAtomics() const {
        return supports64BitGlobalAtomics;
    }
    /**
     * Get whether the device being used supports double precision math.
     */
    bool getSupportsDoublePrecision() const {
      return false;
    }

private:
  __attribute__((__noinline__))
  void precisionFailure(bool reasonMixed) const {
    if (reasonMixed) {
      std::cout << METAL_LOG_HEADER << "Error: Detected usage of mixed precision.";
    } else {
      std::cout << METAL_LOG_HEADER << "Error: Detected usage of double precision.";
    }
    std::cout << std::endl;
    std::cout << METAL_LOG_HEADER << "This precision is no longer supported." << std::endl;
    std::cout << METAL_LOG_HEADER << "Quitting now." << std::endl;
    exit(8);
  }

public:
    /**
     * Get whether double precision is being used.
     */
    bool getUseDoublePrecision() const {
      if (useDoublePrecision) {
        precisionFailure(false);
      }
        return useDoublePrecision;
    }
    /**
     * Get whether mixed precision is being used.
     */
    bool getUseMixedPrecision() const {
      if (useMixedPrecision) {
        precisionFailure(true);
      }
        return useMixedPrecision;
    }
    /**
     * Get whether the periodic box is triclinic.
     */
    bool getBoxIsTriclinic() const {
        return boxIsTriclinic;
    }
    /**
     * Get the vectors defining the periodic box.
     */
    void getPeriodicBoxVectors(Vec3& a, Vec3& b, Vec3& c) const {
        a = Vec3(periodicBoxVecXDouble.x, periodicBoxVecXDouble.y, periodicBoxVecXDouble.z);
        b = Vec3(periodicBoxVecYDouble.x, periodicBoxVecYDouble.y, periodicBoxVecYDouble.z);
        c = Vec3(periodicBoxVecZDouble.x, periodicBoxVecZDouble.y, periodicBoxVecZDouble.z);
    }
    /**
     * Set the vectors defining the periodic box.
     */
    void setPeriodicBoxVectors(const Vec3& a, const Vec3& b, const Vec3& c) {
        periodicBoxVecX = mm_float4((float) a[0], (float) a[1], (float) a[2], 0.0f);
        periodicBoxVecY = mm_float4((float) b[0], (float) b[1], (float) b[2], 0.0f);
        periodicBoxVecZ = mm_float4((float) c[0], (float) c[1], (float) c[2], 0.0f);
        periodicBoxVecXDouble = mm_double4(a[0], a[1], a[2], 0.0);
        periodicBoxVecYDouble = mm_double4(b[0], b[1], b[2], 0.0);
        periodicBoxVecZDouble = mm_double4(c[0], c[1], c[2], 0.0);
        periodicBoxSize = mm_float4((float) a[0], (float) b[1], (float) c[2], 0.0f);
        invPeriodicBoxSize = mm_float4(1.0f/(float) a[0], 1.0f/(float) b[1], 1.0f/(float) c[2], 0.0f);
        periodicBoxSizeDouble = mm_double4(a[0], b[1], c[2], 0.0);
        invPeriodicBoxSizeDouble = mm_double4(1.0/a[0], 1.0/b[1], 1.0/c[2], 0.0);
    }
    /**
     * Get the size of the periodic box.
     */
    mm_float4 getPeriodicBoxSize() const {
        return periodicBoxSize;
    }
    /**
     * Get the size of the periodic box.
     */
    mm_double4 getPeriodicBoxSizeDouble() const {
        return periodicBoxSizeDouble;
    }
    /**
     * Get the inverse of the size of the periodic box.
     */
    mm_float4 getInvPeriodicBoxSize() const {
        return invPeriodicBoxSize;
    }
    /**
     * Get the inverse of the size of the periodic box.
     */
    mm_double4 getInvPeriodicBoxSizeDouble() const {
        return invPeriodicBoxSizeDouble;
    }
    /**
     * Get the first periodic box vector.
     */
    mm_float4 getPeriodicBoxVecX() {
        return periodicBoxVecX;
    }
    /**
     * Get the first periodic box vector.
     */
    mm_double4 getPeriodicBoxVecXDouble() {
        return periodicBoxVecXDouble;
    }
    /**
     * Get the second periodic box vector.
     */
    mm_float4 getPeriodicBoxVecY() {
        return periodicBoxVecY;
    }
    /**
     * Get the second periodic box vector.
     */
    mm_double4 getPeriodicBoxVecYDouble() {
        return periodicBoxVecYDouble;
    }
    /**
     * Get the third periodic box vector.
     */
    mm_float4 getPeriodicBoxVecZ() {
        return periodicBoxVecZ;
    }
    /**
     * Get the third periodic box vector.
     */
    mm_double4 getPeriodicBoxVecZDouble() {
        return periodicBoxVecZDouble;
    }
    /**
     * Get the MetalIntegrationUtilities for this context.
     */
    MetalIntegrationUtilities& getIntegrationUtilities() {
        return *integration;
    }
    /**
     * Get the MetalExpressionUtilities for this context.
     */
    MetalExpressionUtilities& getExpressionUtilities() {
        return *expression;
    }
    /**
     * Get the MetalBondedUtilities for this context.
     */
    MetalBondedUtilities& getBondedUtilities() {
        return *bonded;
    }
    /**
     * Get the MetalNonbondedUtilities for this context.
     */
    MetalNonbondedUtilities& getNonbondedUtilities() {
        return *nonbonded;
    }
    /**
     * Create a new NonbondedUtilities for use with this context.  This should be called
     * only in unusual situations, when a Force needs its own NonbondedUtilities object
     * separate from the standard one.  The caller is responsible for deleting the object
     * when it is no longer needed.
     */
    MetalNonbondedUtilities* createNonbondedUtilities() {
        return new MetalNonbondedUtilities(*this);
    }
    /**
     * This should be called by the Integrator from its own initialize() method.
     * It ensures all contexts are fully initialized.
     */
    void initializeContexts();
    /**
     * Set the particle charges.  These are packed into the fourth element of the posq array.
     */
    void setCharges(const std::vector<double>& charges);
    /**
     * Request to use the fourth element of the posq array for storing charges.  Since only one force can
     * do that, this returns true the first time it is called, and false on all subsequent calls.
     */
    bool requestPosqCharges();
    /**
     * Get the names of all parameters with respect to which energy derivatives are computed.
     */
    const std::vector<std::string>& getEnergyParamDerivNames() const {
        return energyParamDerivNames;
    }
    /**
     * Get a workspace data structure used for accumulating the values of derivatives of the energy
     * with respect to parameters.
     */
    std::map<std::string, double>& getEnergyParamDerivWorkspace() {
        return energyParamDerivWorkspace;
    }
    /**
     * Register that the derivative of potential energy with respect to a context parameter
     * will need to be calculated.  If this is called multiple times for a single parameter,
     * it is only added to the list once.
     *
     * @param param    the name of the parameter to add
     */
    void addEnergyParameterDerivative(const std::string& param);
    /**
     * Wait until all work that has been queued (kernel executions, asynchronous data transfers, etc.)
     * has been submitted to the device.  This does not mean it has necessarily been completed.
     * Calling this periodically may improve the responsiveness of the computer's GUI, but at the
     * expense of reduced simulation performance.
     */
    void flushQueue();
    /**
     * Ensure a command buffer exists. Creates one if needed, but does not create an encoder.
     */
    void ensureCommandBuffer();
    /**
     * End the active compute encoder without committing the command buffer.
     * Used to switch to a blit encoder on the same command buffer.
     */
    void endComputeEncoder();
    /**
     * Get the current command buffer (id<MTLCommandBuffer>) for blit operations.
     * Ends the active compute encoder and ensures a command buffer exists.
     */
    void* getCommandBufferForBlit();
private:
    MetalPlatform::PlatformData& platformData;
    void printProfilingEvents();
    /**
     * Create a utility kernel pipeline state from a Metal library.
     *
     * @param library  the Metal library (id<MTLLibrary>) containing the function
     * @param name     the function name in the library
     * @param pso      output: the pipeline state (id<MTLComputePipelineState>) stored as void*
     */
    void createUtilityKernel(void* library, const std::string& name, void*& pso);
    /**
     * Execute a utility kernel with sequential buffer + primitive arguments.
     *
     * @param pso                    the pipeline state to execute
     * @param name                   the kernel name for debugging
     * @param workUnits              the maximum number of work units
     * @param blockSize              the thread block size (-1 for default)
     * @param bufferArgs             array arguments (ordered)
     * @param primitiveArgValues     primitive arguments as (pointer, size) pairs
     * @param threadgroupMemorySizes threadgroup memory sizes keyed by argument index
     */
    void executeUtilityKernel(void* pso, const std::string& name,
                               int workUnits, int blockSize,
                               const std::vector<MetalArray*>& bufferArgs,
                               const std::vector<std::pair<const void*, int>>& primitiveArgValues,
                               const std::map<int, int>& threadgroupMemorySizes = std::map<int, int>());
    int deviceIndex;
    int platformIndex;
    int contextIndex;
    int numAtomBlocks;
    int numThreadBlocks;
    int numForceBuffers;
    int simdWidth;
    int maxThreadBlockSize;
    int reduceEnergyThreadgroups;
    bool supports64BitGlobalAtomics, supportsDoublePrecision, useDoublePrecision, useMixedPrecision, boxIsTriclinic, hasAssignedPosqCharges, enableKernelProfiling, enableDumpSource, enableLogArgs;
    int dumpSourceCounter;
    uint64_t profileStartTime;
    int profilingEventCount;
    mm_float4 periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ;
    mm_double4 periodicBoxSizeDouble, invPeriodicBoxSizeDouble, periodicBoxVecXDouble, periodicBoxVecYDouble, periodicBoxVecZDouble;
    std::string defaultOptimizationOptions;
    std::map<std::string, std::string> compilationDefines;
    void* mtlDevice;                    // id<MTLDevice>
    void* mtlCommandQueue;              // id<MTLCommandQueue>
    void* currentCommandBuffer;         // id<MTLCommandBuffer> — persistent batch (NULL when idle)
    void* currentComputeEncoder;        // id<MTLComputeCommandEncoder> — active encoder (NULL when idle)
    void* clearBufferPipeline;          // id<MTLComputePipelineState>
    void* clearTwoBuffersPipeline;      // id<MTLComputePipelineState>
    void* clearThreeBuffersPipeline;    // id<MTLComputePipelineState>
    void* clearFourBuffersPipeline;     // id<MTLComputePipelineState>
    void* clearFiveBuffersPipeline;     // id<MTLComputePipelineState>
    void* clearSixBuffersPipeline;      // id<MTLComputePipelineState>
    void* reduceReal4Pipeline;          // id<MTLComputePipelineState>
    void* reduceForcesPipeline;         // id<MTLComputePipelineState>
    void* reduceEnergyPipeline;         // id<MTLComputePipelineState>
    void* setChargesPipeline;           // id<MTLComputePipelineState>
    void* pinnedBuffer;                 // id<MTLBuffer> — shared/managed storage for host-device transfers
    void* pinnedMemory;                 // CPU-accessible pointer to pinnedBuffer contents
    MetalArray posq;
    MetalArray posqCorrection;
    MetalArray velm;
    MetalArray force;
    MetalArray forceBuffers;
    MetalArray longForceBuffer;
    MetalArray atomicForceBuffer;
    MetalArray energyBuffer;
    MetalArray energySum;
    MetalArray energyParamDerivBuffer;
    MetalArray atomIndexDevice;
    MetalArray chargeBuffer;
    std::vector<std::string> energyParamDerivNames;
    std::map<std::string, double> energyParamDerivWorkspace;
    std::vector<void*> autoclearBuffers;    // each is id<MTLBuffer>
    std::vector<int> autoclearBufferSizes;
    MetalIntegrationUtilities* integration;
    MetalExpressionUtilities* expression;
    MetalBondedUtilities* bonded;
    MetalNonbondedUtilities* nonbonded;
};

/**
 * This class exists only for backward compatibility.  Use ComputeContext::WorkTask instead.
 */
class OPENMM_EXPORT_COMMON MetalContext::WorkTask : public ComputeContext::WorkTask {
};

/**
 * This class exists only for backward compatibility.  Use ComputeContext::ReorderListener instead.
 */
class OPENMM_EXPORT_COMMON MetalContext::ReorderListener : public ComputeContext::ReorderListener {
};

/**
 * This class exists only for backward compatibility.  Use ComputeContext::ForcePreComputation instead.
 */
class OPENMM_EXPORT_COMMON MetalContext::ForcePreComputation : public ComputeContext::ForcePreComputation {
};

/**
 * This class exists only for backward compatibility.  Use ComputeContext::ForcePostComputation instead.
 */
class OPENMM_EXPORT_COMMON MetalContext::ForcePostComputation : public ComputeContext::ForcePostComputation {
};

} // namespace OpenMM

#endif /*OPENMM_METALCONTEXT_H_*/
