#ifndef OPENMM_OPENCLNONBONDEDUTILITIES_H_
#define OPENMM_OPENCLNONBONDEDUTILITIES_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2009-2022 Stanford University and the Authors.      *
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

#include "openmm/System.h"
#include "MetalArray.h"
#include "MetalExpressionUtilities.h"
#include "openmm/common/NonbondedUtilities.h"
#include <sstream>
#include <string>
#include <vector>

namespace OpenMM {

class MetalContext;
class MetalSort;

/**
 * This class provides a generic interface for calculating nonbonded interactions.
 */

class OPENMM_EXPORT_COMMON MetalNonbondedUtilities : public NonbondedUtilities {
public:
    class ParameterInfo;
    MetalNonbondedUtilities(MetalContext& context);
    ~MetalNonbondedUtilities();
    void addInteraction(bool usesCutoff, bool usesPeriodic, bool usesExclusions, double cutoffDistance, const std::vector<std::vector<int> >& exclusionList, const std::string& kernel, int forceGroup, bool useNeighborList);

    void addInteraction(bool usesCutoff, bool usesPeriodic, bool usesExclusions, double cutoffDistance, const std::vector<std::vector<int> >& exclusionList, const std::string& kernel, int forceGroup) {
        addInteraction(usesCutoff, usesPeriodic, usesExclusions, cutoffDistance, exclusionList, kernel, forceGroup, true);
    }

    void addParameter(ComputeParameterInfo parameter);
    void addParameter(const ParameterInfo& parameter);
    void addArgument(ComputeParameterInfo parameter);
    void addArgument(const ParameterInfo& parameter);
    std::string addEnergyParameterDerivative(const std::string& param);
    void requestExclusions(const std::vector<std::vector<int> >& exclusionList);
    void initialize(const System& system);
    int getNumForceBuffers() const {
        return 1;
    }
    int getNumEnergyBuffers() {
        return numForceThreadBlocks*forceThreadBlockSize;
    }
    bool getUseCutoff() {
        return useCutoff;
    }
    bool getUsePeriodic() {
        return usePeriodic;
    }
    int getNumForceThreadBlocks() {
        return numForceThreadBlocks;
    }
    int getForceThreadBlockSize() {
        return forceThreadBlockSize;
    }
    double getMaxCutoffDistance();
    bool getHasInteractions() {
        return (groupCutoff.size() > 0);
    }
    double padCutoff(double cutoff);
    void prepareInteractions(int forceGroups);
    void computeInteractions(int forceGroups, bool includeForces, bool includeEnergy);
    bool updateNeighborListSize();
    MetalArray& getBlockCenters() {
        return blockCenter;
    }
    MetalArray& getBlockBoundingBoxes() {
        return blockBoundingBox;
    }
    MetalArray& getInteractionCount() {
        return interactionCount;
    }
    MetalArray& getInteractingTiles() {
        return interactingTiles;
    }
    MetalArray& getInteractingAtoms() {
        return interactingAtoms;
    }
    MetalArray& getExclusions() {
        return exclusions;
    }
    MetalArray& getExclusionTiles() {
        return exclusionTiles;
    }
    MetalArray& getExclusionIndices() {
        return exclusionIndices;
    }
    MetalArray& getExclusionRowIndices() {
        return exclusionRowIndices;
    }
    MetalArray& getRebuildNeighborList() {
        return rebuildNeighborList;
    }
    int getStartTileIndex() const {
        return startTileIndex;
    }
    int getNumTiles() const {
        return numTiles;
    }
    void setUsePadding(bool padding);
    void setAtomBlockRange(double startFraction, double endFraction);
    /**
     * Create a Kernel for evaluating a nonbonded interaction.
     */
    ComputeKernel createInteractionKernel(const std::string& source, const std::vector<ParameterInfo>& params, const std::vector<ParameterInfo>& arguments, bool useExclusions, bool isSymmetric, int groups, bool includeForces, bool includeEnergy);
    void createKernelsForGroups(int groups);
    void setKernelSource(const std::string& source);
private:
    class KernelSet;
    class BlockSortTrait;
    MetalContext& context;
    std::map<int, KernelSet> groupKernels;
    MetalArray exclusionTiles;
    MetalArray exclusions;
    MetalArray exclusionIndices;
    MetalArray exclusionRowIndices;
    MetalArray interactingTiles;
    MetalArray interactingAtoms;
    MetalArray interactionCount;
    MetalArray blockCenter;
    MetalArray blockBoundingBox;
    MetalArray sortedBlocks;
    MetalArray sortedBlockCenter;
    MetalArray sortedBlockBoundingBox;
    MetalArray largeBlockCenter;
    MetalArray largeBlockBoundingBox;
    MetalArray oldPositions;
    MetalArray rebuildNeighborList;
    MetalSort* blockSorter;
    MetalArray pinnedCountArray;
    unsigned int* pinnedCountMemory;
    std::vector<std::vector<int> > atomExclusions;
    std::vector<ParameterInfo> parameters;
    std::vector<ParameterInfo> arguments;
    std::vector<std::string> energyParameterDerivatives;
    std::map<int, double> groupCutoff;
    std::map<int, std::string> groupKernelSource;
    double lastCutoff;
    bool useCutoff, usePeriodic, deviceIsCpu, anyExclusions, usePadding, useNeighborList, forceRebuildNeighborList, useLargeBlocks;
    int startTileIndex, startBlockIndex, numBlocks, maxExclusions, numForceThreadBlocks;
    int forceThreadBlockSize, interactingBlocksThreadBlockSize, groupFlags;
    unsigned int tilesAfterReorder;
    long long numTiles;
    std::string kernelSource;
};

/**
 * This class stores the kernels to execute for a set of force groups.
 */

class MetalNonbondedUtilities::KernelSet {
public:
    bool hasForces;
    double cutoffDistance;
    std::string source;
    ComputeKernel forceKernel, energyKernel, forceEnergyKernel;
    ComputeKernel findBlockBoundsKernel;
    ComputeKernel sortBoxDataKernel;
    ComputeKernel findInteractingBlocksKernel;
    ComputeKernel findInteractionsWithinBlocksKernel;
};

/**
 * This class stores information about a per-atom parameter that may be used in a nonbonded kernel.
 */

class MetalNonbondedUtilities::ParameterInfo {
public:
    /**
     * Create a ParameterInfo object.
     *
     * @param name           the name of the parameter
     * @param type           the data type of the parameter's components
     * @param numComponents  the number of components in the parameter
     * @param size           the size of the parameter in bytes
     * @param buffer         the Metal buffer (void*) containing the parameter values
     * @param constant       whether the memory should be marked as constant
     */
    ParameterInfo(const std::string& name, const std::string& componentType, int numComponents, int size, void* buffer, bool constant=true, MetalArray* metalArray=nullptr) :
            name(name), componentType(componentType), numComponents(numComponents), size(size), buffer(buffer), constant(constant), metalArray(metalArray) {
        if (numComponents == 1)
            type = componentType;
        else {
            std::stringstream s;
            s << componentType << numComponents;
            type = s.str();
        }
    }
    const std::string& getName() const {
        return name;
    }
    const std::string& getComponentType() const {
        return componentType;
    }
    const std::string& getType() const {
        return type;
    }
    int getNumComponents() const {
        return numComponents;
    }
    int getSize() const {
        return size;
    }
    void* getBuffer() const {
        return buffer;
    }
    bool isConstant() const {
        return constant;
    }
    MetalArray* getMetalArray() const {
        return metalArray;
    }
private:
    std::string name;
    std::string componentType;
    std::string type;
    int size, numComponents;
    void* buffer;
    bool constant;
    MetalArray* metalArray;
};

} // namespace OpenMM

#endif /*OPENMM_OPENCLNONBONDEDUTILITIES_H_*/
