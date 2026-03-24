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

#import <Metal/Metal.h>
#include "openmm/OpenMMException.h"
#include "MetalNonbondedUtilities.h"
#include "MetalArray.h"
#include "MetalContext.h"
#include "MetalKernelSources.h"
#include "MetalLogging.h"
#include "MetalExpressionUtilities.h"
#include "MetalSort.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <utility>

using namespace OpenMM;
using namespace std;

class MetalNonbondedUtilities::BlockSortTrait : public MetalSort::SortTrait {
public:
    BlockSortTrait(bool useDouble) : useDouble(useDouble) {
    }
    int getDataSize() const {return useDouble ? sizeof(mm_double2) : sizeof(mm_float2);}
    int getKeySize() const {return useDouble ? sizeof(double) : sizeof(float);}
    const char* getDataType() const {return "real2";}
    const char* getKeyType() const {return "real";}
    const char* getMinKey() const {return "-MAXFLOAT";}
    const char* getMaxKey() const {return "MAXFLOAT";}
    const char* getMaxValue() const {return "(real2) (MAXFLOAT, MAXFLOAT)";}
    const char* getSortKey() const {return "value.x";}
private:
    bool useDouble;
};

MetalNonbondedUtilities::MetalNonbondedUtilities(MetalContext& context) : context(context), useCutoff(false), usePeriodic(false), useNeighborList(false), anyExclusions(false), usePadding(true),
        blockSorter(NULL), pinnedCountMemory(NULL), forceRebuildNeighborList(true), lastCutoff(0.0), groupFlags(0) {
    // Decide how many thread blocks and force buffers to use.
    // Metal always runs on GPU, never CPU.

    deviceIsCpu = false;

    if (context.getSIMDWidth() == 32) {
        // Apple Silicon
        int blocksPerCore = 6;
        numForceThreadBlocks = blocksPerCore * 10; // Approximate GPU core count
        forceThreadBlockSize = 256;
    }
    else {
        numForceThreadBlocks = context.getNumThreadBlocks();
        forceThreadBlockSize = (context.getSIMDWidth() >= 32 ? MetalContext::ThreadBlockSize : 32);
    }

    // Allocate pinned count buffer using Metal shared memory
    pinnedCountArray.initialize<unsigned int>(context, 1, "pinnedCountBuffer");
    vector<unsigned int> zeroCount(1, 0);
    pinnedCountArray.upload(zeroCount);
    // For Metal, we download the count synchronously
    pinnedCountMemory = new unsigned int[1];
    pinnedCountMemory[0] = 0;

    {
        // USE_LARGE_BLOCKS disabled for Metal
        this->useLargeBlocks = false;

        char *overrideUseLargeBlocks = getenv("OPENMM_METAL_USE_LARGE_BLOCKS");
        if (overrideUseLargeBlocks != nullptr) {
            if (strcmp(overrideUseLargeBlocks, "0") == 0) {
                this->useLargeBlocks = false;
            } else if (strcmp(overrideUseLargeBlocks, "1") == 0) {
                this->useLargeBlocks = true;
            } else {
                std::cout << std::endl;
                std::cout << METAL_LOG_HEADER << "Error: Invalid option for ";
                std::cout << "'OPENMM_METAL_USE_LARGE_BLOCKS'." << std::endl;
                std::cout << METAL_LOG_HEADER << "Specified '" << overrideUseLargeBlocks << "', but ";
                std::cout << "expected either '0' or '1'." << std::endl;
                std::cout << METAL_LOG_HEADER << "Quitting now." << std::endl;
                exit(7);
            }
        }
    }

    setKernelSource(MetalKernelSources::nonbonded);
}

MetalNonbondedUtilities::~MetalNonbondedUtilities() {
    if (blockSorter != NULL)
        delete blockSorter;
    if (pinnedCountMemory != NULL)
        delete[] pinnedCountMemory;
}

void MetalNonbondedUtilities::addInteraction(bool usesCutoff, bool usesPeriodic, bool usesExclusions, double cutoffDistance, const vector<vector<int> >& exclusionList, const string& kernel, int forceGroup, bool useNeighborList) {
    if (groupCutoff.size() > 0) {
        if (usesCutoff != useCutoff)
            throw OpenMMException("All Forces must agree on whether to use a cutoff");
        if (usesPeriodic != usePeriodic)
            throw OpenMMException("All Forces must agree on whether to use periodic boundary conditions");
        if (usesCutoff && groupCutoff.find(forceGroup) != groupCutoff.end() && groupCutoff[forceGroup] != cutoffDistance)
            throw OpenMMException("All Forces in a single force group must use the same cutoff distance");
    }
    if (usesExclusions)
        requestExclusions(exclusionList);
    useCutoff = usesCutoff;
    usePeriodic = usesPeriodic;
    this->useNeighborList |= (useNeighborList && useCutoff);

    {
        char *overrideUseList = getenv("OPENMM_METAL_USE_NEIGHBOR_LIST");
        if (overrideUseList != nullptr) {
            if (strcmp(overrideUseList, "0") == 0) {
                this->useNeighborList = false;
            } else if (strcmp(overrideUseList, "1") == 0) {
                if (useCutoff) {
                    this->useNeighborList = true;
                } else {
                    std::cout << std::endl;
                    std::cout << METAL_LOG_HEADER << "Error: Used a neighbor list without a cutoff. ";
                    std::cout << METAL_LOG_HEADER << "Quitting now." << std::endl;
                    exit(7);
                }
            } else {
                std::cout << std::endl;
                std::cout << METAL_LOG_HEADER << "Error: Invalid option for ";
                std::cout << "'OPENMM_METAL_USE_NEIGHBOR_LIST'." << std::endl;
                std::cout << METAL_LOG_HEADER << "Specified '" << overrideUseList << "', but ";
                std::cout << "expected either '0' or '1'." << std::endl;
                std::cout << METAL_LOG_HEADER << "Quitting now." << std::endl;
                exit(7);
            }
        }
    }

    groupCutoff[forceGroup] = cutoffDistance;
    groupFlags |= 1<<forceGroup;
    if (kernel.size() > 0) {
        if (groupKernelSource.find(forceGroup) == groupKernelSource.end())
            groupKernelSource[forceGroup] = "";
        map<string, string> replacements;
        replacements["CUTOFF"] = "CUTOFF_"+context.intToString(forceGroup);
        replacements["CUTOFF_SQUARED"] = "CUTOFF_"+context.intToString(forceGroup)+"_SQUARED";
        groupKernelSource[forceGroup] += context.replaceStrings(kernel, replacements)+"\n";
    }
}

void MetalNonbondedUtilities::addParameter(ComputeParameterInfo parameter) {
    MetalArray& array = context.unwrap(parameter.getArray());
    parameters.push_back(ParameterInfo(parameter.getName(), parameter.getComponentType(), parameter.getNumComponents(),
            parameter.getSize(), array.getDeviceBuffer(), parameter.isConstant(), &array));
}

void MetalNonbondedUtilities::addParameter(const ParameterInfo& parameter) {
    parameters.push_back(parameter);
}

void MetalNonbondedUtilities::addArgument(ComputeParameterInfo parameter) {
    MetalArray& array = context.unwrap(parameter.getArray());
    arguments.push_back(ParameterInfo(parameter.getName(), parameter.getComponentType(), parameter.getNumComponents(),
            parameter.getSize(), array.getDeviceBuffer(), parameter.isConstant(), &array));
}

void MetalNonbondedUtilities::addArgument(const ParameterInfo& parameter) {
    arguments.push_back(parameter);
}

string MetalNonbondedUtilities::addEnergyParameterDerivative(const string& param) {
    int index;
    for (index = 0; index < energyParameterDerivatives.size(); index++)
        if (param == energyParameterDerivatives[index])
            break;
    if (index == energyParameterDerivatives.size())
        energyParameterDerivatives.push_back(param);
    context.addEnergyParameterDerivative(param);
    return string("energyParamDeriv")+context.intToString(index);
}

void MetalNonbondedUtilities::requestExclusions(const vector<vector<int> >& exclusionList) {
    if (anyExclusions) {
        bool sameExclusions = (exclusionList.size() == atomExclusions.size());
        for (int i = 0; i < (int) exclusionList.size() && sameExclusions; i++) {
            if (exclusionList[i].size() != atomExclusions[i].size())
                sameExclusions = false;
            set<int> expectedExclusions;
            expectedExclusions.insert(atomExclusions[i].begin(), atomExclusions[i].end());
            for (int j = 0; j < (int) exclusionList[i].size(); j++)
                if (expectedExclusions.find(exclusionList[i][j]) == expectedExclusions.end())
                    sameExclusions = false;
        }
        if (!sameExclusions)
            throw OpenMMException("All Forces must have identical exceptions");
    }
    else {
        atomExclusions = exclusionList;
        anyExclusions = true;
    }
}

static bool compareInt2(mm_int2 a, mm_int2 b) {
    return ((a.y < b.y) || (a.y == b.y && a.x < b.x));
}

static bool compareInt2LargeSIMD(mm_int2 a, mm_int2 b) {
    if (a.x == a.y) {
        if (b.x == b.y)
            return (a.x < b.x);
        return true;
    }
    if (b.x == b.y)
        return false;
    return ((a.y < b.y) || (a.y == b.y && a.x < b.x));
}

void MetalNonbondedUtilities::initialize(const System& system) {
    // For Apple Silicon, tune thread blocks based on atom count.
    if (context.getSIMDWidth() == 32) {
        if (context.getNumAtoms() < 10000)
            numForceThreadBlocks /= 2;
    }

    if (atomExclusions.size() == 0) {
        atomExclusions.resize(context.getNumAtoms());
        for (int i = 0; i < (int) atomExclusions.size(); i++)
            atomExclusions[i].push_back(i);
    }

    int numAtomBlocks = context.getNumAtomBlocks();
    int numContexts = context.getPlatformData().contexts.size();
    setAtomBlockRange(context.getContextIndex()/(double) numContexts, (context.getContextIndex()+1)/(double) numContexts);

    set<pair<int, int> > tilesWithExclusions;
    for (int atom1 = 0; atom1 < (int) atomExclusions.size(); ++atom1) {
        int x = atom1/MetalContext::TileSize;
        for (int j = 0; j < (int) atomExclusions[atom1].size(); ++j) {
            int atom2 = atomExclusions[atom1][j];
            int y = atom2/MetalContext::TileSize;
            tilesWithExclusions.insert(make_pair(max(x, y), min(x, y)));
        }
    }
    vector<mm_int2> exclusionTilesVec;
    for (set<pair<int, int> >::const_iterator iter = tilesWithExclusions.begin(); iter != tilesWithExclusions.end(); ++iter)
        exclusionTilesVec.push_back(mm_int2(iter->first, iter->second));
    sort(exclusionTilesVec.begin(), exclusionTilesVec.end(), context.getSIMDWidth() <= 32 || !useCutoff ? compareInt2 : compareInt2LargeSIMD);
    exclusionTiles.initialize<mm_int2>(context, exclusionTilesVec.size(), "exclusionTiles");
    exclusionTiles.upload(exclusionTilesVec);
    map<pair<int, int>, int> exclusionTileMap;
    for (int i = 0; i < (int) exclusionTilesVec.size(); i++) {
        mm_int2 tile = exclusionTilesVec[i];
        exclusionTileMap[make_pair(tile.x, tile.y)] = i;
    }
    vector<vector<int> > exclusionBlocksForBlock(numAtomBlocks);
    for (set<pair<int, int> >::const_iterator iter = tilesWithExclusions.begin(); iter != tilesWithExclusions.end(); ++iter) {
        exclusionBlocksForBlock[iter->first].push_back(iter->second);
        if (iter->first != iter->second)
            exclusionBlocksForBlock[iter->second].push_back(iter->first);
    }
    vector<unsigned int> exclusionRowIndicesVec(numAtomBlocks+1, 0);
    vector<unsigned int> exclusionIndicesVec;
    for (int i = 0; i < numAtomBlocks; i++) {
        exclusionIndicesVec.insert(exclusionIndicesVec.end(), exclusionBlocksForBlock[i].begin(), exclusionBlocksForBlock[i].end());
        exclusionRowIndicesVec[i+1] = exclusionIndicesVec.size();
    }
    maxExclusions = 0;
    for (int i = 0; i < (int) exclusionBlocksForBlock.size(); i++)
        maxExclusions = (maxExclusions > exclusionBlocksForBlock[i].size() ? maxExclusions : exclusionBlocksForBlock[i].size());
    exclusionIndices.initialize<unsigned int>(context, exclusionIndicesVec.size(), "exclusionIndices");
    exclusionRowIndices.initialize<unsigned int>(context, exclusionRowIndicesVec.size(), "exclusionRowIndices");
    exclusionIndices.upload(exclusionIndicesVec);
    exclusionRowIndices.upload(exclusionRowIndicesVec);

    exclusions.initialize<unsigned int>(context, tilesWithExclusions.size()*MetalContext::TileSize, "exclusions");
    unsigned int allFlags = (unsigned int) -1;
    vector<unsigned int> exclusionVec(exclusions.getSize(), allFlags);
    for (int i = 0; i < exclusions.getSize(); ++i)
        exclusionVec[i] = 0xFFFFFFFF;
    for (int atom1 = 0; atom1 < (int) atomExclusions.size(); ++atom1) {
        int x = atom1/MetalContext::TileSize;
        int offset1 = atom1-x*MetalContext::TileSize;
        for (int j = 0; j < (int) atomExclusions[atom1].size(); ++j) {
            int atom2 = atomExclusions[atom1][j];
            int y = atom2/MetalContext::TileSize;
            int offset2 = atom2-y*MetalContext::TileSize;
            if (x > y) {
                int index = exclusionTileMap[make_pair(x, y)]*MetalContext::TileSize;
                exclusionVec[index+offset1] &= allFlags-(1<<offset2);
            }
            else {
                int index = exclusionTileMap[make_pair(y, x)]*MetalContext::TileSize;
                exclusionVec[index+offset2] &= allFlags-(1<<offset1);
            }
        }
    }
    atomExclusions.clear();
    exclusions.upload(exclusionVec);

    if (useCutoff) {
        int numAtoms = context.getNumAtoms();
        int maxTiles = 20*numAtomBlocks;
        if (maxTiles > numTiles)
            maxTiles = numTiles;
        if (maxTiles < 1)
            maxTiles = 1;
        interactingTiles.initialize<int>(context, maxTiles, "interactingTiles");
        interactingAtoms.initialize<int>(context, MetalContext::TileSize*maxTiles, "interactingAtoms");
        interactionCount.initialize<unsigned int>(context, 1, "interactionCount");
        int elementSize = (context.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
        blockCenter.initialize(context, numAtomBlocks, 4*elementSize, "blockCenter");
        blockBoundingBox.initialize(context, numAtomBlocks, 4*elementSize, "blockBoundingBox");
        sortedBlocks.initialize(context, numAtomBlocks, 2*elementSize, "sortedBlocks");
        sortedBlockCenter.initialize(context, numAtomBlocks+1, 4*elementSize, "sortedBlockCenter");
        sortedBlockBoundingBox.initialize(context, numAtomBlocks+1, 4*elementSize, "sortedBlockBoundingBox");
        largeBlockCenter.initialize(context, numAtomBlocks, 4*elementSize, "largeBlockCenter");
        largeBlockBoundingBox.initialize(context, numAtomBlocks, 4*elementSize, "largeBlockBoundingBox");
        oldPositions.initialize(context, numAtoms, 4*elementSize, "oldPositions");
        rebuildNeighborList.initialize<int>(context, 1, "rebuildNeighborList");

        blockSorter = new MetalSort(context, new BlockSortTrait(context.getUseDoublePrecision()), numAtomBlocks, false);
        vector<unsigned int> count(1, 0);
        interactionCount.upload(count);
        rebuildNeighborList.upload(count);
    }
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

double MetalNonbondedUtilities::getMaxCutoffDistance() {
    double cutoff = 0.0;
    for (map<int, double>::const_iterator iter = groupCutoff.begin(); iter != groupCutoff.end(); ++iter)
        cutoff = max(cutoff, iter->second);
    return cutoff;
}

double MetalNonbondedUtilities::padCutoff(double cutoff) {
    double padding = (usePadding ? 0.1*cutoff : 0.0);
    return cutoff+padding;
}

void MetalNonbondedUtilities::prepareInteractions(int forceGroups) {
    if ((forceGroups&groupFlags) == 0)
        return;
    if (groupKernels.find(forceGroups) == groupKernels.end())
        createKernelsForGroups(forceGroups);
    KernelSet& kernels = groupKernels[forceGroups];
    if (useCutoff && usePeriodic) {
        mm_float4 box = context.getPeriodicBoxSize();
        double minAllowedSize = 1.999999*kernels.cutoffDistance;
        if (box.x < minAllowedSize || box.y < minAllowedSize || box.z < minAllowedSize)
            throw OpenMMException("The periodic box size has decreased to less than twice the nonbonded cutoff.");
    }
    if (!useNeighborList)
        return;
    if (numTiles == 0)
        return;

    setPeriodicBoxArgs(context, kernels.findBlockBoundsKernel, 1);
    kernels.findBlockBoundsKernel->execute(context.getNumAtoms());
    if (!useLargeBlocks) {
        blockSorter->sort(sortedBlocks);
    }
    kernels.sortBoxDataKernel->setArg(9, (int)forceRebuildNeighborList);
    kernels.sortBoxDataKernel->execute(context.getNumAtoms());
    setPeriodicBoxArgs(context, kernels.findInteractingBlocksKernel, 0);

    kernels.findInteractingBlocksKernel->execute(context.getNumAtoms(), interactingBlocksThreadBlockSize);

    forceRebuildNeighborList = false;
    lastCutoff = kernels.cutoffDistance;

    // The GPU nonbonded kernel reads interactionCount directly from the GPU buffer.
    // The CPU only needs the count to check if the interactingTiles buffer needs growing
    // (rare — buffer has 20% headroom) or if a reorder is needed (checked after 25 steps).
    // Skip the download (and its expensive flushQueue) on most steps.
    int stepsSinceReorder = context.getStepsSinceReorder();
    if (stepsSinceReorder <= 1 || context.getComputeForceCount() % 8 == 0) {
        interactionCount.download(pinnedCountMemory);
    }
}

void MetalNonbondedUtilities::computeInteractions(int forceGroups, bool includeForces, bool includeEnergy) {
    if ((forceGroups&groupFlags) == 0)
        return;
    KernelSet& kernels = groupKernels[forceGroups];
    if (kernels.hasForces) {
        ComputeKernel& kernel = (includeForces ? (includeEnergy ? kernels.forceEnergyKernel : kernels.forceKernel) : kernels.energyKernel);
        if (kernel.get() == nullptr)
            kernel = createInteractionKernel(kernels.source, parameters, arguments, true, true, forceGroups, includeForces, includeEnergy);
        if (useCutoff)
            setPeriodicBoxArgs(context, kernel, 9);
        kernel->execute(numForceThreadBlocks*forceThreadBlockSize, forceThreadBlockSize);
    }
    if (useNeighborList && numTiles > 0) {
        updateNeighborListSize();
    }
}

bool MetalNonbondedUtilities::updateNeighborListSize() {
    if (!useCutoff)
        return false;
    if (context.getStepsSinceReorder() == 0 || tilesAfterReorder == 0)
        tilesAfterReorder = pinnedCountMemory[0];
    else if (context.getStepsSinceReorder() > 25 && pinnedCountMemory[0] > 1.1*tilesAfterReorder)
        context.forceReorder();
    if (pinnedCountMemory[0] <= interactingTiles.getSize())
        return false;

    unsigned int maxTiles = (unsigned int) (1.2*pinnedCountMemory[0]);
    unsigned int numBlocks = context.getNumAtomBlocks();
    int totalTiles = numBlocks*(numBlocks+1)/2;
    if (maxTiles > totalTiles)
        maxTiles = totalTiles;
    interactingTiles.resize(maxTiles);
    interactingAtoms.resize(MetalContext::TileSize*(size_t) maxTiles);
    for (map<int, KernelSet>::iterator iter = groupKernels.begin(); iter != groupKernels.end(); ++iter) {
        KernelSet& kernels = iter->second;
        if (kernels.forceKernel.get() != nullptr) {
            kernels.forceKernel->setArg(7, interactingTiles);
            kernels.forceKernel->setArg(14, (unsigned int)maxTiles);
            kernels.forceKernel->setArg(17, interactingAtoms);
        }
        if (kernels.energyKernel.get() != nullptr) {
            kernels.energyKernel->setArg(7, interactingTiles);
            kernels.energyKernel->setArg(14, (unsigned int)maxTiles);
            kernels.energyKernel->setArg(17, interactingAtoms);
        }
        if (kernels.forceEnergyKernel.get() != nullptr) {
            kernels.forceEnergyKernel->setArg(7, interactingTiles);
            kernels.forceEnergyKernel->setArg(14, (unsigned int)maxTiles);
            kernels.forceEnergyKernel->setArg(17, interactingAtoms);
        }
        kernels.findInteractingBlocksKernel->setArg(6, interactingTiles);
        kernels.findInteractingBlocksKernel->setArg(7, interactingAtoms);
        kernels.findInteractingBlocksKernel->setArg(9, (unsigned int)maxTiles);
    }
    forceRebuildNeighborList = true;
    context.setForcesValid(false);
    return true;
}

void MetalNonbondedUtilities::setUsePadding(bool padding) {
    usePadding = padding;
}

void MetalNonbondedUtilities::setAtomBlockRange(double startFraction, double endFraction) {
    int numAtomBlocks = context.getNumAtomBlocks();
    startBlockIndex = (int) (startFraction*numAtomBlocks);
    numBlocks = (int) (endFraction*numAtomBlocks)-startBlockIndex;
    long long totalTiles = context.getNumAtomBlocks()*((long long)context.getNumAtomBlocks()+1)/2;
    startTileIndex = (int) (startFraction*totalTiles);
    numTiles = (long long) (endFraction*totalTiles)-startTileIndex;
    if (useCutoff) {
        for (map<int, KernelSet>::iterator iter = groupKernels.begin(); iter != groupKernels.end(); ++iter) {
            KernelSet& kernels = iter->second;
            if (kernels.forceKernel.get() != nullptr) {
                kernels.forceKernel->setArg(5, (unsigned int)startTileIndex);
                kernels.forceKernel->setArg(6, (unsigned long long)numTiles);
            }
            if (kernels.energyKernel.get() != nullptr) {
                kernels.energyKernel->setArg(5, (unsigned int)startTileIndex);
                kernels.energyKernel->setArg(6, (unsigned long long)numTiles);
            }
            if (kernels.forceEnergyKernel.get() != nullptr) {
                kernels.forceEnergyKernel->setArg(5, (unsigned int)startTileIndex);
                kernels.forceEnergyKernel->setArg(6, (unsigned long long)numTiles);
            }
            kernels.findInteractingBlocksKernel->setArg(10, (unsigned int)startBlockIndex);
            kernels.findInteractingBlocksKernel->setArg(11, (unsigned int)numBlocks);
        }
        forceRebuildNeighborList = true;
    }
}

void MetalNonbondedUtilities::createKernelsForGroups(int groups) {
    KernelSet kernels;
    double cutoff = 0.0;
    string source;
    for (int i = 0; i < 32; i++) {
        if ((groups&(1<<i)) != 0) {
            cutoff = max(cutoff, groupCutoff[i]);
            source += groupKernelSource[i];
        }
    }
    kernels.hasForces = (source.size() > 0);
    kernels.cutoffDistance = cutoff;
    kernels.source = source;
    if (useCutoff) {
        double paddedCutoff = padCutoff(cutoff);
        map<string, string> defines;
        defines["TILE_SIZE"] = context.intToString(MetalContext::TileSize);
        defines["NUM_ATOMS"] = context.intToString(context.getNumAtoms());
        defines["PADDING"] = context.doubleToString(paddedCutoff-cutoff);
        defines["PADDED_CUTOFF"] = context.doubleToString(paddedCutoff);
        defines["PADDED_CUTOFF_SQUARED"] = context.doubleToString(paddedCutoff*paddedCutoff);
        defines["NUM_TILES_WITH_EXCLUSIONS"] = context.intToString(exclusionTiles.getSize());
        defines["NUM_BLOCKS"] = context.intToString(context.getNumAtomBlocks());
        defines["SIMD_WIDTH"] = context.intToString(context.getSIMDWidth());
        if (usePeriodic)
            defines["USE_PERIODIC"] = "1";
        if (context.getBoxIsTriclinic())
            defines["TRICLINIC"] = "1";
        if (useLargeBlocks)
            defines["USE_LARGE_BLOCKS"] = "1";
        defines["MAX_EXCLUSIONS"] = context.intToString(maxExclusions);
        defines["BUFFER_GROUPS"] = "2";
        string file = MetalKernelSources::findInteractingBlocks;
        int groupSize = (context.getSIMDWidth() < 32 ? 32 : 256);
        defines["GROUP_SIZE"] = context.intToString(groupSize);
        ComputeProgram interactingBlocksProgram = context.compileProgram(file, defines);

        kernels.findBlockBoundsKernel = interactingBlocksProgram->createKernel("findBlockBounds");
        kernels.findBlockBoundsKernel->addArg(context.getNumAtoms());
        // args 1-5 are periodic box (set dynamically)
        kernels.findBlockBoundsKernel->addArg(context.getPeriodicBoxSize());
        kernels.findBlockBoundsKernel->addArg(context.getInvPeriodicBoxSize());
        kernels.findBlockBoundsKernel->addArg(context.getPeriodicBoxVecX());
        kernels.findBlockBoundsKernel->addArg(context.getPeriodicBoxVecY());
        kernels.findBlockBoundsKernel->addArg(context.getPeriodicBoxVecZ());
        kernels.findBlockBoundsKernel->addArg(context.getPosq());
        kernels.findBlockBoundsKernel->addArg(blockCenter);
        kernels.findBlockBoundsKernel->addArg(blockBoundingBox);
        kernels.findBlockBoundsKernel->addArg(rebuildNeighborList);
        kernels.findBlockBoundsKernel->addArg(sortedBlocks);

        kernels.sortBoxDataKernel = interactingBlocksProgram->createKernel("sortBoxData");
        kernels.sortBoxDataKernel->addArg(sortedBlocks);
        kernels.sortBoxDataKernel->addArg(blockCenter);
        kernels.sortBoxDataKernel->addArg(blockBoundingBox);
        kernels.sortBoxDataKernel->addArg(sortedBlockCenter);
        kernels.sortBoxDataKernel->addArg(sortedBlockBoundingBox);
        kernels.sortBoxDataKernel->addArg(context.getPosq());
        kernels.sortBoxDataKernel->addArg(oldPositions);
        kernels.sortBoxDataKernel->addArg(interactionCount);
        kernels.sortBoxDataKernel->addArg(rebuildNeighborList);
        kernels.sortBoxDataKernel->addArg((int)true);
        if (useLargeBlocks) {
            kernels.sortBoxDataKernel->addArg(largeBlockCenter);
            kernels.sortBoxDataKernel->addArg(largeBlockBoundingBox);
        }

        kernels.findInteractingBlocksKernel = interactingBlocksProgram->createKernel("findBlocksWithInteractions");
        // args 0-4 are periodic box (set dynamically)
        kernels.findInteractingBlocksKernel->addArg(context.getPeriodicBoxSize());
        kernels.findInteractingBlocksKernel->addArg(context.getInvPeriodicBoxSize());
        kernels.findInteractingBlocksKernel->addArg(context.getPeriodicBoxVecX());
        kernels.findInteractingBlocksKernel->addArg(context.getPeriodicBoxVecY());
        kernels.findInteractingBlocksKernel->addArg(context.getPeriodicBoxVecZ());
        kernels.findInteractingBlocksKernel->addArg(interactionCount);
        kernels.findInteractingBlocksKernel->addArg(interactingTiles);
        kernels.findInteractingBlocksKernel->addArg(interactingAtoms);
        kernels.findInteractingBlocksKernel->addArg(context.getPosq());
        kernels.findInteractingBlocksKernel->addArg((unsigned int)interactingTiles.getSize());
        kernels.findInteractingBlocksKernel->addArg((unsigned int)startBlockIndex);
        kernels.findInteractingBlocksKernel->addArg((unsigned int)numBlocks);
        kernels.findInteractingBlocksKernel->addArg(sortedBlocks);
        kernels.findInteractingBlocksKernel->addArg(sortedBlockCenter);
        kernels.findInteractingBlocksKernel->addArg(sortedBlockBoundingBox);
        kernels.findInteractingBlocksKernel->addArg(exclusionIndices);
        kernels.findInteractingBlocksKernel->addArg(exclusionRowIndices);
        kernels.findInteractingBlocksKernel->addArg(oldPositions);
        kernels.findInteractingBlocksKernel->addArg(rebuildNeighborList);
        if (useLargeBlocks) {
            kernels.findInteractingBlocksKernel->addArg(largeBlockCenter);
            kernels.findInteractingBlocksKernel->addArg(largeBlockBoundingBox);
        }
        interactingBlocksThreadBlockSize = groupSize;
    }
    groupKernels[groups] = kernels;
}

ComputeKernel MetalNonbondedUtilities::createInteractionKernel(const string& source, const vector<ParameterInfo>& params, const vector<ParameterInfo>& arguments, bool useExclusions, bool isSymmetric, int groups, bool includeForces, bool includeEnergy) {
    map<string, string> replacements;
    replacements["COMPUTE_INTERACTION"] = source;
    const string suffixes[] = {"x", "y", "z", "w"};
    stringstream localData;
    int localDataSize = 0;
    for (const ParameterInfo& param : params) {
        if (param.getNumComponents() == 1)
            localData<<param.getType()<<" "<<param.getName()<<";\n";
        else {
            for (int j = 0; j < param.getNumComponents(); ++j)
                localData<<param.getComponentType()<<" "<<param.getName()<<"_"<<suffixes[j]<<";\n";
        }
        localDataSize += param.getSize();
    }
    replacements["ATOM_PARAMETER_DATA"] = localData.str();
    stringstream args;
    for (const ParameterInfo& param : params) {
        args << ", __global ";
        if (param.isConstant())
            args << "const ";
        if (param.getNumComponents() == 3)
            args << param.getComponentType();
        else
            args << param.getType();
        args << "* restrict global_";
        args << param.getName();
    }
    for (const ParameterInfo& arg : arguments) {
        args << ", __global ";
        if (arg.isConstant())
            args << "const ";
        args << arg.getType();
        args << "* restrict ";
        args << arg.getName();
    }
    if (energyParameterDerivatives.size() > 0)
        args << ", __global mixed* restrict energyParamDerivs";
    replacements["PARAMETER_ARGUMENTS"] = args.str();
    stringstream loadLocal1;
    for (const ParameterInfo& param : params) {
        if (param.getNumComponents() == 1) {
            loadLocal1<<"localData[localAtomIndex]."<<param.getName()<<" = "<<param.getName()<<"1;\n";
        }
        else {
            for (int j = 0; j < param.getNumComponents(); ++j)
                loadLocal1<<"localData[localAtomIndex]."<<param.getName()<<"_"<<suffixes[j]<<" = "<<param.getName()<<"1."<<suffixes[j]<<";\n";
        }
    }
    replacements["LOAD_LOCAL_PARAMETERS_FROM_1"] = loadLocal1.str();
    replacements["DECLARE_LOCAL_PARAMETERS"] = "";
    stringstream loadLocal2;
    for (const ParameterInfo& param : params) {
        if (param.getNumComponents() == 1) {
            loadLocal2<<"localData[localAtomIndex]."<<param.getName()<<" = global_"<<param.getName()<<"[j];\n";
        }
        else {
            if (param.getNumComponents() == 3)
                loadLocal2<<param.getType()<<" temp_"<<param.getName()<<" = make_"<<param.getType()<<"(global_"<<param.getName()<<"[3*j], global_"<<param.getName()<<"[3*j+1], global_"<<param.getName()<<"[3*j+2]);\n";
            else
                loadLocal2<<param.getType()<<" temp_"<<param.getName()<<" = global_"<<param.getName()<<"[j];\n";
            for (int j = 0; j < param.getNumComponents(); ++j)
                loadLocal2<<"localData[localAtomIndex]."<<param.getName()<<"_"<<suffixes[j]<<" = temp_"<<param.getName()<<"."<<suffixes[j]<<";\n";
        }
    }
    replacements["LOAD_LOCAL_PARAMETERS_FROM_GLOBAL"] = loadLocal2.str();
    stringstream load1;
    for (const ParameterInfo& param : params) {
        load1<<param.getType()<<" "<<param.getName()<<"1 = ";
        if (param.getNumComponents() == 3)
            load1<<"make_"<<param.getType()<<"(global_"<<param.getName()<<"[3*atom1], global_"<<param.getName()<<"[3*atom1+1], global_"<<param.getName()<<"[3*atom1+2]);\n";
        else
            load1<<"global_"<<param.getName()<<"[atom1];\n";
    }
    replacements["LOAD_ATOM1_PARAMETERS"] = load1.str();
    stringstream load2j;
    for (const ParameterInfo& param : params) {
        if (param.getNumComponents() == 1) {
            load2j<<param.getType()<<" "<<param.getName()<<"2 = localData[atom2]."<<param.getName()<<";\n";
        }
        else {
            load2j<<param.getType()<<" "<<param.getName()<<"2 = ("<<param.getType()<<") (";
            for (int j = 0; j < param.getNumComponents(); ++j) {
                if (j > 0)
                    load2j<<", ";
                load2j<<"localData[atom2]."<<param.getName()<<"_"<<suffixes[j];
            }
            load2j<<");\n";
        }
    }
    replacements["LOAD_ATOM2_PARAMETERS"] = load2j.str();
    stringstream clearLocal;
    for (const ParameterInfo& param : params) {
        if (param.getNumComponents() == 1)
            clearLocal<<"localData[localAtomIndex]."<<param.getName()<<" = 0;\n";
        else
            for (int j = 0; j < param.getNumComponents(); ++j)
                clearLocal<<"localData[localAtomIndex]."<<param.getName()<<"_"<<suffixes[j]<<" = 0;\n";
    }
    replacements["CLEAR_LOCAL_PARAMETERS"] = clearLocal.str();
    stringstream initDerivs;
    for (int i = 0; i < energyParameterDerivatives.size(); i++)
        initDerivs<<"mixed energyParamDeriv"<<i<<" = 0;\n";
    replacements["INIT_DERIVATIVES"] = initDerivs.str();
    stringstream saveDerivs;
    const vector<string>& allParamDerivNames = context.getEnergyParamDerivNames();
    int numDerivs = allParamDerivNames.size();
    for (int i = 0; i < energyParameterDerivatives.size(); i++)
        for (int index = 0; index < numDerivs; index++)
            if (allParamDerivNames[index] == energyParameterDerivatives[i])
                saveDerivs<<"energyParamDerivs[GLOBAL_ID*"<<numDerivs<<"+"<<index<<"] += energyParamDeriv"<<i<<";\n";
    replacements["SAVE_DERIVATIVES"] = saveDerivs.str();
    map<string, string> defines;
    if (useCutoff)
        defines["USE_CUTOFF"] = "1";
    if (usePeriodic)
        defines["USE_PERIODIC"] = "1";
    if (useExclusions)
        defines["USE_EXCLUSIONS"] = "1";
    if (isSymmetric)
        defines["USE_SYMMETRIC"] = "1";
    if (useNeighborList)
        defines["USE_NEIGHBOR_LIST"] = "1";
    if (useCutoff && context.getSIMDWidth() < 32)
        defines["PRUNE_BY_CUTOFF"] = "1";
    if (includeForces)
        defines["INCLUDE_FORCES"] = "1";
    if (includeEnergy)
        defines["INCLUDE_ENERGY"] = "1";
    if (useNeighborList && context.getSIMDWidth() >= 32)
        defines["SYNC_WARPS"] = "simdgroup_barrier(metal::mem_flags::mem_threadgroup)";
    defines["THREAD_BLOCK_SIZE"] = context.intToString(forceThreadBlockSize);
    defines["FORCE_WORK_GROUP_SIZE"] = context.intToString(forceThreadBlockSize);
    double maxCutoff = 0.0;
    for (int i = 0; i < 32; i++) {
        if ((groups&(1<<i)) != 0) {
            double cutoff = groupCutoff[i];
            maxCutoff = max(maxCutoff, cutoff);
            defines["CUTOFF_"+context.intToString(i)+"_SQUARED"] = context.doubleToString(cutoff*cutoff);
            defines["CUTOFF_"+context.intToString(i)] = context.doubleToString(cutoff);
        }
    }
    defines["MAX_CUTOFF"] = context.doubleToString(maxCutoff);
    defines["NUM_ATOMS"] = context.intToString(context.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = context.intToString(context.getPaddedNumAtoms());
    defines["NUM_BLOCKS"] = context.intToString(context.getNumAtomBlocks());
    defines["TILE_SIZE"] = context.intToString(MetalContext::TileSize);
    int numExclusionTiles = exclusionTiles.getSize();
    defines["NUM_TILES_WITH_EXCLUSIONS"] = context.intToString(numExclusionTiles);
    int numContexts = context.getPlatformData().contexts.size();
    int startExclusionIndex = context.getContextIndex()*numExclusionTiles/numContexts;
    int endExclusionIndex = (context.getContextIndex()+1)*numExclusionTiles/numContexts;
    defines["FIRST_EXCLUSION_TILE"] = context.intToString(startExclusionIndex);
    defines["LAST_EXCLUSION_TILE"] = context.intToString(endExclusionIndex);
    if ((localDataSize/4)%2 == 0)
        defines["PARAMETER_SIZE_IS_EVEN"] = "1";
    ComputeProgram program;
    try {
        program = context.compileProgram(context.replaceStrings(kernelSource, replacements), defines);
    } catch (OpenMMException& e) {
        fprintf(stderr, "[Metal] Nonbonded kernel compilation failed for group %d "
                "(useExclusions=%d, isSymmetric=%d, forces=%d, energy=%d, params=%d, args=%d)\n",
                groups, useExclusions, isSymmetric, includeForces, includeEnergy,
                (int)params.size(), (int)arguments.size());
        throw;
    }
    ComputeKernel kernel = program->createKernel("computeNonbonded");

    // Set arguments to the Kernel.

    kernel->addArg(context.getAtomicForceBuffer());
    kernel->addArg(context.getEnergyBuffer());
    kernel->addArg(context.getPosq());
    kernel->addArg(exclusions);
    kernel->addArg(exclusionTiles);
    kernel->addArg((unsigned int)startTileIndex);
    kernel->addArg((unsigned long long)numTiles);
    if (useCutoff) {
        kernel->addArg(interactingTiles);
        kernel->addArg(interactionCount);
        // args 9-13: periodic box (set dynamically)
        kernel->addArg(context.getPeriodicBoxSize());
        kernel->addArg(context.getInvPeriodicBoxSize());
        kernel->addArg(context.getPeriodicBoxVecX());
        kernel->addArg(context.getPeriodicBoxVecY());
        kernel->addArg(context.getPeriodicBoxVecZ());
        kernel->addArg((unsigned int)interactingTiles.getSize());
        kernel->addArg(blockCenter);
        kernel->addArg(blockBoundingBox);
        kernel->addArg(interactingAtoms);
    }
    for (const ParameterInfo& param : params) {
        if (param.getMetalArray() == nullptr)
            kernel->addArg(param.getBuffer());
        else
            kernel->addArg(*param.getMetalArray());
    }
    for (const ParameterInfo& arg : arguments) {
        if (arg.getMetalArray() == nullptr)
            kernel->addArg(arg.getBuffer());
        else
            kernel->addArg(*arg.getMetalArray());
    }
    if (energyParameterDerivatives.size() > 0)
        kernel->addArg(context.getEnergyParamDerivBuffer());
    return kernel;
}

void MetalNonbondedUtilities::setKernelSource(const string& source) {
    kernelSource = source;
}
