#ifndef OPENMM_OPENCLBONDEDUTILITIES_H_
#define OPENMM_OPENCLBONDEDUTILITIES_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2011-2022 Stanford University and the Authors.      *
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

#include "MetalArray.h"
#include "openmm/System.h"
#include "openmm/common/BondedUtilities.h"
#include "openmm/common/ComputeProgram.h"
#include <string>
#include <vector>

namespace OpenMM {

class MetalContext;

/**
 * This class provides a generic mechanism for evaluating bonded interactions.
 */

class OPENMM_EXPORT_COMMON MetalBondedUtilities : public BondedUtilities {
public:
    MetalBondedUtilities(MetalContext& context);
    void addInteraction(const std::vector<std::vector<int> >& atoms, const std::string& source, int group);
    std::string addArgument(ArrayInterface& data, const std::string& type);
    std::string addEnergyParameterDerivative(const std::string& param);
    void addPrefixCode(const std::string& source);
    void initialize(const System& system);
    void computeInteractions(int groups);
private:
    std::string createForceSource(int forceIndex, int numBonds, int numAtoms, int group, const std::string& computeForce);
    MetalContext& context;
    ComputeKernel kernel;
    std::vector<std::vector<std::vector<int> > > forceAtoms;
    std::vector<int> indexWidth;
    std::vector<std::string> forceSource;
    std::vector<int> forceGroup;
    std::vector<std::string> argTypes;
    std::vector<ArrayInterface*> arguments;
    std::vector<MetalArray> atomIndices;
    std::vector<std::string> prefixCode;
    std::vector<std::string> energyParameterDerivatives;
    int maxBonds, allGroups;
    bool hasInitializedKernels;
};

} // namespace OpenMM

#endif /*OPENMM_OPENCLBONDEDUTILITIES_H_*/
