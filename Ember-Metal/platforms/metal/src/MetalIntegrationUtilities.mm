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
#include "MetalIntegrationUtilities.h"
#include "MetalContext.h"

using namespace OpenMM;
using namespace std;

MetalIntegrationUtilities::MetalIntegrationUtilities(MetalContext& context, const System& system) : IntegrationUtilities(context, system) {
    // Metal on Apple Silicon always uses shared memory, so no special pinned buffer needed.
    // The ccmaUseDirectBuffer flag is not relevant for Metal -- we always use direct buffer access.
    ccmaUseDirectBuffer = false;
}

MetalArray& MetalIntegrationUtilities::getPosDelta() {
    return dynamic_cast<MetalContext&>(context).unwrap(posDelta);
}

MetalArray& MetalIntegrationUtilities::getRandom() {
    return dynamic_cast<MetalContext&>(context).unwrap(random);
}

MetalArray& MetalIntegrationUtilities::getStepSize() {
    return dynamic_cast<MetalContext&>(context).unwrap(stepSize);
}

void MetalIntegrationUtilities::applyConstraintsImpl(bool constrainVelocities, double tol) {
    ComputeKernel settleKernel, shakeKernel, ccmaForceKernel;
    if (constrainVelocities) {
        settleKernel = settleVelKernel;
        shakeKernel = shakeVelKernel;
        ccmaForceKernel = ccmaVelForceKernel;
    }
    else {
        settleKernel = settlePosKernel;
        shakeKernel = shakePosKernel;
        ccmaForceKernel = ccmaPosForceKernel;
    }
    if (settleAtoms.isInitialized()) {
        if (context.getUseDoublePrecision() || context.getUseMixedPrecision())
            settleKernel->setArg(1, tol);
        else
            settleKernel->setArg(1, (float) tol);
        settleKernel->execute(settleAtoms.getSize());
    }
    if (shakeAtoms.isInitialized()) {
        if (context.getUseDoublePrecision() || context.getUseMixedPrecision())
            shakeKernel->setArg(1, tol);
        else
            shakeKernel->setArg(1, (float) tol);
        shakeKernel->execute(shakeAtoms.getSize());
    }
    if (ccmaConstraintAtoms.isInitialized()) {
        if (ccmaConstraintAtoms.getSize() <= 1024) {
            // Use the version of CCMA that runs in a single kernel with one workgroup.
            ccmaFullKernel->setArg(0, (int) constrainVelocities);
            if (context.getUseDoublePrecision() || context.getUseMixedPrecision())
                ccmaFullKernel->setArg(14, tol);
            else
                ccmaFullKernel->setArg(14, (float) tol);
            ccmaFullKernel->execute(128, 128);
        }
        else {
            // Use the version of CCMA that uses multiple kernels.
            if (context.getUseDoublePrecision() || context.getUseMixedPrecision())
                ccmaForceKernel->setArg(7, tol);
            else
                ccmaForceKernel->setArg(7, (float) tol);
            ccmaDirectionsKernel->execute(ccmaConstraintAtoms.getSize());
            const int checkInterval = 4;
            MetalContext& cl = dynamic_cast<MetalContext&>(context);
            int* converged = (int*) context.getPinnedBuffer();
            converged[0] = 0;
            ccmaUpdateKernel->setArg(4, constrainVelocities ? context.getVelm() : posDelta);
            for (int i = 0; i < 150; i++) {
                ccmaForceKernel->setArg(8, i);
                ccmaForceKernel->execute(ccmaConstraintAtoms.getSize());
                ccmaMultiplyKernel->setArg(5, i);
                ccmaMultiplyKernel->execute(ccmaConstraintAtoms.getSize());
                ccmaUpdateKernel->setArg(9, i);
                ccmaUpdateKernel->execute(context.getNumAtoms());
                if ((i+1)%checkInterval == 0) {
                    // In Metal, kernel execution is synchronous via waitUntilCompleted,
                    // so the converged buffer is already available after execution.
                    cl.unwrap(ccmaConverged).download(converged);
                    if (converged[i%2])
                        break;
                }
            }
        }
    }
}

void MetalIntegrationUtilities::distributeForcesFromVirtualSites() {
    if (numVsites > 0) {
        vsiteForceKernel->setArg(2, context.getLongForceBuffer());
        vsiteForceKernel->execute(numVsites);
        vsiteSaveForcesKernel->setArg(0, context.getLongForceBuffer());
        vsiteSaveForcesKernel->setArg(1, context.getForceBuffers());
        vsiteSaveForcesKernel->execute(context.getNumAtoms());
   }
}
