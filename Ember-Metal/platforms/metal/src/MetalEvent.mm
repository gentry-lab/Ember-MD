/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2019 Stanford University and the Authors.           *
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
#include "MetalEvent.h"

using namespace OpenMM;

MetalEvent::MetalEvent(MetalContext& context) : context(context), signaled(false) {
}

void MetalEvent::enqueue() {
    @autoreleasepool {
        // In our Metal dispatch model, each kernel execution creates its own
        // command buffer and calls waitUntilCompleted, so all prior work is
        // already complete by the time this is called. We simply record the
        // enqueue point by marking the event as not-yet-signaled, then
        // immediately signal it since all preceding work is synchronous.
        signaled.store(true, std::memory_order_release);
    }
}

void MetalEvent::wait() {
    // With our synchronous dispatch model (command buffer per kernel,
    // waitUntilCompleted after each), all work prior to enqueue() has
    // already completed. This is effectively a no-op, but we spin
    // briefly to respect the memory ordering contract.
    while (!signaled.load(std::memory_order_acquire)) {
        // Should never actually spin since enqueue() sets this immediately.
    }
}
