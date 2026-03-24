#ifndef __OPENMM_METALFFT3D_H__
#define __OPENMM_METALFFT3D_H__

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2009-2023 Stanford University and the Authors.      *
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

// Disable VkFFT for now — backend 5 (Metal) requires metal-cpp headers which
// aren't bundled, and backend 3 (OpenCL) requires cl:: objects we removed.
// Use the manual radix-based FFT implementation instead. This is well-tested
// and supports dimensions with factors of 2, 3, 5, and 7.
// #define USE_VKFFT

#include "MetalArray.h"

namespace OpenMM {

/**
 * This class performs three dimensional Fast Fourier Transforms using a
 * native Metal implementation based on the mixed radix algorithm described in
 *
 * Takahashi, D. and Kanada, Y., "High-Performance Radix-2, 3 and 5 Parallel 1-D Complex
 * FFT Algorithms for Distributed-Memory Parallel Computers."  Journal of Supercomputing,
 * 15, 207-228 (2000).
 *
 * The size of each dimension may have no prime factors other than 2, 3, 5, and 7.
 * Call findLegalDimension() to determine the smallest valid size >= a minimum.
 *
 * Note that this class performs an unnormalized transform.  A forward transform
 * followed by an inverse transform multiplies every value by the total number of
 * data points.
 */

class OPENMM_EXPORT_COMMON MetalFFT3D {
public:
    /**
     * Create an MetalFFT3D object for performing transforms of a particular size.
     *
     * @param context the context in which to perform calculations
     * @param xsize   the first dimension of the data sets on which FFTs will be performed
     * @param ysize   the second dimension of the data sets on which FFTs will be performed
     * @param zsize   the third dimension of the data sets on which FFTs will be performed
     * @param realToComplex  if true, a real-to-complex transform will be done.  Otherwise, it is complex-to-complex.
     */
    MetalFFT3D(MetalContext& context, int xsize, int ysize, int zsize, bool realToComplex=false);
    ~MetalFFT3D();
    /**
     * Perform a Fourier transform.  The transform cannot be done in-place: the input and output
     * arrays must be different.  Also, the input array is used as workspace, so its contents
     * are destroyed.  This also means that both arrays must be large enough to hold complex values,
     * even when performing a real-to-complex transform.
     *
     * When performing a real-to-complex transform, the output data is of size xsize*ysize*(zsize/2+1)
     * and contains only the non-redundant elements.
     *
     * @param in       the data to transform, ordered such that in[x*ysize*zsize + y*zsize + z] contains element (x, y, z)
     * @param out      on exit, this contains the transformed data
     * @param forward  true to perform a forward transform, false to perform an inverse transform
     */
    void execFFT(MetalArray& in, MetalArray& out, bool forward = true);
    /**
     * Get the smallest legal size for a dimension of the grid (that is, a size with no prime
     * factors other than 2, 3, 5, and 7).
     *
     * @param minimum   the minimum size the return value must be greater than or equal to
     */
    static int findLegalDimension(int minimum);
private:
    void* createKernelPipeline(int xsize, int ysize, int zsize, int& threads, int axis, bool forward, bool inputIsReal);
    int xsize, ysize, zsize;
    int xthreads, ythreads, zthreads;
    int r2cLocalMemSize;
    bool packRealAsComplex;
    MetalContext& context;
    void* xPipeline;
    void* yPipeline;
    void* zPipeline;
    void* invxPipeline;
    void* invyPipeline;
    void* invzPipeline;
    void* packForwardPipeline;
    void* unpackForwardPipeline;
    void* packBackwardPipeline;
    void* unpackBackwardPipeline;
};

} // namespace OpenMM

#endif // __OPENMM_METALFFT3D_H__
