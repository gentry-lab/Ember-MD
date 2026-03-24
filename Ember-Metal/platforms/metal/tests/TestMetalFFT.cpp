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
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

/**
 * This tests the Metal implementation of FFT.
 */

#include "openmm/internal/AssertionUtilities.h"
#include "MetalArray.h"
#include "MetalContext.h"
#include "MetalFFT3D.h"
#include "MetalSort.h"
#include "sfmt/SFMT.h"
#include "openmm/System.h"
#include <complex>
#include <iostream>
#include <cmath>
#include <set>
#ifdef _MSC_VER
  #define POCKETFFT_NO_VECTORS
#endif
#include "pocketfft_hdronly.h"

using namespace OpenMM;
using namespace std;

static MetalPlatform platform;

template <class Real2>
void testTransform(bool realToComplex, int xsize, int ysize, int zsize) {
    System system;
    system.addParticle(0.0);
    MetalPlatform::PlatformData platformData(system, "", "", platform.getPropertyDefaultValue("MetalPrecision"), "false", "false", 1, NULL);
    MetalContext& context = *platformData.contexts[0];
    context.initialize();
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    vector<Real2> original(xsize*ysize*zsize);
    vector<complex<double> > reference(original.size());
    for (int i = 0; i < (int) original.size(); i++) {
        Real2 value = Real2((float) genrand_real2(sfmt), (float) genrand_real2(sfmt));
        original[i] = value;
        reference[i] = complex<double>(value.x, value.y);
    }
    for (int i = 0; i < (int) reference.size(); i++) {
        if (realToComplex)
            reference[i] = complex<double>(i%2 == 0 ? original[i/2].x : original[i/2].y, 0);
        else
            reference[i] = complex<double>(original[i].x, original[i].y);
    }
    MetalArray grid1(context, original.size(), sizeof(Real2), "grid1");
    MetalArray grid2(context, original.size(), sizeof(Real2), "grid2");
    grid1.upload(original);
    MetalFFT3D fft(context, xsize, ysize, zsize, realToComplex);

    // Perform a forward FFT, then verify the result is correct.

    fft.execFFT(grid1, grid2, true);
    vector<Real2> result;
    grid2.download(result);
    vector<size_t> shape = {(size_t) xsize, (size_t) ysize, (size_t) zsize};
    vector<size_t> axes = {0, 1, 2};
    vector<ptrdiff_t> stride = {(ptrdiff_t) (ysize*zsize*sizeof(complex<double>)),
                                (ptrdiff_t) (zsize*sizeof(complex<double>)),
                                (ptrdiff_t) sizeof(complex<double>)};
    pocketfft::c2c(shape, stride, stride, axes, true, reference.data(), reference.data(), 1.0);
    int outputZSize = (realToComplex ? zsize/2+1 : zsize);
    for (int x = 0; x < xsize; x++)
        for (int y = 0; y < ysize; y++)
            for (int z = 0; z < outputZSize; z++) {
                int index1 = x*ysize*zsize + y*zsize + z;
                int index2 = x*ysize*outputZSize + y*outputZSize + z;
                ASSERT_EQUAL_TOL(reference[index1].real(), result[index2].x, 1e-3);
                ASSERT_EQUAL_TOL(reference[index1].imag(), result[index2].y, 1e-3);
            }

    // Perform a backward transform and see if we get the original values.

    fft.execFFT(grid2, grid1, false);
    grid1.download(result);
    double scale = 1.0/(xsize*ysize*zsize);
    int valuesToCheck = (realToComplex ? original.size()/2 : original.size());
    for (int i = 0; i < valuesToCheck; ++i) {
        ASSERT_EQUAL_TOL(original[i].x, scale*result[i].x, 1e-4);
        ASSERT_EQUAL_TOL(original[i].y, scale*result[i].y, 1e-4);
    }
}

void testMinimalFFT() {
    System system;
    system.addParticle(0.0);
    MetalPlatform::PlatformData platformData(system, "", "", "single", "false", "false", 1, NULL);
    MetalContext& context = *platformData.contexts[0];
    context.initialize();

    // Test 1: 1x1x2 (simplest radix-2)
    {
        vector<mm_float2> input = {{3,0},{7,0}};
        MetalArray g1(context, 2, sizeof(mm_float2), "g1");
        MetalArray g2(context, 2, sizeof(mm_float2), "g2");
        g1.upload(input);
        MetalFFT3D fft(context, 1, 1, 2, false);
        fft.execFFT(g1, g2, true);
        vector<mm_float2> out(2);
        g2.download(out);
        printf("1x1x2: X[0]=(%g,%g) X[1]=(%g,%g)\n", out[0].x, out[0].y, out[1].x, out[1].y);
        // Expected: (10,0), (-4,0)
        ASSERT_EQUAL_TOL(10.0, out[0].x, 1e-3);
        ASSERT_EQUAL_TOL(-4.0, out[1].x, 1e-3);
    }

    // Test 2: 1x1x4 (radix-4) - check both grids after execFFT
    {
        vector<mm_float2> input = {{1,0},{2,0},{3,0},{4,0}};
        MetalArray g1(context, 4, sizeof(mm_float2), "g1");
        MetalArray g2(context, 4, sizeof(mm_float2), "g2");
        g1.upload(input);
        MetalFFT3D fft(context, 1, 1, 4, false);
        fft.execFFT(g1, g2, true);
        vector<mm_float2> out1(4), out2(4);
        g1.download(out1);
        g2.download(out2);
        printf("1x1x4 g1 (after stage2→g1): ");
        for (int i = 0; i < 4; i++) printf("(%g,%g) ", out1[i].x, out1[i].y);
        printf("\n");
        printf("1x1x4 g2 (final, after stage3→g2): ");
        for (int i = 0; i < 4; i++) printf("(%g,%g) ", out2[i].x, out2[i].y);
        printf("\n");
    }

    // Test 3: 2x1x2 (tests ping-pong between stages)
    {
        // Input: row0=[1,2], row1=[3,4]. After z-FFT: row0=[3,-1], row1=[7,-1]
        // After x-FFT: row0=[10,-2], row1=[-4,0]
        vector<mm_float2> input = {{1,0},{2,0},{3,0},{4,0}};
        MetalArray g1(context, 4, sizeof(mm_float2), "g1");
        MetalArray g2(context, 4, sizeof(mm_float2), "g2");
        g1.upload(input);
        MetalFFT3D fft(context, 2, 1, 2, false);
        fft.execFFT(g1, g2, true);
        vector<mm_float2> out(4);
        g2.download(out);
        printf("2x1x2: ");
        for (int i = 0; i < 4; i++) printf("X[%d]=(%g,%g) ", i, out[i].x, out[i].y);
        printf("\n");
    }

    // Test 4: 1x1x3 (radix-3) - DFT of [1,2,3] = [6, -1.5+0.866i, -1.5-0.866i]
    {
        vector<mm_float2> input = {{1,0},{2,0},{3,0}};
        MetalArray g1(context, 3, sizeof(mm_float2), "g1");
        MetalArray g2(context, 3, sizeof(mm_float2), "g2");
        g1.upload(input);
        MetalFFT3D fft(context, 1, 1, 3, false);
        fft.execFFT(g1, g2, true);
        vector<mm_float2> out(3);
        g2.download(out);
        printf("1x1x3: ");
        for (int i = 0; i < 3; i++) printf("X[%d]=(%g,%g) ", i, out[i].x, out[i].y);
        printf("\n");
        ASSERT_EQUAL_TOL(6.0, out[0].x, 1e-3);
        ASSERT_EQUAL_TOL(-1.5, out[1].x, 1e-3);
        ASSERT_EQUAL_TOL(0.866025, out[1].y, 1e-3);
    }

    // Test 5: 1x1x5 (radix-5)
    {
        vector<mm_float2> input = {{1,0},{2,0},{3,0},{4,0},{5,0}};
        MetalArray g1(context, 5, sizeof(mm_float2), "g1");
        MetalArray g2(context, 5, sizeof(mm_float2), "g2");
        g1.upload(input);
        MetalFFT3D fft(context, 1, 1, 5, false);
        fft.execFFT(g1, g2, true);
        vector<mm_float2> out(5);
        g2.download(out);
        printf("1x1x5: ");
        for (int i = 0; i < 5; i++) printf("X[%d]=(%g,%g) ", i, out[i].x, out[i].y);
        printf("\n");
    }

    // Test 6: 1x1x6 (radix-3 then radix-2, TWO passes)
    {
        vector<mm_float2> input = {{1,0},{2,0},{3,0},{4,0},{5,0},{6,0}};
        MetalArray g1(context, 6, sizeof(mm_float2), "g1");
        MetalArray g2(context, 6, sizeof(mm_float2), "g2");
        g1.upload(input);
        MetalFFT3D fft(context, 1, 1, 6, false);
        fft.execFFT(g1, g2, true);
        vector<mm_float2> out(6);
        g2.download(out);
        printf("1x1x6: ");
        for (int i = 0; i < 6; i++) printf("X[%d]=(%g,%g) ", i, out[i].x, out[i].y);
        printf("\n");
        ASSERT_EQUAL_TOL(21.0, out[0].x, 1e-3);
    }

    printf("Minimal FFT tests done\n");
}

int main(int argc, char* argv[]) {
    try {
        if (argc > 1)
            platform.setPropertyDefaultValue("MetalPrecision", string(argv[1]));
        if (platform.getPropertyDefaultValue("MetalPrecision") == "double") {
            testTransform<mm_double2>(false, 28, 25, 30);
            testTransform<mm_double2>(true, 28, 25, 25);
            testTransform<mm_double2>(true, 25, 28, 25);
            testTransform<mm_double2>(true, 25, 25, 28);
            testTransform<mm_double2>(true, 21, 25, 27);
        }
        else {
            testTransform<mm_float2>(false, 28, 25, 30);
            testTransform<mm_float2>(true, 28, 25, 25);
            testTransform<mm_float2>(true, 25, 28, 25);
            testTransform<mm_float2>(true, 25, 25, 28);
            testTransform<mm_float2>(true, 21, 25, 27);
        }
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
