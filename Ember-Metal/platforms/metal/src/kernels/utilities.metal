/**
 * Fill a buffer with 0.
 * (Helper function — not a kernel. Called by clearTwoBuffers etc.)
 */

void clearBuffer_impl(device int* restrict buffer, int size, MM_THREAD_PARAMS) {
    int index = GLOBAL_ID;
    device int4* buffer4 = (device int4*) buffer;
    int sizeDiv4 = size/4;
    while (index < sizeDiv4) {
        buffer4[index] = int4(0);
        index += GLOBAL_SIZE;
    }
    if (GLOBAL_ID == 0)
        for (int i = sizeDiv4*4; i < size; i++)
            buffer[i] = 0;
}

/**
 * Fill a buffer with 0 (kernel version — dispatched directly from host).
 */
KERNEL void clearBuffer(device int* restrict buffer, int size,
        MM_THREAD_ARGS) {
    clearBuffer_impl(buffer, size, MM_THREAD_PASS);
}

/**
 * Fill two buffers with 0.
 */
KERNEL void clearTwoBuffers(device int* restrict buffer1, int size1, device int* restrict buffer2, int size2,
        MM_THREAD_ARGS) {
    clearBuffer_impl(buffer1, size1, MM_THREAD_PASS);
    clearBuffer_impl(buffer2, size2, MM_THREAD_PASS);
}

/**
 * Fill three buffers with 0.
 */
KERNEL void clearThreeBuffers(device int* restrict buffer1, int size1, device int* restrict buffer2, int size2, device int* restrict buffer3, int size3,
        MM_THREAD_ARGS) {
    clearBuffer_impl(buffer1, size1, MM_THREAD_PASS);
    clearBuffer_impl(buffer2, size2, MM_THREAD_PASS);
    clearBuffer_impl(buffer3, size3, MM_THREAD_PASS);
}

/**
 * Fill four buffers with 0.
 */
KERNEL void clearFourBuffers(device int* restrict buffer1, int size1, device int* restrict buffer2, int size2, device int* restrict buffer3, int size3, device int* restrict buffer4, int size4,
        MM_THREAD_ARGS) {
    clearBuffer_impl(buffer1, size1, MM_THREAD_PASS);
    clearBuffer_impl(buffer2, size2, MM_THREAD_PASS);
    clearBuffer_impl(buffer3, size3, MM_THREAD_PASS);
    clearBuffer_impl(buffer4, size4, MM_THREAD_PASS);
}

/**
 * Fill five buffers with 0.
 */
KERNEL void clearFiveBuffers(device int* restrict buffer1, int size1, device int* restrict buffer2, int size2, device int* restrict buffer3, int size3, device int* restrict buffer4, int size4, device int* restrict buffer5, int size5,
        MM_THREAD_ARGS) {
    clearBuffer_impl(buffer1, size1, MM_THREAD_PASS);
    clearBuffer_impl(buffer2, size2, MM_THREAD_PASS);
    clearBuffer_impl(buffer3, size3, MM_THREAD_PASS);
    clearBuffer_impl(buffer4, size4, MM_THREAD_PASS);
    clearBuffer_impl(buffer5, size5, MM_THREAD_PASS);
}

/**
 * Fill six buffers with 0.
 */
KERNEL void clearSixBuffers(device int* restrict buffer1, int size1, device int* restrict buffer2, int size2, device int* restrict buffer3, int size3, device int* restrict buffer4, int size4, device int* restrict buffer5, int size5, device int* restrict buffer6, int size6,
        MM_THREAD_ARGS) {
    clearBuffer_impl(buffer1, size1, MM_THREAD_PASS);
    clearBuffer_impl(buffer2, size2, MM_THREAD_PASS);
    clearBuffer_impl(buffer3, size3, MM_THREAD_PASS);
    clearBuffer_impl(buffer4, size4, MM_THREAD_PASS);
    clearBuffer_impl(buffer5, size5, MM_THREAD_PASS);
    clearBuffer_impl(buffer6, size6, MM_THREAD_PASS);
}

/**
 * Sum a collection of buffers into the first one.
 * Also, write the result into a 64-bit fixed point buffer (overwriting its contents).
 */

KERNEL void reduceReal4Buffer(GLOBAL real4* restrict buffer, GLOBAL long* restrict longBuffer, int bufferSize, int numBuffers,
        MM_THREAD_ARGS) {
    int index = GLOBAL_ID;
    int totalSize = bufferSize*numBuffers;
    while (index < bufferSize) {
        real4 sum = buffer[index];
        for (int i = index+bufferSize; i < totalSize; i += bufferSize)
            sum += buffer[i];
        buffer[index] = sum;
        longBuffer[index] = (long) (sum.x*0x100000000);
        longBuffer[index+bufferSize] = (long) (sum.y*0x100000000);
        longBuffer[index+2*bufferSize] = (long) (sum.z*0x100000000);
        index += GLOBAL_SIZE;
    }
}

/**
 * Sum the various buffers containing forces.
 */
KERNEL void reduceForces(GLOBAL long* restrict longBuffer, GLOBAL float* restrict floatBuffer,
        GLOBAL real4* restrict buffer, int bufferSize, int numBuffers,
        MM_THREAD_ARGS) {
    int totalSize = bufferSize*numBuffers;
    real scale = 1/(real) 0x100000000;
    for (int index = GLOBAL_ID; index < bufferSize; index += GLOBAL_SIZE) {
        real4 sum = real4(scale*longBuffer[index], scale*longBuffer[index+bufferSize], scale*longBuffer[index+2*bufferSize], 0);
        sum.x += floatBuffer[index];
        sum.y += floatBuffer[index+bufferSize];
        sum.z += floatBuffer[index+2*bufferSize];
        for (int i = index; i < totalSize; i += bufferSize)
            sum += buffer[i];
        buffer[index] = sum;
        longBuffer[index] = realToFixedPoint(sum.x);
        longBuffer[index+bufferSize] = realToFixedPoint(sum.y);
        longBuffer[index+2*bufferSize] = realToFixedPoint(sum.z);
    }
}

/**
 * Sum the energy buffer.
 */
KERNEL void reduceEnergy(GLOBAL const mixed* restrict energyBuffer, GLOBAL mixed* restrict result, int bufferSize, int workGroupSize, LOCAL mixed* tempBuffer,
        MM_THREAD_ARGS) {
    const unsigned int threadIdx = LOCAL_ID;
    mixed sum = 0;
    int bufferSizeFloor = bufferSize - bufferSize % 16;

  #if REDUCE_ENERGY_MULTIPLE_THREADGROUPS
    for (unsigned int index = GLOBAL_ID; index < bufferSizeFloor / 16; index += GLOBAL_SIZE)
  #else
    for (unsigned int index = threadIdx; index < bufferSizeFloor / 16; index += LOCAL_SIZE)
  #endif
    {
      mm_float16 value = ((const device mm_float16*)energyBuffer)[index];
      sum += value[0];
      sum += value[1];
      sum += value[2];
      sum += value[3];
      sum += value[4];
      sum += value[5];
      sum += value[6];
      sum += value[7];
      sum += value[8];
      sum += value[9];
      sum += value[10];
      sum += value[11];
      sum += value[12];
      sum += value[13];
      sum += value[14];
      sum += value[15];
    }

    unsigned int index = bufferSizeFloor + threadIdx;
    if (index < bufferSize) {
      sum += energyBuffer[index];
    }

    // SIMD reduction (native Metal — VENDOR_APPLE always defined)
    sum = sub_group_reduce_add(sum);

    if (threadIdx % 32 == 0) {
      tempBuffer[threadIdx / 32] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (threadIdx < 32) {
      if (threadIdx >= LOCAL_SIZE / 32) {
        sum = 0;
      } else {
        sum = tempBuffer[threadIdx];
      }
      sum = sub_group_reduce_add(sum);
#if REDUCE_ENERGY_MULTIPLE_THREADGROUPS
      result[GROUP_ID] = sum;
#else
      *result = sum;
#endif
    }
}

/**
 * This is called to determine the accuracy of various native functions.
 */

KERNEL void determineNativeAccuracy(GLOBAL mm_float8* restrict values, int numValues,
        MM_THREAD_ARGS) {
    for (int i = GLOBAL_ID; i < numValues; i += GLOBAL_SIZE) {
        float v = values[i].s0;
        mm_float8 result;
        result.s0 = v;
        result.s1 = native_sqrt(v);
        result.s2 = native_rsqrt(v);
        result.s3 = native_recip(v);
        result.s4 = native_exp(v);
        result.s5 = native_log(v);
        result.s6 = 0.0f;
        result.s7 = 0.0f;
        values[i] = result;
    }
}

/**
 * Record the atomic charges into the posq array.
 */
KERNEL void setCharges(GLOBAL real* restrict charges, GLOBAL real4* restrict posq, GLOBAL int* restrict atomOrder, int numAtoms,
        MM_THREAD_ARGS) {
    for (int i = GLOBAL_ID; i < numAtoms; i += GLOBAL_SIZE)
        posq[i].w = charges[atomOrder[i]];
}
