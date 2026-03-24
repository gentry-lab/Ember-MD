/**
 * Query Metal device properties.
 * Replaces the OpenCL-based device query with native Metal API.
 */

#import <Metal/Metal.h>
#include <iostream>
#include <iomanip>

using namespace std;

int main() {
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices.count == 0) {
            cout << "No Metal devices found." << endl;
            return 1;
        }

        cout << "Metal devices:" << endl << endl;

        for (NSUInteger i = 0; i < devices.count; i++) {
            id<MTLDevice> d = devices[i];

            cout << "MetalDeviceIndex " << i << ": \"" << [[d name] UTF8String] << "\"" << endl;
            cout << "    " << left << setw(32) << "Platform" << " = Apple Metal" << endl;
            cout << "    " << left << setw(32) << "Device Name" << " = " << [[d name] UTF8String] << endl;
            cout << "    " << left << setw(32) << "Low Power" << " = " << (d.isLowPower ? "Yes" : "No") << endl;
            cout << "    " << left << setw(32) << "Removable" << " = " << (d.isRemovable ? "Yes" : "No") << endl;
            cout << "    " << left << setw(32) << "Registry ID" << " = " << d.registryID << endl;
            cout << "    " << left << setw(32) << "Max Threads/Threadgroup" << " = " << d.maxThreadsPerThreadgroup.width << endl;
            cout << "    " << left << setw(32) << "Max Buffer Length" << " = " << d.maxBufferLength / (1024*1024) << " MB" << endl;
            cout << "    " << left << setw(32) << "Max Threadgroup Memory" << " = " << d.maxThreadgroupMemoryLength / 1024 << " KB" << endl;
            cout << "    " << left << setw(32) << "Unified Memory" << " = " << (d.hasUnifiedMemory ? "Yes" : "No") << endl;
            cout << "    " << left << setw(32) << "Recommended Max Working Set" << " = " << d.recommendedMaxWorkingSetSize / (1024*1024) << " MB" << endl;

            // GPU family support
            if ([d supportsFamily:MTLGPUFamilyApple7]) {
                cout << "    " << left << setw(32) << "GPU Family" << " = Apple 7+" << endl;
            } else if ([d supportsFamily:MTLGPUFamilyApple6]) {
                cout << "    " << left << setw(32) << "GPU Family" << " = Apple 6" << endl;
            } else if ([d supportsFamily:MTLGPUFamilyApple5]) {
                cout << "    " << left << setw(32) << "GPU Family" << " = Apple 5" << endl;
            }

            cout << endl;
        }
    }
    return 0;
}
