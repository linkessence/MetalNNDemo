//
//  datasource.cpp
//  MetalNNDemo
//
//  Created by Tao Zhang on 2021/5/19.
//

#include <cstdio>

//
// Fill in a float 32 array by the given value.
//
extern "C" void C_fillFloat32Array(void* buffer, float value, size_t numItems) {
    float* ptr = static_cast<float*>(buffer);
    for (size_t i = 0; i < numItems; ++i) {
        *(ptr++) = value;
    }
}
