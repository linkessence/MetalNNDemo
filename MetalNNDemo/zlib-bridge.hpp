//
//  zlib-bridge.hpp
//  MetalNNDemo
//
//  Created by Tao Zhang on 2021/5/11.
//

#pragma once

#include <cstdint>
#include <cstdio>
#include <zlib.h>

//
// Bridge of the zlib and Swift, to decompress the data set files.
//
class ZlibBridge {
 public:
    ZlibBridge();
    
    ssize_t decompressGzipFile(FILE* fp, uint8_t** data);
};

