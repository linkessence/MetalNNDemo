//
//  zlib-bridge.cpp
//  MetalNNDemo
//
//  Created by Tao Zhang on 2021/5/11.
//

#include <strings.h>
#include <cstdlib>
#include "logger.hpp"
#include "zlib-bridge.hpp"

#define COMPRESSED_BUFFER_SIZE              (512 * 1024)
#define DECOMPRESSED_BUFFER_SIZE            (COMPRESSED_BUFFER_SIZE * 2)

// Default constructor
ZlibBridge::ZlibBridge() = default;

//
// Decompress a gzip file.
//
// @param fileName:      the file name.
// @param data:          buffer to hold decompressed data.
// @returns              size of decompressed data, or -1 if any error occurred.
//
// Note:
// Memory will be allocated inside this function.  You must call free() to free the data buffer
// after use.
//
extern "C" ssize_t C_decompressGzipFile(const char* fileName, uint8_t** data) {
    if (nullptr == fileName) {
        Logger::log("Invalid argument: fileName is null.");
        return -1;
    }
    
    if (nullptr == data) {
        Logger::log("Invalid argument: data is null.");
        return -1;
    }
    
    *data = nullptr;
    ZlibBridge bridge;
    FILE* fp = fopen(fileName, "rb");
    
    if (nullptr == fp) {
        Logger::log("Cannot open file: %s", fileName);
        return -1;
    }
    
    ssize_t size = bridge.decompressGzipFile(fp, data);
    fclose(fp);
    return size;
}


ssize_t ZlibBridge::decompressGzipFile(FILE* fp, uint8_t** data) {
    if (nullptr == fp) {
        Logger::log("Invalid argument: fp is null.");
        return -1;
    }
    
    if (nullptr == data) {
        Logger::log("Invalid argument: data is null.");
        return -1;
    }

    // Allocate memory for buffers
    auto* compressedBuffer = static_cast<uint8_t*>(malloc(COMPRESSED_BUFFER_SIZE));
    auto* decompressedBuffer = static_cast<uint8_t*>(malloc(DECOMPRESSED_BUFFER_SIZE));
    size_t decompressedBufferLength = DECOMPRESSED_BUFFER_SIZE;
    
    // Read the first chunk of compressed file
    size_t bytesRead = fread(compressedBuffer, 1, COMPRESSED_BUFFER_SIZE, fp);
    if (bytesRead == 0) {
        // Input file is empty, or cannot be read
        delete[] compressedBuffer;
        delete[] decompressedBuffer;
        *data = nullptr;
        return 0;
    }
    
    // Decompress
    z_stream strm;
    bzero(&strm, sizeof(strm));
    strm.next_in = compressedBuffer;
    
    strm.avail_in = static_cast<uInt>(bytesRead);
    strm.total_out = 0;
    
    if (inflateInit2(&strm, (16+MAX_WBITS)) != Z_OK) {
        free(compressedBuffer);
        free(decompressedBuffer);
        *data = nullptr;
        Logger::log("Failed to initialize the zlib inflator.");
        return -1;
    }
    
    bool done = false;
    
    while(!done) {
        // If the output buffer is too small
        if (strm.total_out >= decompressedBufferLength) {
            // Increase size of output buffer
            decompressedBufferLength += DECOMPRESSED_BUFFER_SIZE;
            decompressedBuffer = static_cast<uint8_t*>(realloc(decompressedBuffer, decompressedBufferLength));
        }
        
        strm.next_out = static_cast<Bytef*>(decompressedBuffer + strm.total_out);
        strm.avail_out = static_cast<uInt>(decompressedBufferLength - strm.total_out);
        
        // Inflate another chunk.
        int err = inflate (&strm, Z_SYNC_FLUSH);
        if (err == Z_STREAM_END) {
            done = true;
        } else if(err != Z_OK)  {
            free(compressedBuffer);
            free(decompressedBuffer);
            *data = nullptr;
            Logger::log("Failed to decompress.");
            return -1;
        } else if(strm.avail_in == 0) {
            // Read next chunk
            bytesRead = fread(compressedBuffer, 1, COMPRESSED_BUFFER_SIZE, fp);
            strm.next_in = compressedBuffer;
            strm.avail_in = static_cast<uInt>(bytesRead);
        }
        printf(".");
    }
    
    if (inflateEnd(&strm) != Z_OK) {
        free(compressedBuffer);
        free(decompressedBuffer);
        *data = nullptr;
        Logger::log("Failed to decompress.");
        return -1;
    }
    
    printf("\n");
    free(compressedBuffer);
    *data = decompressedBuffer;
    
    return strm.total_out;
}
