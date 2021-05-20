//
//  Use this file to import your target's public headers that you would like to expose to Swift.
//

#include <stdio.h>
#include <stdint.h>

typedef void (*LOGGER_CALLBACK)(const char* message);

#ifndef Bool
#define Bool                int
#define True                (1)
#define False               (0)
#endif

//
// Decompress a gzip file.
//
// @param fileName:         the file name.
// @param data:             buffer to hold decompressed data.
// @returns                 size of decompressed data, or -1 if any error occurred.
//
// Note:
// Memory will be allocated inside this function.  You must call free() to free the data buffer
// after use.
//
ssize_t C_decompressGzipFile(const char* fileName, uint8_t** data);

//
// Register the logger callback.
// @param callback          pointer to the callback function.
//
void C_registerLoggerCallback(LOGGER_CALLBACK callback);

//
// Descriptor of the image set.
//
struct ImageSetDescriptor {
    size_t count;           ///< number of items
    size_t width;           ///< width of image in pixels
    size_t height;          ///< height of image in pixels
    size_t offset;          ///< offset of the first image in the data chunk
    size_t stride;          ///< size of each image in bytes
};

//
// Descriptor of the label set.
//
struct LabelSetDescriptor {
    size_t count;           ///< number of items
    size_t offset;          ///< offset of the first label in the data chunk
    size_t stride;          ///< size of each label in bytes, should be 1
};

//
// Read image set descriptor.
//
Bool C_readImageSetDescriptor(const uint8_t* data,
                              size_t dataSize,
                              struct ImageSetDescriptor* descriptor);

//
// Read label set descriptor.
//
Bool C_readLabelSetDescriptor(const uint8_t* data,
                              size_t dataSize,
                              struct LabelSetDescriptor* descriptor);

//
// Fill in a float 32 array by the given value.
//
void C_fillFloat32Array(void* buffer, float value, size_t numItems);
