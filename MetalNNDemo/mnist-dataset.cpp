//
//  mnist-dataset.cpp
//  MetalNNDemo
//
//  Created by Tao Zhang on 2021/5/13.
//

#include <arpa/inet.h>
#include <cstdio>
#include <cstdint>
#include "logger.hpp"

extern "C" {
#include "../MetalNNDemo-Bridging-Header.h"
}

#ifndef Bool
#define Bool                int
#define True                (1)
#define False               (0)
#endif

#define LABEL_MAGIC         (0x0801)
#define IMAGE_MAGIC         (0x0803)

//
// Read image set descriptor.
//
extern "C" Bool C_readImageSetDescriptor(const uint8_t* data,
                                         size_t dataSize,
                                         struct ImageSetDescriptor* descriptor) {
    const struct ImageFileHeader {
        uint32_t magic_be;
        uint32_t count_be;
        uint32_t height_be;
        uint32_t width_be;
    } *header;
    
    if (data == nullptr || descriptor == nullptr) {
        Logger::log("Bad argument. null pointers.");
        return False;
    }
    
    if (dataSize < sizeof(ImageFileHeader)) {
        Logger::log("Bad argument. dataSize is too small.");
        return False;
    }
    
    header = reinterpret_cast<const ImageFileHeader*>(data);
    
    if (ntohl(header->magic_be) != IMAGE_MAGIC) {
        Logger::log("Bad magic number in the image file.");
        return false;
    }
    
    descriptor->count = ntohl(header->count_be);
    descriptor->width = ntohl(header->width_be);
    descriptor->height = ntohl(header->height_be);
    descriptor->stride = descriptor->width * descriptor->height;
    descriptor->offset = sizeof(*header);
    
    return True;
}

//
// Read label set descriptor.
//
extern "C" Bool C_readLabelSetDescriptor(const uint8_t* data,
                                         size_t dataSize,
                                         struct LabelSetDescriptor* descriptor) {
    const struct LabelFileHeader {
        uint32_t magic_be;
        uint32_t count_be;
    } *header;
    
    if (data == nullptr || descriptor == nullptr) {
        Logger::log("Bad argument. null pointers.");
        return False;
    }
    
    if (dataSize < sizeof(LabelFileHeader)) {
        Logger::log("Bad argument. dataSize is too small.");
        return False;
    }
    
    header = reinterpret_cast<const LabelFileHeader*>(data);
    
    if (ntohl(header->magic_be) != LABEL_MAGIC) {
        Logger::log("Bad magic number in the label file.");
        return false;
    }
    
    descriptor->count = ntohl(header->count_be);
    descriptor->stride = 1;
    descriptor->offset = sizeof(*header);
    
    return True;
}
