//
//  logger.cpp
//  MetalNNDemo
//
//  Created by Tao Zhang on 2021/5/12.
//

#include <stdarg.h>
#include <stdio.h>
#include "logger.hpp"

typedef void (*LOGGER_CALLBACK)(const char* message);

static LOGGER_CALLBACK gLoggerCallback = nullptr;

//
// Register the logger callback.
// @param callback          pointer to the callback function.
//
extern "C" void C_registerLoggerCallback(LOGGER_CALLBACK callback) {
    gLoggerCallback = callback;
}

//
// Print a log message.
//
void Logger::log(const char* format, ...) {
    va_list arg_ptr;
    
    va_start(arg_ptr, format);
    auto length = vsnprintf(nullptr, 0, format, arg_ptr) + 1;
    char* buffer = new char[length];
    vsnprintf(buffer, length, format, arg_ptr);
    
    if (nullptr == gLoggerCallback) {
        fprintf(stderr, "%s", buffer);
    } else {
        gLoggerCallback(buffer);
    }
    
    delete[] buffer;
    va_end(arg_ptr);
}
