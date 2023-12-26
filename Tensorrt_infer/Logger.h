//
// Created by ubuntu on 23-12-1.
//

#pragma once
#include <NvInfer.h>
#include <iostream>

using namespace nvinfer1;
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};