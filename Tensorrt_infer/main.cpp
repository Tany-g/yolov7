// main.cpp
#pragma once
#include "YOLOv7Inference.h"

int main() {
    const char* engineFilePath = "/home/ubuntu/GITHUG/yolov7/runs/train/exp15/weights/best.engine";
    YOLOv7Inference yoloInference(engineFilePath);

    const char* inputImagePath = "/home/ubuntu/DataSet/SB2/test/images/random_3_front_bmp.rf.2875a7791bb134446d8e3ec5d8334b4b.jpg";
    cv::Mat inputImage = cv::imread(inputImagePath);

    if (inputImage.empty()) {
        std::cerr << "Failed to read input image" << std::endl;
        return EXIT_FAILURE;
    }

    yoloInference.inferImage(inputImage);

    return EXIT_SUCCESS;
}
