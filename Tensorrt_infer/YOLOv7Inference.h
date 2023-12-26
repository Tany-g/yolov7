//
// Created by ubuntu on 23-12-1.
//
#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include "Logger.h"
#include <stdio.h>

using namespace nvinfer1;
#define INPUT_W 1280
#define INPUT_H 1280
#define DEVICE 0  // GPU id
#define CONF_THRESH 0.9
#define NMS_THRESH 0.6
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

constexpr int MAX_OUTPUT_BBOX_COUNT = 1000 * 16;
constexpr int CLASS_NUM = 2;
constexpr int LOCATIONS = 4;



struct alignas(float) Detection {
    float bbox[4];
    float conf;
    float prob;
};


class YOLOv7Inference {
public:
    YOLOv7Inference(const char *engineFilePath);

    ~YOLOv7Inference();

    void inferImage(const cv::Mat &inputImage);

private:
    Logger gLogger;
    IExecutionContext *context;
    float *prob;
    const int outputSize = 1 * 102000 * 6;

    float *blobFromImage(const cv::Mat &img);

    cv::Mat staticResize(const cv::Mat &img);

    void doInference(float *input, float *output, const int inputShape);

    void
    postprocess_decode(float *featBlob, float prob_threshold, std::vector<Detection> &objects_map);

    float iou(float lbox[4], float rbox[4]);

    void nms(std::vector<Detection> &objects_map, std::vector<Detection> &res, float nms_thresh = 0.5);
};
