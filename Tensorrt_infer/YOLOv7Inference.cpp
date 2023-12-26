//
// Created by ubuntu on 23-12-1.
//
#pragma once

#include "YOLOv7Inference.h"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
const char *INPUT_BLOB_NAME = "images";
const char *OUTPUT_BLOB_NAME = "output";
std::string categories[] = {"solder ball"};

YOLOv7Inference::YOLOv7Inference(const char *engineFilePath) {
    cudaSetDevice(DEVICE);
    std::ifstream file(engineFilePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Unable to open engine file" << std::endl;
        exit(EXIT_FAILURE);
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    char *trtModelStream = new char[size];
    file.read(trtModelStream, size);
    file.close();

    IRuntime *runtime = createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "Failed to create Inference Runtime" << std::endl;
        exit(EXIT_FAILURE);
    }

    context = nullptr;
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size);
    if (engine) {
        context = engine->createExecutionContext();
    } else {
        std::cerr << "Failed to create CUDA engine" << std::endl;
        exit(EXIT_FAILURE);
    }

    delete[] trtModelStream;

    // Allocate memory for output buffer
    prob = new float[outputSize];
}

YOLOv7Inference::~YOLOv7Inference() {
    if (context) {
        context->destroy();
    }
    delete[] prob;
}

float *YOLOv7Inference::blobFromImage(const cv::Mat &img) {
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    float *blob = new float[img.total() * 3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < img_h; h++) {
            for (int w = 0; w < img_w; w++) {
                blob[c * img_w * img_h + h * img_w + w] =
                        (((float) img.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return blob;
}

cv::Mat YOLOv7Inference::staticResize(const cv::Mat &img) {
    float r = std::min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(INPUT_W, INPUT_H, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

void YOLOv7Inference::doInference(float *input, float *output, const int inputShape) {
    const ICudaEngine &engine = context->getEngine();
    assert(engine.getNbBindings() == 2);
    void *buffers[2];


    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);


    CHECK(cudaMalloc(&buffers[inputIndex], inputShape * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], outputSize * sizeof(float)));


    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, inputShape * sizeof(float), cudaMemcpyHostToDevice, stream));
    std::cout << "Input data: ";

    context->enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}


void YOLOv7Inference::postprocess_decode(float *feat_blob, float prob_threshold,
                                         std::vector<Detection> &objects_map) {
    int det_size = sizeof(Detection) / sizeof(float);

    for (int i = 0; i < outputSize / 6; i++) {
        if (feat_blob[det_size * i + 4] <= prob_threshold) continue;
        Detection det;
        memcpy(&det, &feat_blob[det_size * i], det_size * sizeof(float));
        objects_map.push_back(det);
    }
}

float YOLOv7Inference::iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
            (std::max)(lbox[0] - lbox[2] / 2.f, rbox[0] - rbox[2] / 2.f), //left
            (std::min)(lbox[0] + lbox[2] / 2.f, rbox[0] + rbox[2] / 2.f), //right
            (std::max)(lbox[1] - lbox[3] / 2.f, rbox[1] - rbox[3] / 2.f), //top
            (std::min)(lbox[1] + lbox[3] / 2.f, rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

bool cmp(const Detection &a, const Detection &b) {
    return a.conf > b.conf;
}

void YOLOv7Inference::nms(std::vector<Detection> &objects_map, std::vector<Detection> &res,
                          float nms_thresh) {
    std::sort(objects_map.begin(), objects_map.end(), cmp);
    for (unsigned int det_map_i = 0; det_map_i < objects_map.size(); ++det_map_i) {
        auto &item = objects_map[det_map_i];
        res.push_back(item);
        for (unsigned int n = det_map_i + 1; n < objects_map.size(); ++n) {
            if (iou(item.bbox, objects_map[n].bbox) > nms_thresh) {
                objects_map.erase(objects_map.begin() + n);
                --n;
            }
        }
    }
}


void YOLOv7Inference::inferImage(const cv::Mat &inputImage) {
    cv::Mat prImg = staticResize(inputImage);
    float *blob = blobFromImage(prImg);
    doInference(blob, prob, 1 * 3 * INPUT_W * INPUT_W);

    for (int i = 0; i < 30; i++) {
        std::cout << prob[i] << std::endl;
    }
    std::vector<Detection> objects_map;
    postprocess_decode(prob, CONF_THRESH, objects_map);
    std::vector<Detection> objects;
    nms(objects_map, objects, NMS_THRESH);

    std::cout << "NMS之后目标数: " << objects_map.size() << std::endl;

    float r = 0;
    float rW = INPUT_W / (inputImage.cols * 1.0);
    float rH = INPUT_H / (inputImage.rows * 1.0);
    if (inputImage.cols > inputImage.rows) {
        r = rW;
    } else {
        r = rH;
    }
    cv::cvtColor(prImg, prImg, cv::COLOR_RGB2BGR);

//    for (const auto &det: objects_map) {
//        std::cout << "Bbox: ";
//        for (int i = 0; i < LOCATIONS; i++) {
//            std::cout << det.bbox[i] << " ";
//        }
//        std::cout << "Kpt: ";
//        for (int i = 0; i < KEY_POINTS_NUM; i++) {
//            std::cout << "(" << det.kpts[i].x << "," << det.kpts[i].y << ") ";
//        }
//        float half_w = det.bbox[2] / 2;
//        float half_h = det.bbox[3] / 2;
//        cv::Point pt1((det.bbox[0] - half_w) / r, (det.bbox[1] - half_h) / r);
//        cv::Point pt2((det.bbox[0] + half_w) / r, (det.bbox[1] + half_h) / r);
//
//        cv::rectangle(inputImage, pt1, pt2, cv::Scalar(0, 255, 0), 2);
//
//        cv::Point point1(det.kpts[0].x / r, det.kpts[0].y / r);
//        cv::Point point2(det.kpts[1].x / r, det.kpts[1].y / r);
//        cv::Point point3(det.kpts[2].x / r, det.kpts[2].y / r);
//
//        cv::line(inputImage, point1, point2, cv::Scalar(0, 0, 255), 2);
//        cv::line(inputImage, point2, point3, cv::Scalar(255, 0, 0), 2);
//        cv::putText(inputImage, categories[(int) (det.prob[0] > det.prob[1])] + std::to_string(int(det.conf * 100)),
//                    cv::Point((det.bbox[0] - half_w) / r, (det.bbox[1] - half_h)/ r - 3), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 0, 255),
//                    2);
//
//        std::cout << std::endl;
//    }
//
//    cv::imshow("Result", inputImage);
//    cv::waitKey(0);
//    return objects_map;
}