cmake_minimum_required(VERSION 3.26)
project(Yolo_Tensor)

set(CMAKE_CXX_STANDARD 17)

# 设置 CUDA 相关路径
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 设置 TensorRT 相关路径
set(TENSORRT_INCLUDE_DIR /home/ubuntu/mySoftware/TensorRT-8.2.4.2/include/)
set(TENSORRT_LIBRARY_DIR /home/ubuntu/mySoftware/TensorRT-8.2.4.2/lib)
include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIBRARY_DIR})

# 设置 OpenCV 相关路径
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 设置其他依赖库路径

# 添加可执行文件
add_executable(Yolo_Tensor main.cpp
        YOLOv7Inference.cpp
        YOLOv7Inference.h
        Logger.cpp
        Logger.h)

# 链接 CUDA、TensorRT、OpenCV 等库
target_link_libraries(Yolo_Tensor PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(Yolo_Tensor PRIVATE nvinfer nvinfer_plugin nvparsers nvonnxparser)  # 根据实际情况添加 TensorRT 的其他库
target_link_libraries(Yolo_Tensor PRIVATE ${OpenCV_LIBS})

# 添加其他依赖库
# ...

# 其他设置
# ...
