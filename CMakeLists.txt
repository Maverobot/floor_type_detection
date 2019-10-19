cmake_minimum_required(VERSION 3.4)
project(floor_type_detection)

set(LIBTORCH_PATH "" CACHE PATH "Path to the decompressed libtorch folder.")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_PATH})
find_package(Torch CONFIG REQUIRED)

add_executable(${PROJECT_NAME} src/${PROJECT_NAME}.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC ${TORCH_LIBRARIES})
