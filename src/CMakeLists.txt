
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_BUILD_TYPE RELEASE)

add_library(rectify SHARED
    filter.cpp geometry.cpp image.cpp liblgroup.cpp
    line_detector.cpp ransac.cpp threading.cpp transform.cpp
)

set_target_properties(rectify PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(rectify PROPERTIES PUBLIC_HEADER liblgroup.h)
target_include_directories(rectify PUBLIC ${PROJECT_SOURCE_DIR}/include/eigen)
target_link_libraries(rectify PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)


add_executable(autorectify autorectify.cpp)
target_link_libraries(autorectify rectify ${OpenCV_LIBS})