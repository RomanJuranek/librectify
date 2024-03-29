
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_BUILD_TYPE RELEASE)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
     # using Visual Studio C++
     add_definitions(-D_USE_MATH_DEFINES)
endif()

add_compile_options(-ffast-math -msse2 -msse3 -mssse3 -msse4 -msse4.1 -mavx -mfma -mfpmath=sse -O3)
#add_compile_options(-O1)

add_library(rectify SHARED
    filter.cpp geometry.cpp image.cpp interface.cpp
    line_detector.cpp threading.cpp transform.cpp
    line_pencil.cpp math_utils.cpp
)

set_target_properties(rectify PROPERTIES VERSION ${PROJECT_VERSION})
#set_target_properties(rectify PROPERTIES CXX_VISIBILITY_PRESET hidden)
#set_target_properties(rectify PROPERTIES PUBLIC_HEADER librectify.h)
target_compile_definitions(rectify PUBLIC EIGEN_FAST_MATH EIGEN_DONT_PARALLELIZE)
target_include_directories(rectify PUBLIC ${PROJECT_SOURCE_DIR}/include/eigen)

#include(GenerateExportHeader)
#generate_export_header(rectify)

#target_link_directories(rectify PUBLIC ${PROJECT_SOURCE_DIR}/bin)

target_link_libraries(rectify PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)


add_executable(autorectify autorectify.cpp)
target_link_libraries(autorectify rectify ${OpenCV_LIBS})

add_executable(test0 test.cpp)
target_link_libraries(test0 rectify)
