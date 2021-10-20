cmake_minimum_required (VERSION 3.1)
project(sum)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

add_executable (sum src/main.cpp include/Scan.h src/Scan.cpp sum.cl)
target_link_libraries (sum ${OpenCL_LIBRARY})

FILE(GLOB CL_FILES *.cl)
foreach(cl_file ${CL_FILES})
    add_custom_command(
            TARGET sum
            POST_BUILD
            COMMAND ${CMAKE_COMMAND}
            ARGS -E copy ${cl_file} ${PROJECT_BINARY_DIR}/.
    )
endforeach(cl_file)