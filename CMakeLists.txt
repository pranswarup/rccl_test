cmake_minimum_required(VERSION 3.16)

project(rccl_test LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add current directory to module path for FindRCCL.cmake
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find required packages
find_package(HIP REQUIRED)
find_package(MPI REQUIRED)

# Print configuration information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
if(HIP_FOUND)
    message(STATUS "HIP found: ${HIP_VERSION}")
endif()
if(MPI_FOUND)
    message(STATUS "MPI found: ${MPI_CXX_VERSION}")
endif()

# Check for RCCL
find_package(RCCL CONFIG REQUIRED HINTS "${CMAKE_PREFIX_PATH}" PATHS "${ROCM_PATH}")
if (RCCL_FOUND)
    message(STATUS "RCCL version : ${RCCL_VERSION}")
    message(STATUS "RCCL include path : ${RCCL_INCLUDE_DIRS}")
    message(STATUS "RCCL libraries : ${RCCL_LIBRARIES}")
endif()

# Create the executable first
add_executable(rccl_test rccl_test.cpp)

# Set target properties
set_target_properties(rccl_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Define HIP platform for AMD GPUs
target_compile_definitions(rccl_test PRIVATE __HIP_PLATFORM_AMD__)

# Link libraries
target_link_libraries(rccl_test
    PRIVATE
        hip::host
        MPI::MPI_CXX
        roc::rccl
)

# Include directories
target_include_directories(rccl_test
    PRIVATE
        ${HIP_INCLUDE_DIRS}
        ${MPI_CXX_INCLUDE_DIRS}
        ${RCCL_INCLUDE_DIRS}
)