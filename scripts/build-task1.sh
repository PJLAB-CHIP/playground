#!/bin/bash

# Note: Change these variables based on your environment
export CC="gcc"
export CXX="g++"
export CUDACXX="nvcc"
export CUDA_CC="gcc-11"
export CUDA_DIR="/usr/local/cuda"

PROJ_HOME=$(pwd)
TaskNo="1"

BuildType="Release"
CleanFirst="false"
CleanAll="false"
TestKernelVersion=1
TestDataType="float32"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        Release|Debug)
            BuildType=$1
            ;;
        -cf|--clean-first)
            CleanFirst="true"
            ;;
        -ca|--clean-all)
            CleanAll="true"
            ;;
        -v*)
            TestKernelVersion="${1#*v}"
            ;;
        --test-kernel-version=*)
            TestKernelVersion="${1#*=}"
            ;;
        -f32|--float32)
            TestDataType="float32"
            ;;
        -f16|--float16)
            TestDataType="float16"
            ;;
        --help)
            cat ./docs/scripts/build-task1-help.txt
            exit 1
            ;;
        *)
            echo "build fatal: Invalid argument '$1'. Use --help for more information."
            exit 1
            ;;
    esac
    shift
done

echo  -e "\e[1;32m[@~@] Build Start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\e[0m"

if [ "$CleanAll" = "true" ] && [ -d "$PROJ_HOME/build" ]; then
    echo "Cleaning all build files..."
    rm -rf $PROJ_HOME/build
fi

cmake -S . -B ./build -G "Ninja" \
    -DTASK_NO=$TaskNo \
    -DCMAKE_BUILD_TYPE=$BuildType \
    -DTEST_KERNEL_VERSION=$TestKernelVersion \
    -DTEST_DATA_TYPE=$TestDataType

if [ "$CleanFirst" = "true" ]; then
    cmake --build ./build --parallel $(nproc) --clean-first
else
    cmake --build ./build --parallel $(nproc)
fi

echo -e "\e[1;32m[@v@] Build Finished <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\e[0m"
