#!/bin/bash
export CC="gcc-13"
export CXX="g++-13"
export NVCC_CCBIN="gcc-11"

PROJ_HOME=$(pwd)

BuildType="Release"
TestMatmulVersion=1
TestDataType="float32"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        Release|Debug) 
            BuildType=$1 ;;
        -v*|--test-matmul-version=*)
            TestMatmulVersion="${1#*=}" ;;
        -f32|--float32)
            TestDataType="float32" ;;
        -f16|--float16)
            TestDataType="float16" ;;
        *)
            echo "build fatal: Invalid argument '$1'."; exit 1 ;;
    esac
    shift
done

echo  -e "\e[1;32m[@~@] Build Start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\e[0m"

cmake -S ./task-1 -B ./build -G "Ninja" \
    -DCMAKE_BUILD_TYPE=$BuildType \
    -DTEST_MATMUL_VERSION=$TestMatmulVersion \
    -DTEST_DATA_TYPE=$TestDataType

echo -e "\e[1;32m[@v@] Build Finished <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\e[0m"
