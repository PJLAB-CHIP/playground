#!/bin/bash
export CC="gcc"
export CXX="g++"
export NVCC_CCBIN="gcc"


TEST_MATMUL_VERSION=1
TEST_DATA_TYPE="float32"

SOURCE_DIR=./task-1
BUILD_DIR=./build
BUILD_TYPE=Release
CXX_STANDARD=20
CUDA_STANDARD=20
VCPKG_HOME=$VCPKG_HOME

if [ -t 1 ]; then
    STDOUT_IS_TERMINAL=ON
else
    STDOUT_IS_TERMINAL=OFF
fi


# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v*)
            TEST_MATMUL_VERSION="${1#*v}" ;;
        -f32|--float32)
            TEST_DATA_TYPE="float32" ;;
        -f16|--float16)
            TEST_DATA_TYPE="float16" ;;
        -S|--source-dir)
            SOURCE_DIR=$2; shift ;;
        -B|--build-dir)
            BUILD_DIR=$2; shift ;;
        Release|Debug|RelWithDebInfo|RD)
            BUILD_TYPE=${1/RD/RelWithDebInfo} ;;
        --stdc++=*)
            CXX_STANDARD="${1#*=}" ;;
        --stdcuda=*)
            CUDA_STANDARD="${1#*=}" ;;
        --rm-build-dir)
            rm -rf $BUILD_DIR ;;
        --vcpkg-home|--vcpkg-dir|--vcpkg-root)
            VCPKG_HOME=$2; shift ;;
        *)
            echo "build fatal: Invalid argument '$1'."; exit 1 ;;
    esac
    shift
done

cmake -S $SOURCE_DIR -B $BUILD_DIR -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$VCPKG_HOME/scripts/buildsystems/vcpkg.cmake" \
    -DMATMUL_VERSION=$TEST_MATMUL_VERSION \
    -DTEST_DATA_TYPE=$TEST_DATA_TYPE \
    -DSTDOUT_IS_TERMINAL=$STDOUT_IS_TERMINAL \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CXX_STANDARD=$CXX_STANDARD \
    -DCMAKE_CUDA_STANDARD=$CUDA_STANDARD 

cmake --build $BUILD_DIR --parallel 12 

