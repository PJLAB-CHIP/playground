#!/bin/bash
export CC="/usr/bin/gcc"
export CXX="/usr/bin/g++"
export CUDA_CC="/usr/bin/gcc"
export CUDA_HOME="/usr/local/cuda"

BuildType="Release"
CleanFirst="false"
CleanAll="false"
TestKernelVersion="0"
TestDataType="float32"
UseWMMA="false"

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
        --wmma)
            UseWMMA="true"
            ;;
        --help)
            cat ./docs/help-build.txt
            exit 1
            ;;
        *)
            echo "build fatal: Invalid argument '$1'. Use --help for more information."
            exit 1
            ;;
    esac
    shift
done

PROJ_HOME=$(pwd)

if [ "$CleanAll" = "true" ] && [ -d "$PROJ_HOME/build" ]; then
    echo "Cleaning all build files..."
    rm -rf $PROJ_HOME/build
fi

if [ ! -d "$PROJ_HOME/build" ]; then
    mkdir $PROJ_HOME/build
fi

cd $PROJ_HOME/build
cmake ..  \
    -DCMAKE_BUILD_TYPE=$BuildType \
    -DTEST_KERNEL_VERSION=$TestKernelVersion \
    -DTEST_DATA_TYPE=$TestDataType \
    -DUSE_WMMA=$UseWMMA \
    -G "Ninja"

if [ "$CleanFirst" = "true" ]; then
    cmake --build . --parallel $(nproc) --clean-first
else
    cmake --build . --parallel $(nproc)
fi

cd $PROJ_HOME
echo "Build finished."