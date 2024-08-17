#!/bin/bash
export CC="/usr/bin/gcc-11"
export CXX="/usr/bin/g++-11"
export CUDA_CC="/usr/bin/gcc-11"
export CUDA_HOME="/usr/local/cuda"

PROJ_HOME=$(pwd)
TaskNo="1"

BuildType="Release"
CleanFirst="false"
CleanAll="false"
TestKernelVersion="0"
TestDataType="float32"
Prefix="$PROJ_HOME/bin/task-1"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prefix=*)
            Prefix="${1#*=}"
            ;;
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


if [ "$CleanAll" = "true" ] && [ -d "$PROJ_HOME/build" ]; then
    echo "Cleaning all build files..."
    rm -rf $PROJ_HOME/build
fi

if [ ! -d "$PROJ_HOME/build" ]; then
    mkdir $PROJ_HOME/build
fi

cd $PROJ_HOME/build
cmake ..  \
    -G "Ninja" -DCUDA_CC=$CUDA_CC -DCUDA_HOME=$CUDA_HOME \
    -DTASK_NO=$TaskNo \
    -DTARGET_BIN_OUTPUT_DIR=$Prefix \
    -DCMAKE_BUILD_TYPE=$BuildType \
    -DTEST_KERNEL_VERSION=$TestKernelVersion \
    -DTEST_DATA_TYPE=$TestDataType

if [ "$CleanFirst" = "true" ]; then
    cmake --build . --parallel $(nproc) --clean-first
else
    cmake --build . --parallel $(nproc)
fi

cd $PROJ_HOME
echo "Build finished."
