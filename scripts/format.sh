# [NOTE] 
# |- This script is used to format the code using clang-format and black.
# |- Although CI pipeline is configured to format the code after pushing to main,
# |- it is still recommended to run this script manually before committing the code.

set -e  # Exit on error

FORMAT_C_CXX=false
FORMAT_PYTHON=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|-cxx|-c++)
            FORMAT_C_CXX=true ;;
        -py|-python)
            FORMAT_PYTHON=true ;;
        *)
            # [TODO] Add detailed help message
            echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

if [[ "$FORMAT_C_CXX" == "true" ]]; then
    echo "Formatting C/C++/CUDA files..."
    files=$(git ls-files | grep -E '\.(c|h|cpp|hpp|cu|cuh)$')
    if [[ -n "$files" ]]; then
        clang-format -i $files
    fi
fi

if [[ "$FORMAT_PYTHON" == "true" ]]; then
    echo "Formatting Python files..."
    files=$(git ls-files '*.py')
    if [[ -n "$files" ]]; then
        black --quiet --fast $files
    fi
fi

echo "All formatting completed!"