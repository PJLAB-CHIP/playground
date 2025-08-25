TEST_FILE=""
OUTPUT_FILE=""
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--test-file)
            TEST_FILE=$2; shift ;;
        -o|--output-file)
            OUTPUT_FILE=$2; shift ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# Generate default output filename if not provided
if [ -z "$OUTPUT_FILE" ]; then
    if [ -n "$TEST_FILE" ]; then
        # Extract binary name from test file path
        BINARY_NAME=$(basename "$TEST_FILE")
        OUTPUT_FILE="./profiles/${BINARY_NAME}_${TIMESTAMP}.ncu-rep"
        # Create profiles directory if it doesn't exist
        mkdir -p "./profiles"
    else
        OUTPUT_FILE="./nsight_profile_${TIMESTAMP}.ncu-rep"
    fi
fi


echo "Starting Nsight Compute profiling..."
echo "Test file: $TEST_FILE"
echo "Output file: $OUTPUT_FILE"

ncu --export $OUTPUT_FILE \
    --force-overwrite \
    --set "full" \
    $TEST_FILE

if [ $? -eq 0 ]; then
    echo "Profiling completed successfully!"
    echo "Report saved to: $OUTPUT_FILE"
else
    echo "Profiling failed!"
    exit 1
fi 