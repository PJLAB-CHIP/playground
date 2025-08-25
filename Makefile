# 默认参数
FLOAT ?= f32
VER ?= 1

# 根据 FLOAT 参数设置数据类型
ifeq ($(FLOAT),f16)
    FLOAT_TYPE = float16
    FLOAT_FLAG = -f16
else
    FLOAT_TYPE = float32
    FLOAT_FLAG = -f32
endif

# 构建目录和二进制文件路径
BUILD_DIR = ./build/src
BINARY_NAME = task1_$(FLOAT_TYPE)_v$(VER)
BINARY_PATH = $(BUILD_DIR)/$(BINARY_NAME)
LOGS_DIR = logs

# 时间戳
TIMESTAMP = $(shell date +"%Y%m%d_%H%M%S")

.PHONY: all build run debug profile clean help

all: help

# 1) 编译构建指定 FLOAT 类型和 VERSION 的代码
build:
	@echo "=== Building with FLOAT=$(FLOAT) VERSION=$(VER) ==="
	bash scripts/build-task1.sh $(FLOAT_FLAG) -v$(VER)

# 2) 运行指定 FLOAT 类型和 VERSION 的代码
run: build
	@echo "=== Running $(BINARY_NAME) ==="
	@mkdir -p $(LOGS_DIR)
	@echo "Run started at: $(shell date)" > $(LOGS_DIR)/$(BINARY_NAME)_$(TIMESTAMP).log
	@echo "FLOAT_TYPE: $(FLOAT_TYPE)" >> $(LOGS_DIR)/$(BINARY_NAME)_$(TIMESTAMP).log
	@echo "VERSION: $(VER)" >> $(LOGS_DIR)/$(BINARY_NAME)_$(TIMESTAMP).log
	@echo "======================================" >> $(LOGS_DIR)/$(BINARY_NAME)_$(TIMESTAMP).log
	$(BINARY_PATH) 2>&1 | tee -a $(LOGS_DIR)/$(BINARY_NAME)_$(TIMESTAMP).log
	@echo "======================================" >> $(LOGS_DIR)/$(BINARY_NAME)_$(TIMESTAMP).log
	@echo "Run completed at: $(shell date)" >> $(LOGS_DIR)/$(BINARY_NAME)_$(TIMESTAMP).log
	@echo "Log saved to: $(LOGS_DIR)/$(BINARY_NAME)_$(TIMESTAMP).log"

# 3) 编译构建带有完整符号的代码 (RelWithDebInfo)
debug:
	@echo "=== Building Debug version with FLOAT=$(FLOAT) VERSION=$(VER) ==="
	bash scripts/build-task1.sh $(FLOAT_FLAG) -v$(VER) RD

# 4) 用 nsight compute 导出性能分析报告
profile: debug
	@echo "=== Profiling $(BINARY_NAME) with Nsight Compute ==="
	@mkdir -p $(LOGS_DIR)/profiles
	bash scripts/nsight-profile.sh -t $(BINARY_PATH) -o $(LOGS_DIR)/profiles/$(BINARY_NAME)_$(TIMESTAMP).ncu-rep
	@echo "Profile saved to: $(LOGS_DIR)/profiles/$(BINARY_NAME)_$(TIMESTAMP).ncu-rep"

# 清理构建文件
clean:
	rm -rf $(BUILD_DIR)
	@echo "Build directory cleaned"

# 清理日志文件
clean-logs:
	rm -rf $(LOGS_DIR)
	@echo "Logs directory cleaned"

# 显示帮助信息
help:
	@echo "Makefile for building and running task-1"
	@echo ""
	@echo "Usage: make [target] [FLOAT=f32|f16] [VER=version_number]"
	@echo ""
	@echo "Targets:"
	@echo "  build     - Build code with specified FLOAT type and VERSION"
	@echo "  run       - Build and run code, save output to logs"
	@echo "  debug     - Build with debug symbols (RelWithDebInfo)"
	@echo "  profile   - Run nsight compute profiling"
	@echo "  clean     - Clean build directory"
	@echo "  clean-logs- Clean logs directory"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Parameters:"
	@echo "  FLOAT     - Data type: f32 (default) or f16"
	@echo "  VER       - Version number (default: 1)"
	@echo ""
	@echo "Examples:"
	@echo "  make build FLOAT=f16 VER=2    # Build float16 version 2"
	@echo "  make run FLOAT=f32 VER=1      # Run float32 version 1"
	@echo "  make debug FLOAT=f16 VER=3    # Build debug version"
	@echo "  make profile FLOAT=f32 VER=1  # Profile with nsight compute"