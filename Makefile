# ============================================================
# Makefile for KLT Feature Tracking (CPU, GPU-Naive, GPU-Optimized)
# Works on both WSL and HPC environments
# ============================================================

# Compilers
CC      = gcc
NVCC    = nvcc

# Directories
SRC_DIR = src
GPU_DIR = GPU_functions
EX_DIR  = examples
INC_DIR = include
DATA_DIR = data

# Compiler flags
CFLAGS     = -O2 -DNDEBUG -DKLT_USE_QSORT -I$(INC_DIR)
NVCCFLAGS  = -O3 -arch=sm_75 -I$(INC_DIR)

# Libraries
LIBS = -lm

# -------------------------------------------------------------------
# Source files
# -------------------------------------------------------------------
CPU_SOURCES = $(SRC_DIR)/convolve.c $(SRC_DIR)/error.c $(SRC_DIR)/pnmio.c \
              $(SRC_DIR)/pyramid.c $(SRC_DIR)/selectGoodFeatures.c \
              $(SRC_DIR)/storeFeatures.c $(SRC_DIR)/trackFeatures.c \
              $(SRC_DIR)/klt.c $(SRC_DIR)/klt_util.c $(SRC_DIR)/writeFeatures.c

GPU_SOURCES = $(GPU_DIR)/convolve_gpu.cu $(GPU_DIR)/selectGoodFeatures_gpu.cu

CPU_EXAMPLE      = $(EX_DIR)/example3.c
GPU_EXAMPLE      = $(EX_DIR)/example3_gpu.c
OPTIMIZED_MAIN   = main.cpp

# -------------------------------------------------------------------
# Targets
# -------------------------------------------------------------------
.PHONY: all clean cpu gpu-naive gpu-opt help

# Default target
all: cpu

# -------------------------------------------------------------------
# CPU build & run
# -------------------------------------------------------------------
cpu: libklt.a
	@echo "🧠 Building CPU version..."
	$(CC) $(CFLAGS) $(CPU_SOURCES) $(CPU_EXAMPLE) -o myprogram $(LIBS)
	@echo "▶️  Running CPU program with timing..."
	@/usr/bin/time -f "\nCPU Execution Time: %E" ./myprogram $(DATA_DIR)/img1.pgm $(DATA_DIR)/img2.pgm

# -------------------------------------------------------------------
# GPU-Naive build & run
# -------------------------------------------------------------------
gpu-naive: libklt.a
	@echo "⚙️  Building GPU-Naive version..."
	$(NVCC) $(NVCCFLAGS) $(GPU_EXAMPLE) $(GPU_SOURCES) $(CPU_SOURCES) -o myprogram_gpu $(LIBS)
	@echo "▶️  Running GPU-Naive program with timing..."
	@/usr/bin/time -f "\nGPU-Naive Execution Time: %E" ./myprogram_gpu $(DATA_DIR)/img1.pgm $(DATA_DIR)/img2.pgm

# -------------------------------------------------------------------
# GPU-Optimized build & run
# -------------------------------------------------------------------
gpu-opt:
	@echo "🚀 Building Optimized GPU version..."
	$(NVCC) $(NVCCFLAGS) $(OPTIMIZED_MAIN) $(GPU_DIR)/kernels_shared.cu $(GPU_DIR)/klt_gpu.cu \
	$(CPU_SOURCES) -o myprogram_gpu_opt $(LIBS)
	@echo "▶️  Running Optimized GPU version with timing..."
	@/usr/bin/time -f "\nGPU-Optimized Execution Time: %E" ./myprogram_gpu_opt $(DATA_DIR)/img1.pgm $(DATA_DIR)/img2.pgm
	@echo "✅ Built Optimized GPU version successfully!"

# -------------------------------------------------------------------
# Build static library for CPU files
# -------------------------------------------------------------------
libklt.a: $(CPU_SOURCES:.c=.o)
	@echo "📦 Creating static library libklt.a..."
	rm -f libklt.a
	ar ruv libklt.a $(CPU_SOURCES:.c=.o)

# -------------------------------------------------------------------
# Compile .c -> .o
# -------------------------------------------------------------------
$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) -c $(CFLAGS) -o $@ $<

# -------------------------------------------------------------------
# Clean build artifacts
# -------------------------------------------------------------------
clean:
	@echo "🧹 Cleaning project..."
	rm -f $(SRC_DIR)/*.o *.a myprogram myprogram_gpu myprogram_gpu_opt \
		features.ft features.txt feat*.ppm gmon.out analysis.txt
	@echo "✅ Clean complete!"

# -------------------------------------------------------------------
# Help menu
# -------------------------------------------------------------------
help:
	@echo "================ KLT Feature Tracking Build System ================"
	@echo "make cpu        → Build & run CPU implementation"
	@echo "make gpu-naive  → Build & run GPU-naive implementation"
	@echo "make gpu-opt    → Build & run GPU-optimized implementation"
	@echo "make clean      → Remove all build artifacts"
	@echo "==================================================================="
