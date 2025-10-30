# Makefile for KLT CPU and GPU-naive builds

CC = gcc
NVCC = nvcc

# Flags
CFLAGS = -DNDEBUG -DKLT_USE_QSORT -I./include -I./src
NVCCFLAGS = -O3 -I./include -I./src

LIB = -L. -L/usr/local/lib -L/usr/lib -lm

CPU_EXAMPLE = examples/example3.c         # CPU version
GPU_EXAMPLE = GPU_functions/example3_gpu.c     # GPU-naive version

CPU_SOURCES = src/convolve.c src/error.c src/pnmio.c src/pyramid.c src/selectGoodFeatures.c \
              src/storeFeatures.c src/trackFeatures.c src/klt.c src/klt_util.c src/writeFeatures.c
GPU_SOURCES = GPU_functions/convolve_gpu.cu GPU_functions/selectGoodFeatures_gpu.cu

.PHONY: all clean cpu gpu-naive help

# -------------------------------------------------------------------
# Default target
# -------------------------------------------------------------------
all: cpu

# -------------------------------------------------------------------
# CPU build and run
# -------------------------------------------------------------------
cpu: libklt.a
	@echo "Building CPU version..."
	$(CC) -O0 $(CFLAGS) $(CPU_SOURCES) $(CPU_EXAMPLE) -o myprogram $(LIB)
	@echo "Running CPU program with timing..."
	@/usr/bin/time -f "\nCPU Execution Time: %E" ./myprogram img1.ppm img2.ppm

# -------------------------------------------------------------------
# GPU-naive build and run
# -------------------------------------------------------------------
gpu-naive: libklt.a
	@echo "Building GPU-naive version..."
	$(NVCC) $(NVCCFLAGS) -arch=sm_75 -Xcompiler -w \
	         $(GPU_EXAMPLE) $(GPU_SOURCES) \
	         $(CPU_SOURCES) \
	         -o myprogram_gpu -lm
	@echo "Running GPU-naive program with timing..."
	@/usr/bin/time -f "\nGPU Execution Time: %E" ./myprogram_gpu data/img1.ppm data/img2.pgm 

# -------------------------------------------------------------------
# Build static library for CPU code
# -------------------------------------------------------------------
libklt.a: $(CPU_SOURCES:.c=.o)
	rm -f libklt.a
	ar ruv libklt.a $(CPU_SOURCES:.c=.o)

# -------------------------------------------------------------------
# Compile .c -> .o
# -------------------------------------------------------------------
%.o: %.c
	$(CC) -c $(CFLAGS) -o $@ $<

# -------------------------------------------------------------------
# Clean
# -------------------------------------------------------------------
clean:
	rm -f *.o *.a myprogram myprogram_gpu \
		features.ft features.txt gmon.out myprogram.opt analysis.txt feat*.ppm

# -------------------------------------------------------------------
# Help
# -------------------------------------------------------------------
help:
	@echo "Targets:"
	@echo "  make cpu         # build & run CPU implementation"
	@echo "  make gpu-naive   # build & run GPU implementation"
	@echo "  make clean       # remove artifacts"
