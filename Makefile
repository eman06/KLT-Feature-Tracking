# Makefile for KLT CPU and GPU-V3 builds

CC = gcc
NVCC = /opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/bin/nvcc

# Flags
CFLAGS = -DNDEBUG -DKLT_USE_QSORT
NVCCFLAGS = -O3 -DKLT_USE_GPU

LIB = -L. -L/usr/local/lib -L/usr/lib -lm

CPU_EXAMPLE = example3.c         # CPU version
GPU_EXAMPLE = example3_gpu.c     # GPU-Advanced version

CPU_SOURCES = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
              storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c
GPU_SOURCES = convolve_gpu.cu selectGoodFeatures_gpu.cu

.PHONY: all clean cpu gpu-v3 help profile-gpu profile-gpu-ncu profile-gpu-nsys

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
# GPU-V3 build and run
# -------------------------------------------------------------------
gpu-v3: libklt.a
	@echo "Building GPU-V3 version..."
	$(NVCC) $(NVCCFLAGS) -arch=sm_75 -Xcompiler -w \
	         $(GPU_EXAMPLE) $(GPU_SOURCES) \
	         $(CPU_SOURCES) \
	         -o myprogram_gpu -lm
	@echo "Running GPU-V3 program with timing..."
	@/usr/bin/time -f "\nTotal Program Execution Time: %E" ./myprogram_gpu img1.ppm img2.ppm

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
		features.ft features.txt gmon.out myprogram.opt analysis.txt feat*.ppm analysis_report.* *.nsys-rep *.qdrep

# -------------------------------------------------------------------
# GPU Profiling (modern NVIDIA GPUs)
# -------------------------------------------------------------------
profile-gpu-ncu: gpu-v3
	@echo "Profiling GPU binary with Nsight Compute (ncu)..."
	ncu --set full --target-processes all --log-file analysis.txt ./myprogram_gpu img1.ppm img2.ppm || echo "ncu failed or not found."

profile-gpu-nsys: gpu-v3
	@echo "Profiling GPU binary with Nsight Systems (nsys)..."
	nsys profile -o analysis_report ./myprogram_gpu img1.ppm img2.ppm || echo "nsys failed or not found."

profile-gpu: profile-gpu-ncu

# -------------------------------------------------------------------
# Help
# -------------------------------------------------------------------
help:
	@echo "Targets:"
	@echo "  make cpu              # build & run CPU implementation"
	@echo "  make gpu-v3           # build & run GPU implementation"
	@echo "  make profile-gpu      # profile GPU with Nsight Compute (ncu), output to analysis.txt"
	@echo "  make profile-gpu-ncu  # profile GPU with ncu, output to analysis.txt"
	@echo "  make profile-gpu-nsys # profile GPU with nsys, output to analysis_report.qdrep"
	@echo "  make clean            # remove artifacts"
