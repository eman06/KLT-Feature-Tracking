# Makefile for KLT builds and profiling with reorganized directory structure
# Usage:
#   make run-profile   # builds instrumented binary, runs it, and generates analysis.txt
#   make profile       # builds instrumented binary only
#   make all           # builds libklt.a and example programs (unoptimized)
#   make clean         # remove build/artifacts
#
# Notes:
#  - The 'run-profile' target will compile with -pg and -O0 (the same flags you used),
#    run the program to produce gmon.out, and then run gprof to create analysis.txt.
#  - For realistic hotspot identification before GPU porting, also run the
#    "optimized profiling" commands documented in README.md (perf / flamegraphs).
#  - Ensure this Makefile is edited to point to your actual image filenames if needed.

CC = gcc

# Flags
FLAG1 = -DNDEBUG
FLAG2 = -DKLT_USE_QSORT
CFLAGS = $(FLAG1) $(FLAG2) -I./include

# Directories
SRC_DIR = src
INCLUDE_DIR = include
EXAMPLE_DIR = examples
DATA_DIR = data
BUILD_DIR = build

# Source files
SRC_FILES = $(wildcard $(SRC_DIR)/*.c)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRC_FILES))

# Example files
EXAMPLES = $(wildcard $(EXAMPLE_DIR)/*.c)
EXAMPLE_BINS = $(patsubst $(EXAMPLE_DIR)/%.c,%,$(EXAMPLES))

# Library
LIB_FILE = $(BUILD_DIR)/libklt.a

# Default target
all: $(LIB_FILE) $(EXAMPLE_BINS)

# Build the library
$(LIB_FILE): $(OBJ_FILES)
	ar rcs $@ $^

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -o $@ $<

# Build examples
%: $(EXAMPLE_DIR)/%.c $(LIB_FILE)
	$(CC) $(CFLAGS) -o $@ $< -L$(BUILD_DIR) -lklt -lm

# Create build directory
$(BUILD_DIR):
	mkdir -p $@

# Clean target
clean:
	rm -rf $(BUILD_DIR)/*.o $(BUILD_DIR)/*.a $(EXAMPLE_BINS)
	rm -f feat*.ppm features.ft features.txt gmon.out myprogram myprogram.opt analysis.txt

# Profiling targets
profile: clean
	@echo "Building instrumented program (gcc -pg -O0 ... -> myprogram)"
	$(CC) -pg -O0 $(CFLAGS) -I$(INCLUDE_DIR) $(EXAMPLE_DIR)/example1.c $(SRC_FILES) -o myprogram -lm
	@echo "Built myprogram (instrumented)."

run-profile: profile
	@echo "Running instrumented program (will generate gmon.out)"
	cd $(DATA_DIR) && ../myprogram img1.pgm img2.pgm
	gprof ./myprogram gmon.out > analysis.txt
	@echo "Created analysis.txt"

profile-optimized:
	@echo "Building optimized program with debug symbols (gcc -O3 -g -fno-omit-frame-pointer -> myprogram.opt)"
	$(CC) -O3 -g -fno-omit-frame-pointer $(CFLAGS) -I$(INCLUDE_DIR) $(EXAMPLE_DIR)/example1.c $(SRC_FILES) -o myprogram.opt -lm
	@echo "Built myprogram.opt"