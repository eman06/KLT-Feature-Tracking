# Makefile for KLT builds and profiling
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
FLAG2 = -DKLT_USE_QSORT   # Uncomment for standard qsort()
CFLAGS = $(FLAG1) $(FLAG2)

LIB = -L/usr/local/lib -L/usr/lib

EXAMPLES = example1.c example2.c example3.c example4.c example5.c
ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c

.PHONY: all libklt.a profile run-profile depend clean help

all: libklt.a $(EXAMPLES:.c=)

# compile .c -> .o
%.o: %.c
	$(CC) -c $(CFLAGS) -o $@ $<

# static library
libklt.a: $(ARCH:.c=.o)
	rm -f libklt.a
	ar ruv libklt.a $(ARCH:.c=.o)

# example targets (unoptimized by default to make debugging easy)
example1: libklt.a
	$(CC) -O0 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example2: libklt.a
	$(CC) -O0 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example3: libklt.a
	$(CC) -O0 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example4: libklt.a
	$(CC) -O0 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example5: libklt.a
	$(CC) -O0 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

# -----------------------------------------------------------------------------
# Profiling build (instrumented for gprof) -- matches the workflow you used.
# Use 'make run-profile' to build, run, and generate analysis.txt.
# -----------------------------------------------------------------------------
profile: clean
	@echo "Building instrumented program (gcc -pg -O0 ... -> myprogram)"
	$(CC) -pg -O0 $(CFLAGS) $(ARCH) example1.c -o myprogram -lm
	@echo "Built myprogram (instrumented)."

run-profile: profile
	@echo "Running instrumented program (will generate gmon.out)"
	./myprogram img1.ppm img2.ppm
	@echo "Running gprof to produce analysis.txt"
	gprof ./myprogram gmon.out > analysis.txt || true
	@echo "Profile saved to analysis.txt (and gmon.out was produced in the current directory)"

# -----------------------------------------------------------------------------
# Helpful: optimized build + perf profiling (recommended before GPU porting)
# -----------------------------------------------------------------------------
profile-optimized: clean
	@echo "Building optimized binary for sampling profilers (-O3 -g -fno-omit-frame-pointer)"
	$(CC) -O3 -g -fno-omit-frame-pointer $(CFLAGS) $(ARCH) example1.c -o myprogram.opt -lm
	@echo "Use a sampling profiler (perf, VTune, etc.) with ./myprogram.opt ..."

# dependency helper (requires makedepend installed)
depend:
	makedepend $(ARCH) $(EXAMPLES) || true

clean:
	rm -f *.o *.a $(EXAMPLES:.c=) *.tar *.tar.gz libklt.a \
	\teat*.ppm features.ft features.txt gmon.out myprogram myprogram.opt analysis.txt

help:
	@echo "Targets:"
	@echo "  make run-profile   # build instrumented program, run it, and gprof -> analysis.txt"
	@echo "  make profile       # build instrumented program only"
	@echo "  make profile-optimized # build optimized binary for perf / sampling profilers"
	@echo "  make all           # build libklt.a and examples (unoptimized)"
	@echo "  make clean         # remove artifacts"
