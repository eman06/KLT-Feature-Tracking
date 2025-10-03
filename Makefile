######################################################################

# Compiler

CC = gcc

######################################################################

# Flags

FLAG1 = -DNDEBUG

# FLAG2 = -DKLT_USE_QSORT   # Uncomment to use standard qsort()

# Base flags

CFLAGS = $(FLAG1) $(FLAG2)

######################################################################

# Sources

EXAMPLES = example1.c example2.c example3.c example4.c example5.c
ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c 
storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c
LIB = -L/usr/local/lib -L/usr/lib

######################################################################

# Rules

.SUFFIXES:  .c .o

all: lib $(EXAMPLES:.c=)

.c.o:
$(CC) -c $(CFLAGS) $<

lib: $(ARCH:.c=.o)
rm -f libklt.a
ar ruv libklt.a $(ARCH:.c=.o)
rm -f *.o

# Normal builds (optimized, no profiling)

example1: libklt.a
$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm
example2: libklt.a
$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm
example3: libklt.a
$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm
example4: libklt.a
$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm
example5: libklt.a
$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

# Profiling build (no optimizations, with gprof instrumentation)

profile: clean
$(CC) -pg -O0 $(ARCH) example1.c -o myprogram -lm

run-profile: profile
./myprogram img1.ppm img2.ppm
gprof ./myprogram gmon.out > analysis.txt
@echo "Profile saved to analysis.txt"

depend:
makedepend $(ARCH) $(EXAMPLES)

clean:
rm -f *.o *.a $(EXAMPLES:.c=) *.tar *.tar.gz libklt.a 
feat*.ppm features.ft features.txt gmon.out myprogram analysis.txt
