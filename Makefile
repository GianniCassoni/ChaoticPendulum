# Makefile for compiling Fractal_mamaCUDA.cu and Fractal_mama_Parallel.cpp

# Compiler for CUDA
NVCC = nvcc

# Compiler for C++
CC = g++

# Compiler flags
CFLAGS = -I ./usr/include

# Source files
CUDA_SRC = Fractal_mamaCUDA.cu
CPP_SRC = Fractal_mama_Parallel.cpp

# Output executables
CUDA_TARGET = Fractal_mamaCUDA
CPP_TARGET = Fractal_mama

# Default target
all: $(CUDA_TARGET) $(CPP_TARGET)

# Rule to compile the CUDA source file
$(CUDA_TARGET): $(CUDA_SRC)
	$(NVCC) $(CUDA_SRC) -o $(CUDA_TARGET)

# Rule to compile the C++ source file
$(CPP_TARGET): $(CPP_SRC)
	$(CC) $(CFLAGS) -O3 -o $@ $(CPP_SRC)

# Clean target to remove the compiled executables
clean:
	rm -f $(CUDA_TARGET) $(CPP_TARGET)
