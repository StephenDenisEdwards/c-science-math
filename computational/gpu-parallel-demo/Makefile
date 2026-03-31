CC      = gcc
CFLAGS  = -O2 -Wall -Wextra
NVCC    = nvcc
NVFLAGS = -O2

all: cuda opencl openmp

cuda: vector_add.cu
	$(NVCC) $(NVFLAGS) -o vector_add vector_add.cu

opencl: vector_add_opencl.c
	$(CC) $(CFLAGS) -o vector_add_opencl vector_add_opencl.c -lOpenCL

openmp: vector_add_openmp.c
	$(CC) $(CFLAGS) -fopenmp -o vector_add_openmp vector_add_openmp.c

openmp-gpu: vector_add_openmp.c
	$(CC) $(CFLAGS) -fopenmp -foffload=nvptx-none -o vector_add_openmp vector_add_openmp.c

clean:
	rm -f vector_add vector_add_opencl vector_add_openmp

.PHONY: all cuda opencl openmp openmp-gpu clean
