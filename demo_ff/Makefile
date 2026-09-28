NVCC ?= nvcc
MPICXX ?= mpicxx
NVCCFLAGS ?= -O2 -std=c++17
MPI_CFLAGS := $(shell $(MPICXX) --showme:compile)
MPI_LIBS := $(shell $(MPICXX) --showme:link)
HEADERS := $(wildcard *.cuh kernels/*.cuh)

.PHONY: all clean
all: multi-gpu

multi-gpu: main.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(MPI_CFLAGS) $< -o $@ $(MPI_LIBS) -lcublas -lnccl

clean:
	rm -f multi-gpu
