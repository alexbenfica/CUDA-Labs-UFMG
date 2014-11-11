EXECUTABLE  := life
OBJECTS := life.o utils.o

CUDA_PATH ?=/usr/local/cuda
NVCC=$(CUDA_PATH)/bin/nvcc

CUDAFLAGS := -g
LDFLAGS := -g

%.o: %.cu
	$(NVCC) $(CUDAFLAGS) -c $<

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(NVCC) $(OBJECTS)  $(LDFLAGS) -o $(EXECUTABLE)

life.o: life_kernel.cu

clean:
	rm -f $(EXECUTABLE) $(OBJECTS)

