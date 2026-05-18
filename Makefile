NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -arch=sm_120

TARGET = vector_add

all: $(TARGET)

$(TARGET): vector_add.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
