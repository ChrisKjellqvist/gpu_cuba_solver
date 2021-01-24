FLAGS = -O3

gpu_solver: main.cu
	nvcc $(FLAGS) $^ -o gpu_solver

.PHONY: clean
clean:
	rm -rf gpu_solver
