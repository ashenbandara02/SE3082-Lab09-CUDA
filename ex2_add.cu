#include <cstdio>

// Kernel: this code runs on the device (the NVIDIA GPU)
__global__ void add(int *a, int *b, int *c) {
    *c = *a + *b;
}

// host = CPU, device = GPU
int main(void) {
    int a, b, c;                // host copies of a, b, c
    int *d_a, *d_b, *d_c;       // device copies of a, b, c
    int size = sizeof(int);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Set up input values
    a = 2;
    b = 7;

    // 1. Copy the inputs from host to device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    // 2. Launch the add() kernel on the GPU: 1 block, 1 thread
    add<<<1, 1>>>(d_a, d_b, d_c);

    // Colab edition: report a failed launch instead of silently printing garbage
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));

    // 3. Copy the result from device to host
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
    printf("Result is %d\n", c);

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
