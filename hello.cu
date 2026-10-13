#include <cstdio>

__global__ void hello() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main(void) {
    hello<<<1, 4>>>();          // 1 block of 4 threads
    cudaDeviceSynchronize();    // wait so the GPU's output is flushed
    return 0;
}
