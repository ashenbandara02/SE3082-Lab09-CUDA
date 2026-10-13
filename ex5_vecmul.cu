#include <cstdio>
#include <cstdlib>

#define N 10000000              // 10 million elements
#define THREADS_PER_BLOCK 512

__global__ void vecMul(int *a, int *b, int *c, int n) {
    // Solution: global index of this thread (use blockIdx, blockDim, threadIdx)
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Solution: only compute if i is inside the array. Why is this needed?
    if (i < n)
        c[i] = a[i] * b[i];
}

void random_ints(int *x, int size) {
    for (int i = 0; i < size; i++)
        x[i] = rand() % 100;
}

int main(void) {
    int *a, *b, *c;             // host copies of a, b, c
    int *d_a, *d_b, *d_c;       // device copies of a, b, c
    size_t size = (size_t)N * sizeof(int);

    // Allocate device memory
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Allocate host memory and generate the input values
    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Solution: number of blocks needed to cover N elements (round UP)
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    printf("Launching %d blocks x %d threads\n", blocks, THREADS_PER_BLOCK);

    vecMul<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = N - 1000; i < N; i++)
        printf("%d) %d x %d = %d\n", i, a[i], b[i], c[i]);

    int errors = 0;
    for (int i = 0; i < N; i++)
        if (c[i] != a[i] * b[i]) errors++;
    printf("Verification: %d mismatches out of %d\n", errors, N);

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
