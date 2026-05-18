#include <cstdio>
#include <cuda_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1024;
    size_t bytes = n * sizeof(float);

    float *h_a = new float[n], *h_b = new float[n], *h_c = new float[n];
    for (int i = 0; i < n; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < n; i++) {
        float diff = h_c[i] - (h_a[i] + h_b[i]);
        if (diff < 0) diff = -diff;
        if (diff > 1e-5f) { ok = false; break; }
    }
    printf("Vector Add (%d elements): %s\n", n, ok ? "PASS" : "FAIL");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    return 0;
}
