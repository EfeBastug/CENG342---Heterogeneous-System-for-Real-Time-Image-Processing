#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cfloat>
#include <cstdlib>
#include <ctime>

#define K 8
#define MAX_ITER 4
#define BLOCK_SIZE 256
#define CONVERGENCE_THRESHOLD 0.5f

__global__ void assignClusters(uchar3* pixels, float3* centroids, int* labels, int totalPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalPixels) return;

    uchar3 pixel = pixels[idx];
    float min_dist = FLT_MAX;
    int cluster = 0;

    for (int k = 0; k < K; k++) {
        float3 c = centroids[k];
        float dr = pixel.x - c.x;
        float dg = pixel.y - c.y;
        float db = pixel.z - c.z;
        float dist = dr * dr + dg * dg + db * db;

        if (dist < min_dist) {
            min_dist = dist;
            cluster = k;
        }
    }
    labels[idx] = cluster;
}

__global__ void applyColors(uchar3* output, uchar3* input, float3* centroids, int* labels, int totalPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalPixels) return;


    int cluster = labels[idx];
    output[idx] = make_uchar3(centroids[cluster].x, centroids[cluster].y, centroids[cluster].z);
}

__global__ void countAndSum(uchar3* pixels, int* labels, int* counts, float3* sums, int totalPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalPixels) return;

    int cluster = labels[idx];
    uchar3 pixel = pixels[idx];

    atomicAdd(&counts[cluster], 1);
    atomicAdd(&sums[cluster].x, (float)pixel.x);
    atomicAdd(&sums[cluster].y, (float)pixel.y);
    atomicAdd(&sums[cluster].z, (float)pixel.z);
}

__global__ void updateCentroids(float3* centroids, float3* sums, int* counts) {
    int k = threadIdx.x;
    if (k >= K) return;

    if (counts[k] > 0) {
        centroids[k].x = sums[k].x / counts[k];
        centroids[k].y = sums[k].y / counts[k];
        centroids[k].z = sums[k].z / counts[k];
    }
}

void kmeansCUDA(uchar3* d_input, float3* d_centroids, int* d_labels,
    int* d_counts, float3* d_sums, int totalPixels, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((totalPixels + block.x - 1) / block.x);

    // Assign clusters
    assignClusters << <grid, block, 0, stream >> > (d_input, d_centroids, d_labels, totalPixels);

    // Reset counts and sums
    cudaMemsetAsync(d_counts, 0, K * sizeof(int), stream);
    cudaMemsetAsync(d_sums, 0, K * sizeof(float3), stream);

    // Calculate sums and counts
    countAndSum <<<grid, block, 0, stream >> > (d_input, d_labels, d_counts, d_sums, totalPixels);

    // Update centroids
    updateCentroids <<<1, K, 0, stream >> > (d_centroids, d_sums, d_counts);
}

int main() {
    std::srand(std::time(nullptr));
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    cv::Mat frame;
    cap >> frame;
    const int totalPixels = frame.rows * frame.cols;

    // CUDA resources
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate device memory
    uchar3* d_input, * d_output;
    float3* d_centroids;
    int* d_labels, * d_counts;
    float3* d_sums;

    cudaMalloc(&d_input, totalPixels * sizeof(uchar3));
    cudaMalloc(&d_output, totalPixels * sizeof(uchar3));
    cudaMalloc(&d_centroids, K * sizeof(float3));
    cudaMalloc(&d_labels, totalPixels * sizeof(int));
    cudaMalloc(&d_counts, K * sizeof(int));
    cudaMalloc(&d_sums, K * sizeof(float3));

    // Host centroid array
    float3 h_centroids[K];
    for (int k = 0; k < K; k++) {
        // rand() returns [0, RAND_MAX], scale to [0–255]
        h_centroids[k].x = float(std::rand() % 256);
        h_centroids[k].y = float(std::rand() % 256);
        h_centroids[k].z = float(std::rand() % 256);
    }

    // Copy to device
    cudaMemcpy(d_centroids, h_centroids, K * sizeof(float3), cudaMemcpyHostToDevice);

    while (true) {

        cap >> frame;
        if (frame.empty()) break;

        // Upload frame to device
        cudaMemcpyAsync(d_input, frame.data, totalPixels * sizeof(uchar3),
            cudaMemcpyHostToDevice, stream);

        // Process frame with convergence check
        float3 prev_centroids[K], curr_centroids[K];

        for (int iter = 0; iter < MAX_ITER; iter++) {
            cudaMemcpyAsync(prev_centroids, d_centroids, K * sizeof(float3),
                cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            kmeansCUDA(d_input, d_centroids, d_labels, d_counts, d_sums, totalPixels, stream);

            // Check convergence
            float3 curr_centroids[K];
            cudaMemcpyAsync(curr_centroids, d_centroids, K * sizeof(float3),
                cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            float max_delta = 0;
            for (int k = 0; k < K; k++) {
                float dx = curr_centroids[k].x - prev_centroids[k].x;
                float dy = curr_centroids[k].y - prev_centroids[k].y;
                float dz = curr_centroids[k].z - prev_centroids[k].z;
                max_delta = fmaxf(max_delta, sqrtf(dx * dx + dy * dy + dz * dz));
            }

            if (max_delta < CONVERGENCE_THRESHOLD) break;
        }

        // Apply colors to output buffer
        applyColors << <(totalPixels + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream >>> (d_output, d_input, d_centroids, d_labels, totalPixels);

        // Download result
        cudaMemcpyAsync(frame.data, d_output, totalPixels * sizeof(uchar3),
            cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // Display
        cv::imshow("CUDA K-Means", frame);
        if (cv::waitKey(1) == 27) break;
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_counts);
    cudaFree(d_sums);
    cudaStreamDestroy(stream);

    return 0;
}