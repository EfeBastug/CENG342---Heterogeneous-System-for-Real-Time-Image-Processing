#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cfloat>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <numeric>
#include <limits>
#include <cmath>
#include <vector>

#define K 6 // Number of clusters (KMeans)
#define MAX_ITER 10 // Maximum iterations (KMeans)
#define BLOCK_SIZE 256 // Threads per block (KMeans, general purpose)
#define WARPS_PER_BLOCK  (BLOCK_SIZE / 32) // Number of warps per block (KMeans)
#define CONVERGENCE_THRESHOLD 0.01f // Convergence threshold (KMeans)

// --- Constants for Gaussian Blur ---
#define GAUSS_KERNEL_SIZE 5       // Example kernel size (must be odd)
#define GAUSS_KERNEL_RADIUS (GAUSS_KERNEL_SIZE / 2)
#define GAUSS_SIGMA 1.5f        // Example sigma value
#define GAUSS_BLOCK_DIM_X 16    // Block dimension for Gaussian Blur kernel
#define GAUSS_BLOCK_DIM_Y 16

enum class FilterType {
    KMeans,
    EdgeDetection,
    Otsu,
    GaussianBlur // Added Gaussian Blur
};

std::ostream& operator<<(std::ostream& os, const FilterType& filter) {
    switch (filter) {
    case FilterType::KMeans:
        os << "KMeans";
        break;
    case FilterType::EdgeDetection:
        os << "EdgeDetection";
        break;
    case FilterType::Otsu:
        os << "Otsu";
        break;
    case FilterType::GaussianBlur: // Added Gaussian Blur output
        os << "GaussianBlur";
        break;
    }
    return os;
}

// --- CUDA Error Checking Macro ---
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " (" << cudaGetErrorString(result) << ") "
            << "at " << file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}


/**************************************************************
*
*                       KMEANS
*
************************************************************/

/*
* CUDA kernel to perform warp reduction for integer
* * * * */
__device__ int   warpReduceIntSum(int   value)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

/*
* CUDA kernel to perform warp reduction for float
* * * * */
__device__ float warpReduceFltSum(float value)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

/*
*  CUDA kernel to assign clusters to pixels
*  This kernel calculates the distance from each pixel to each centroid
*  and assigns the pixel to the nearest centroid.
*  The labels array stores the cluster index for each pixel.
* * * * */
__global__ void assignClusters(uchar3* pixels, float3* centroids, int* labels, int totalPixels) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;// Calculate global thread index
    if (index >= totalPixels) return; // Check bounds

    uchar3 pixel = pixels[index]; // Get pixel data
    float min_dist = FLT_MAX; // Initialize minimum distance
    int cluster = 0; // Initialize cluster index

    // Loop through each centroid to find the nearest one
    #pragma unroll
    for (int k = 0; k < K; k++) {
        float3 c = centroids[k];
        float db = pixel.x - c.x;
        float dg = pixel.y - c.y;
        float dr = pixel.z - c.z;
        float dist = dr * dr + dg * dg + db * db;

        if (dist < min_dist) {
            min_dist = dist;
            cluster = k;
        }
    }

    labels[index] = cluster; // Assign the pixel to the nearest cluster
}

/*
* CUDA kernel to apply colors to the output image
* This kernel assigns the color of the centroid to each pixel
* based on the cluster it belongs to.
* * * * */
__global__ void applyColors(uchar3* output, float3* centroids, int* labels, int totalPixels) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;// Calculate global thread index
    if (idx >= totalPixels) return;// Check bounds

    // Get the cluster index for the pixel and assign the color of the centroid to the pixel
    int cluster = labels[idx];
    output[idx] = make_uchar3((unsigned char)centroids[cluster].x, (unsigned char)centroids[cluster].y, (unsigned char)centroids[cluster].z);
}

/*
*  CUDA kernel to count pixels and calculate sums for each cluster
*  This kernel counts the number of pixels in each cluster and calculates
*  the sum of pixel colors for each cluster.
*  The counts array stores the number of pixels in each cluster,
*  and the sums array stores the sum of pixel colors for each cluster.
* * * * */
__global__ void countAndSum(uchar3* pixels, int* labels, int* counts, float3* sums, int totalPixels) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;// Calculate global thread index
    bool isValid = index < totalPixels;// Check if the pixel is valid

    int local_cluster_index = isValid ? labels[index] : 0; // Get the cluster index for the pixel, 0 if invalid
    uchar3 local_pixel = isValid ? pixels[index] : make_uchar3(0, 0, 0); // Get the pixel data, 0 if invalid

    int lane = threadIdx.x % warpSize;// Get the lane index within the warp
    int warp_id = threadIdx.x / warpSize;// Get the warp index

    // Initialize shared memory for counts and sums
    __shared__ int local_counts[WARPS_PER_BLOCK][K];
    __shared__ float3 local_sums[WARPS_PER_BLOCK][K];


    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    * Warp-level reduction to count pixels and calculate sums  *
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
    #pragma unroll
    for (int cluster_index = 0; cluster_index < K; cluster_index++) {

        //If it belongs to the cluster, increment the count and add the pixel color to the sum
        int local_count = isValid && (cluster_index == local_cluster_index) ? 1 : 0;
        float b = isValid && (cluster_index == local_cluster_index) ? local_pixel.x : 0;
        float g = isValid && (cluster_index == local_cluster_index) ? local_pixel.y : 0;
        float r = isValid && (cluster_index == local_cluster_index) ? local_pixel.z : 0;

        // Reduce the count and sum of pixel colors within the warp
        local_count = warpReduceIntSum(local_count);
        b = warpReduceFltSum(b);
        g = warpReduceFltSum(g);
        r = warpReduceFltSum(r);


        // Only the first thread in the warp writes the results to shared memory
        if (lane == 0) {
            local_counts[warp_id][cluster_index] = local_count;
            local_sums[warp_id][cluster_index] = make_float3(b, g, r);
        }
    }

    __syncthreads(); // Synchronize threads within the block


    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    * Block-level reduction to count pixels and calculate sums *
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

    if (warp_id == 0 && threadIdx.x < K) {// Only the first warp in the block processes the results

        int local_block_cluster_count = 0;
        float3 local_block_cluster_sum = make_float3(0.0f, 0.0f, 0.0f);

        for (int warp_index = 0; warp_index < WARPS_PER_BLOCK; warp_index++) {// Iterate through all warps in the block

            local_block_cluster_count += local_counts[warp_index][threadIdx.x];// Sum counts from all warps for the cluster (0..K)
            local_block_cluster_sum.x += local_sums[warp_index][threadIdx.x].x;// Sum blue channel from all warps for the cluster (0..K)
            local_block_cluster_sum.y += local_sums[warp_index][threadIdx.x].y;// Sum green channel from all warps for the cluster (0..K)
            local_block_cluster_sum.z += local_sums[warp_index][threadIdx.x].z;// Sum red channel from all warps for the cluster (0..K)
        }


        /* * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        * Grid-level reduction to count pixels and calculate sum *
        * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

        // Total counts and sums for each cluster in the block added to the global counts and sums
        atomicAdd(&counts[threadIdx.x], local_block_cluster_count);
        atomicAdd(&sums[threadIdx.x].x, local_block_cluster_sum.x);
        atomicAdd(&sums[threadIdx.x].y, local_block_cluster_sum.y);
        atomicAdd(&sums[threadIdx.x].z, local_block_cluster_sum.z);
    }
}

/*
* CUDA kernel to update centroids
* This kernel calculates the new centroids by dividing the sum of pixel colors
* by the number of pixels in each cluster.
* The centroids array stores the new centroid colors.
* * * * */
__global__ void updateCentroids(float3* centroids, float3* sums, int* counts) {

    int k = threadIdx.x;
    if (k >= K) return;

    if (counts[k] > 0) {
        centroids[k].x = sums[k].x / counts[k];
        centroids[k].y = sums[k].y / counts[k];
        centroids[k].z = sums[k].z / counts[k];
    } else {
        // Optional: Handle case where a cluster becomes empty
        // Reinitialize randomly or leave as is
        // centroids[k] = make_float3(rand() % 256, rand() % 256, rand() % 256); // Example reinitialization
    }
}

/*
*  K-means clustering function (host)
*  This function performs the K-means clustering algorithm on the GPU.
*  It assigns clusters to each pixel, calculates the sums and counts of pixels
*  in each cluster, and updates the centroids iteratively.
* * * * */
void runKMeansCUDA(uchar3* device_input, uchar3* device_output, float3* device_centroids, int* device_labels,
                   int* device_counts, float3* device_sums, int totalPixels, cudaStream_t stream) {

    dim3 block(BLOCK_SIZE); // Threads per block
    dim3 grid((totalPixels + block.x - 1) / block.x); // Blocks per grid

    float3 previous_centroids[K], current_centroids[K];

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Copy centroids before update for convergence check
        // Using synchronous copy here for simplicity in convergence check logic
        checkCudaErrors(cudaMemcpy(previous_centroids, device_centroids, K * sizeof(float3), cudaMemcpyDeviceToHost));

        // Assign clusters
        assignClusters<<<grid, block, 0, stream>>>(device_input, device_centroids, device_labels, totalPixels);
        checkCudaErrors(cudaGetLastError());

        // Reset counts and sums
        checkCudaErrors(cudaMemsetAsync(device_counts, 0, K * sizeof(int), stream));
        checkCudaErrors(cudaMemsetAsync(device_sums, 0, K * sizeof(float3), stream));

        // Calculate sums and counts
        countAndSum<<<grid, block, 0, stream>>>(device_input, device_labels, device_counts, device_sums, totalPixels);
        checkCudaErrors(cudaGetLastError());

        // Update centroids
        updateCentroids<<<1, K, 0, stream>>>(device_centroids, device_sums, device_counts);
        checkCudaErrors(cudaGetLastError());

        // Copy updated centroids to host for convergence check
        checkCudaErrors(cudaMemcpy(current_centroids, device_centroids, K * sizeof(float3), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaStreamSynchronize(stream)); // Ensure all previous async operations are done before checking

        /*
        *  Check for convergence
        * * * * */
        float max_delta = 0.0f; // Maximum change in centroids
        for (int k = 0; k < K; k++) {
            float dx = current_centroids[k].x - previous_centroids[k].x;
            float dy = current_centroids[k].y - previous_centroids[k].y;
            float dz = current_centroids[k].z - previous_centroids[k].z;
            float dist2 = dx * dx + dy * dy + dz * dz; // Squared distance
            max_delta = fmaxf(max_delta, dist2);      // Update max_delta
        }

        // Break if converged
        if (max_delta < CONVERGENCE_THRESHOLD * CONVERGENCE_THRESHOLD) { // Compare squared distance
            // std::cout << "KMeans converged after " << iter + 1 << " iterations." << std::endl;
            break;
        }
    }

    // Apply final colors to output image
    applyColors<<<grid, block, 0, stream>>>(device_output, device_centroids, device_labels, totalPixels);
    checkCudaErrors(cudaGetLastError());
}


/********************************************************************
*
*                        EDGE DETECTION (SOBEL)
*
*****************************************************************/

__global__ void sobelKernel(const uchar3* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check image boundaries
    if (x >= width || y >= height) return;

    // Sobel operators
    const int Gx[3][3] = {
        { -1, 0, 1 },
        { -2, 0, 2 },
        { -1, 0, 1 }
    };

    const int Gy[3][3] = {
        { -1, -2, -1 },
        {  0,  0,  0 },
        {  1,  2,  1 }
    };

    float sumX = 0.0f;
    float sumY = 0.0f;

    // Apply convolution
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            // Clamp coordinates to handle borders
            int currentX = min(max(x + i, 0), width - 1);
            int currentY = min(max(y + j, 0), height - 1);

            // Read pixel and convert to grayscale (average)
            uchar3 pixel = input[currentY * width + currentX];
            float gray = (float)(pixel.x + pixel.y + pixel.z) / 3.0f;

            // Apply Sobel operator weights
            sumX += gray * Gx[j + 1][i + 1];
            sumY += gray * Gy[j + 1][i + 1];
        }
    }

    // Calculate magnitude
    float magnitude = sqrtf(sumX * sumX + sumY * sumY);

    // Clamp magnitude to 0-255 and write to output
    output[y * width + x] = (unsigned char)min(max(magnitude, 0.0f), 255.0f);
}

void applyEdgeDetectionCUDA(const cv::Mat& input, cv::Mat& output, cudaStream_t stream = 0) {
    if (input.empty() || input.channels() != 3) {
        std::cerr << "Edge Detection requires a 3-channel BGR image." << std::endl;
        output = cv::Mat::zeros(input.size(), CV_8UC1); // Return black image on error
        return;
    }
    if (!input.isContinuous()) {
        std::cerr << "Warning: Input matrix for Edge Detection is not continuous. Performance might be affected." << std::endl;
        // Optionally, clone to make it continuous: input = input.clone();
    }


    int width = input.cols;
    int height = input.rows;
    size_t inputSize = width * height * sizeof(uchar3);
    size_t outputSize = width * height * sizeof(unsigned char);

    uchar3* d_input;
    unsigned char* d_output;

    checkCudaErrors(cudaMalloc(&d_input, inputSize));
    checkCudaErrors(cudaMalloc(&d_output, outputSize));

    checkCudaErrors(cudaMemcpyAsync(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice, stream));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    sobelKernel <<<grid, block, 0, stream >>> (d_input, d_output, width, height);
    checkCudaErrors(cudaGetLastError());

    // Ensure output Mat is created *before* copying data back
    output.create(height, width, CV_8UC1);
    checkCudaErrors(cudaMemcpyAsync(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost, stream));

    // Note: Normalization happens after stream synchronization in the main loop if needed.

    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
}


/**********************************************************************
*
*                      OTSU'S THRESHOLDING
*
**************************************************************************/

// Kernel 1: Calculate Histogram
__global__ void histogramKernel(const unsigned char* inputGray, unsigned int* histogram, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixelIndex = y * width + x;
        unsigned char grayValue = inputGray[pixelIndex];
        // Atomically increment the histogram bin corresponding to the pixel's gray value
        atomicAdd(&histogram[grayValue], 1);
    }
}

// Kernel 2: Apply Threshold
__global__ void applyThresholdKernel(const unsigned char* inputGray, unsigned char* outputBinary, int width, int height, unsigned char threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixelIndex = y * width + x;
        unsigned char grayValue = inputGray[pixelIndex];
        outputBinary[pixelIndex] = (grayValue > threshold) ? 255 : 0;
    }
}

// --- Host Function to Calculate Otsu Threshold ---
int calculateOtsuThresholdCPU(const std::vector<unsigned int>& histogram, int totalPixels) {
    if (totalPixels <= 0) return 0; // Avoid division by zero

    // Calculate weighted sum of intensities (needed for means)
    double sum = 0; // Use double for potentially large sums
    for (int i = 0; i < 256; ++i) {
        sum += (double)i * histogram[i];
    }

    double sumB = 0;          // Weighted sum for background
    long long wB = 0;         // Weight (pixel count) for background (use long long)
    long long wF = 0;         // Weight (pixel count) for foreground
    double maxVariance = 0;
    int optimalThreshold = 0;

    for (int t = 0; t < 256; ++t) {
        wB += histogram[t]; // Add pixels with intensity t to background weight
        if (wB == 0) continue; // Skip if no pixels in background

        wF = (long long)totalPixels - wB; // Foreground weight
        if (wF == 0) break;    // Skip if no pixels in foreground (rest will be the same)

        sumB += (double)t * histogram[t]; // Add weighted intensity for background

        double mB = sumB / wB; // Mean of background
        double mF = (sum - sumB) / wF; // Mean of foreground

        // Calculate between-class variance
        double varianceBetween = (double)wB * (double)wF * (mB - mF) * (mB - mF);

        // Check if new maximum found
        if (varianceBetween > maxVariance) {
            maxVariance = varianceBetween;
            optimalThreshold = t;
        }
    }

    return optimalThreshold;
}


// --- Host Function to Orchestrate CUDA Otsu ---
void applyOtsuThresholdingCUDA(const cv::Mat& input, cv::Mat& output, cudaStream_t stream = 0) {
    if (input.empty()) {
        std::cerr << "Input image for Otsu is empty!" << std::endl;
        output = cv::Mat::zeros(input.size(), CV_8UC1);
        return;
    }
    if (input.channels() != 3 && input.channels() != 1) {
        std::cerr << "Input image for Otsu must be 1-channel grayscale or 3-channel color." << std::endl;
         output = cv::Mat::zeros(input.size(), CV_8UC1);
        return;
    }

    cv::Mat inputGray;
    // Convert to grayscale if necessary
    if (input.channels() == 3) {
        cv::cvtColor(input, inputGray, cv::COLOR_BGR2GRAY);
    }
    else {
        inputGray = input.clone(); // Clone to ensure we have a separate copy if input was already grayscale
    }

    if (!inputGray.isContinuous()) {
        std::cerr << "Warning: Grayscale input matrix for Otsu is not continuous. Performance might be affected." << std::endl;
        inputGray = inputGray.clone(); // Make continuous
    }

    int width = inputGray.cols;
    int height = inputGray.rows;
    int totalPixels = width * height;
    size_t grayImageSize = width * height * sizeof(unsigned char);
    size_t histogramSize = 256 * sizeof(unsigned int);

    // --- Device Memory Allocation ---
    unsigned char* d_inputGray;
    unsigned char* d_outputBinary;
    unsigned int* d_histogram;

    checkCudaErrors(cudaMalloc(&d_inputGray, grayImageSize));
    checkCudaErrors(cudaMalloc(&d_outputBinary, grayImageSize)); // Output is also grayscale (binary)
    checkCudaErrors(cudaMalloc(&d_histogram, histogramSize));

    // Initialize histogram bins to zero on the device
    checkCudaErrors(cudaMemsetAsync(d_histogram, 0, histogramSize, stream));

    // --- Data Transfer: Host -> Device ---
    checkCudaErrors(cudaMemcpyAsync(d_inputGray, inputGray.ptr(), grayImageSize, cudaMemcpyHostToDevice, stream));

    // --- Kernel Launch Configuration ---
    dim3 blockDim(16, 16); // 256 threads per block
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // --- Kernel 1: Calculate Histogram ---
    histogramKernel <<<gridDim, blockDim, 0, stream >>> (d_inputGray, d_histogram, width, height);
    checkCudaErrors(cudaGetLastError());

    // --- Data Transfer: Device -> Host (Histogram only) ---
    // Need to synchronize before histogram copy and CPU calculation
    checkCudaErrors(cudaStreamSynchronize(stream));
    std::vector<unsigned int> h_histogram(256);
    checkCudaErrors(cudaMemcpy(h_histogram.data(), d_histogram, histogramSize, cudaMemcpyDeviceToHost));

    // --- CPU Calculation: Find Otsu Threshold ---
    int otsuThreshold = calculateOtsuThresholdCPU(h_histogram, totalPixels);
    // std::cout << "Calculated Otsu Threshold: " << otsuThreshold << std::endl; // Optional debug output

    // --- Kernel 2: Apply Threshold ---
    // Reuse the same grid/block dimensions
    applyThresholdKernel <<<gridDim, blockDim, 0, stream >>> (d_inputGray, d_outputBinary, width, height, (unsigned char)otsuThreshold);
    checkCudaErrors(cudaGetLastError());

    // --- Data Transfer: Device -> Host (Final Binary Image) ---
    output.create(height, width, CV_8UC1); // Ensure output Mat is correctly sized and typed
    checkCudaErrors(cudaMemcpyAsync(output.ptr(), d_outputBinary, grayImageSize, cudaMemcpyDeviceToHost, stream));

    // --- Free Device Memory ---
    // Use cudaFreeAsync if supported and desired, otherwise sync before freeing is safer
    checkCudaErrors(cudaStreamSynchronize(stream)); // Ensure kernels are done before freeing
    checkCudaErrors(cudaFree(d_histogram));
    checkCudaErrors(cudaFree(d_outputBinary));
    checkCudaErrors(cudaFree(d_inputGray));
}


/**********************************************************************
*
*                      GAUSSIAN BLUR
*
**************************************************************************/

// --- Device function to clamp coordinates ---
__device__ inline int clamp(int v, int lo, int hi) {
    return max(lo, min(v, hi));
}

// --- Gaussian Blur Kernel ---
// Uses shared memory for caching input pixels to reduce global memory reads
__global__ void gaussianBlurKernel(const uchar3* input, uchar3* output, int width, int height, const float* kernelWeights) {

    // Shared memory tile size includes halo for kernel radius
    const int tileDimX = GAUSS_BLOCK_DIM_X + 2 * GAUSS_KERNEL_RADIUS;
    const int tileDimY = GAUSS_BLOCK_DIM_Y + 2 * GAUSS_KERNEL_RADIUS;
    __shared__ uchar3 tile[tileDimY][tileDimX];

    // Global thread indices
    int gX = blockIdx.x * blockDim.x + threadIdx.x;
    int gY = blockIdx.y * blockDim.y + threadIdx.y;

    // Local thread indices within the block
    int tX = threadIdx.x;
    int tY = threadIdx.y;

    // Calculate coordinates for loading into shared memory, including halo
    int loadX = blockIdx.x * blockDim.x + tX - GAUSS_KERNEL_RADIUS;
    int loadY = blockIdx.y * blockDim.y + tY - GAUSS_KERNEL_RADIUS;

    // Load pixel into the center of the thread's responsibility in the tile
    int tileX = tX + GAUSS_KERNEL_RADIUS;
    int tileY = tY + GAUSS_KERNEL_RADIUS;

    // Clamp global coordinates for reading input image (border handling)
    int clampedLoadX = clamp(loadX, 0, width - 1);
    int clampedLoadY = clamp(loadY, 0, height - 1);

    // Load pixel from global memory to shared memory
    tile[tileY][tileX] = input[clampedLoadY * width + clampedLoadX];

    // Handle loading the halo regions (requires additional loads if blockDim < halo size)
    // Simplified approach: Each thread loads its corresponding pixel + potential halo pixels
    // More efficient: Load halo in separate steps or by threads at the edges

    // Load left halo
    if (tX < GAUSS_KERNEL_RADIUS) {
        clampedLoadX = clamp(loadX - GAUSS_BLOCK_DIM_X, 0, width - 1); // Load from the block to the left
        tile[tileY][tX] = input[clampedLoadY * width + clampedLoadX];
    }
    // Load right halo
    if (tX >= blockDim.x - GAUSS_KERNEL_RADIUS) {
         clampedLoadX = clamp(loadX + GAUSS_BLOCK_DIM_X, 0, width - 1); // Load from the block to the right
         tile[tileY][tX + 2 * GAUSS_KERNEL_RADIUS] = input[clampedLoadY * width + clampedLoadX]; // ERROR corrected index
    }
    // Load top halo
    if (tY < GAUSS_KERNEL_RADIUS) {
        clampedLoadY = clamp(loadY - GAUSS_BLOCK_DIM_Y, 0, height - 1); // Load from the block above
        tile[tY][tileX] = input[clampedLoadY * width + clampedLoadX]; // Use original clampedLoadX
    }
    // Load bottom halo
     if (tY >= blockDim.y - GAUSS_KERNEL_RADIUS) {
        clampedLoadY = clamp(loadY + GAUSS_BLOCK_DIM_Y, 0, height - 1); // Load from the block below
        tile[tY + 2 * GAUSS_KERNEL_RADIUS][tileX] = input[clampedLoadY * width + clampedLoadX]; // ERROR corrected index
    }

    // TODO: Load corner halo regions if necessary (often less critical, depends on kernel size vs block size)

    // Synchronize to ensure all data is loaded into shared memory
    __syncthreads();

    // --- Perform convolution using shared memory ---
    if (gX < width && gY < height) {
        float3 sum = make_float3(0.0f, 0.0f, 0.0f);
        int kernelIndex = 0;

        for (int j = -GAUSS_KERNEL_RADIUS; j <= GAUSS_KERNEL_RADIUS; ++j) {
            for (int i = -GAUSS_KERNEL_RADIUS; i <= GAUSS_KERNEL_RADIUS; ++i) {
                // Read pixel from shared memory tile
                uchar3 pixel = tile[tileY + j][tileX + i];
                float weight = kernelWeights[kernelIndex++];

                // Accumulate weighted sum
                sum.x += (float)pixel.x * weight; // Blue
                sum.y += (float)pixel.y * weight; // Green
                sum.z += (float)pixel.z * weight; // Red
            }
        }

        // Clamp result to 0-255 and write to global output memory
        output[gY * width + gX] = make_uchar3(
            (unsigned char)clamp((int)(sum.x + 0.5f), 0, 255), // Add 0.5f for rounding
            (unsigned char)clamp((int)(sum.y + 0.5f), 0, 255),
            (unsigned char)clamp((int)(sum.z + 0.5f), 0, 255)
        );
    }
}


// --- Host function to generate Gaussian kernel weights ---
void generateGaussianKernel(std::vector<float>& kernel, int size, float sigma) {
    if (size % 2 == 0) {
        std::cerr << "Gaussian kernel size must be odd!" << std::endl;
        size++; // Force odd size
    }
    kernel.resize(size * size);
    int radius = size / 2;
    float sum = 0.0f;
    float sigmaSq = 2.0f * sigma * sigma;
    float piSigmaSq = M_PI * sigmaSq; // M_PI defined in cmath

    int kIndex = 0;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            float exponent = -(float)(x * x + y * y) / sigmaSq;
            kernel[kIndex] = expf(exponent) / piSigmaSq; // Use expf for float
            sum += kernel[kIndex];
            kIndex++;
        }
    }

    // Normalize the kernel
    kIndex = 0;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            kernel[kIndex++] /= sum;
        }
    }
}

// --- Host Function to Orchestrate CUDA Gaussian Blur ---
void applyGaussianBlurCUDA(const cv::Mat& input, cv::Mat& output, int kernelSize, float sigma, cudaStream_t stream = 0) {
     if (input.empty() || input.channels() != 3) {
        std::cerr << "Gaussian Blur requires a 3-channel BGR image." << std::endl;
        output = input.clone(); // Return original on error
        return;
    }
     if (!input.isContinuous()) {
        std::cerr << "Warning: Input matrix for Gaussian Blur is not continuous. Performance might be affected." << std::endl;
        // input = input.clone(); // Consider cloning if needed
    }


    int width = input.cols;
    int height = input.rows;
    size_t imageSize = width * height * sizeof(uchar3);
    size_t kernelSizeBytes = kernelSize * kernelSize * sizeof(float);

    // --- Generate Gaussian Kernel Weights on CPU ---
    std::vector<float> h_kernelWeights;
    generateGaussianKernel(h_kernelWeights, kernelSize, sigma);

    // --- Device Memory Allocation ---
    uchar3* d_input;
    uchar3* d_output;
    float* d_kernelWeights;

    checkCudaErrors(cudaMalloc(&d_input, imageSize));
    checkCudaErrors(cudaMalloc(&d_output, imageSize)); // Output is also 3-channel
    checkCudaErrors(cudaMalloc(&d_kernelWeights, kernelSizeBytes));

    // --- Data Transfer: Host -> Device ---
    checkCudaErrors(cudaMemcpyAsync(d_input, input.ptr(), imageSize, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_kernelWeights, h_kernelWeights.data(), kernelSizeBytes, cudaMemcpyHostToDevice, stream));

    // --- Kernel Launch Configuration ---
    // Use specific block dimensions defined for Gaussian Blur
    dim3 blockDim(GAUSS_BLOCK_DIM_X, GAUSS_BLOCK_DIM_Y);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // --- Launch Kernel ---
    gaussianBlurKernel<<<gridDim, blockDim, 0, stream>>>(d_input, d_output, width, height, d_kernelWeights);
    checkCudaErrors(cudaGetLastError()); // Check for launch errors

    // --- Data Transfer: Device -> Host ---
    output.create(height, width, CV_8UC3); // Ensure output Mat is correctly sized and typed
    checkCudaErrors(cudaMemcpyAsync(output.ptr(), d_output, imageSize, cudaMemcpyDeviceToHost, stream));

    // --- Free Device Memory ---
    // Sync before free recommended if not using cudaFreeAsync
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaFree(d_kernelWeights));
    checkCudaErrors(cudaFree(d_output));
    checkCudaErrors(cudaFree(d_input));
}


/**********************************************************************
*
*                      MAIN APPLICATION
*
**************************************************************************/

int main() {

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);// Suppress OpenCV logs
    cv::VideoCapture capture(0);// Open default camera
    if (!capture.isOpened()) {
         std::cerr << "Error: Could not open camera!" << std::endl;
         return -1;
    }

    cv::Mat frame, processedFrame, combined, originalFrame;
    cv::Mat edges, edgesColor; // For Sobel
    cv::Mat otsuBinary, otsuColor; // For Otsu
    cv::Mat blurredFrame; // For Gaussian Blur

    capture >> frame;// Capture initial frame to get dimensions
    if (frame.empty()) {
        std::cerr << "Error: Captured empty frame!" << std::endl;
        capture.release();
        return -1;
    }
    const int width = frame.cols;
    const int height = frame.rows;
    const int totalPixels = height * width;// Total pixels in the frame

    cudaStream_t stream;// CUDA stream for asynchronous operations
    checkCudaErrors(cudaStreamCreate(&stream));// Create the stream

    /*
    *  Allocate common device (GPU) memory needed across filters
    *  Filters might allocate their specific buffers inside their wrapper functions
    * * * * */
    uchar3* device_input = nullptr; // Input image (common)
    uchar3* device_output_kmeans = nullptr; // Output specific to KMeans
    float3* device_centroids = nullptr; // Centroids for K-means
    int* device_labels = nullptr;    // Cluster labels for each pixel (KMeans)
    int* device_counts = nullptr;    // Counts of pixels in each cluster (KMeans)
    float3* device_sums = nullptr;    // Sums of pixel colors for each cluster (KMeans)

    // Allocate memory used commonly or specifically by KMeans
    checkCudaErrors(cudaMalloc(&device_input, totalPixels * sizeof(uchar3)));
    checkCudaErrors(cudaMalloc(&device_output_kmeans, totalPixels * sizeof(uchar3)));
    checkCudaErrors(cudaMalloc(&device_centroids, K * sizeof(float3)));
    checkCudaErrors(cudaMalloc(&device_labels, totalPixels * sizeof(int)));
    checkCudaErrors(cudaMalloc(&device_counts, K * sizeof(int)));
    checkCudaErrors(cudaMalloc(&device_sums, K * sizeof(float3)));


    /*
    *  Initialize KMeans centroids with random colors
    *  And copy them to device memory
    * * * * */
    float3 host_centroids[K];// Host array for centroids
    std::srand(static_cast<unsigned int>(std::time(nullptr))); // Seed RNG
    for (int k = 0; k < K; k++) {
        host_centroids[k].x = float(std::rand() % 256);
        host_centroids[k].y = float(std::rand() % 256);
        host_centroids[k].z = float(std::rand() % 256);
    }
    checkCudaErrors(cudaMemcpy(device_centroids, host_centroids, K * sizeof(float3), cudaMemcpyHostToDevice));// Copy centroids to device

    FilterType filter = FilterType::KMeans; // Default filter
    std::cout << "Press 1: KMeans, 2: Sobel, 3: Otsu, 4: GaussianBlur, ESC/e: Exit\n"; // Updated help text
    std::cout << "Current Filter: " << filter << '\n';

    // Main loop for processing frames
    while (true) {

        capture >> frame;
        if (frame.empty()) {
            std::cerr << "Warning: Captured empty frame during loop." << std::endl;
            break;
        }

        // Ensure frame is continuous for CUDA memory copies
        if (!frame.isContinuous()) {
            frame = frame.clone();
        }
        // Keep a copy of the original frame before processing
        originalFrame = frame.clone();

        // --- Select and Apply Filter ---
        switch (filter) {
        case FilterType::KMeans: {
            // Copy frame's data to the common device input buffer
            checkCudaErrors(cudaMemcpyAsync(device_input, frame.data, totalPixels * sizeof(uchar3), cudaMemcpyHostToDevice, stream));

            // Perform K-means clustering
            runKMeansCUDA(device_input, device_output_kmeans, device_centroids, device_labels, device_counts, device_sums, totalPixels, stream);

            // Copy KMeans output image back to host Mat `processedFrame`
            processedFrame.create(height, width, CV_8UC3); // Ensure size/type
            checkCudaErrors(cudaMemcpyAsync(processedFrame.data, device_output_kmeans, totalPixels * sizeof(uchar3), cudaMemcpyDeviceToHost, stream));
            checkCudaErrors(cudaStreamSynchronize(stream)); // Wait for KMeans operations

            cv::hconcat(originalFrame, processedFrame, combined);
            break;
        }
        case FilterType::EdgeDetection: {
            // Sobel wrapper handles its own allocation/transfers
            applyEdgeDetectionCUDA(originalFrame, edges, stream); // Use originalFrame as input
            checkCudaErrors(cudaStreamSynchronize(stream)); // Wait for Sobel operations

            // Convert grayscale edge image to color for concatenation
            cv::cvtColor(edges, edgesColor, cv::COLOR_GRAY2BGR);

            // Combine original and edge-detected images
            cv::hconcat(originalFrame, edgesColor, combined);
            break;
        }
        case FilterType::Otsu: {
            // Otsu wrapper handles its own allocation/transfers
            applyOtsuThresholdingCUDA(originalFrame, otsuBinary, stream); // Use originalFrame as input
            checkCudaErrors(cudaStreamSynchronize(stream)); // Wait for Otsu operations

            // Prepare for display: Convert binary image to color BGR
            cv::cvtColor(otsuBinary, otsuColor, cv::COLOR_GRAY2BGR);

            // Combine original and Otsu images side-by-side
            cv::hconcat(originalFrame, otsuColor, combined);
            break;
        }
        case FilterType::GaussianBlur: { // Added Gaussian Blur case
            // Gaussian Blur wrapper handles its own allocation/transfers
            applyGaussianBlurCUDA(originalFrame, blurredFrame, GAUSS_KERNEL_SIZE, GAUSS_SIGMA, stream); // Use originalFrame as input
            checkCudaErrors(cudaStreamSynchronize(stream)); // Wait for blur operations

            // Combine original and blurred images side-by-side
            cv::hconcat(originalFrame, blurredFrame, combined);
            break;
        }
        } // End switch

        // Display the combined output image
        if (!combined.empty()){
             cv::imshow("Original | Filtered", combined);
        } else {
             cv::imshow("Original | Filtered", originalFrame); // Show original if processing failed
             std::cerr << "Warning: Combined frame is empty, showing original." << std::endl;
        }


        int key = cv::waitKey(1);  // waits 1 ms
        if (key == 27 || key == 'e') break; // Exit on ESC or 'e' key

        bool filterChanged = false;
        if (key == '1' && filter != FilterType::KMeans) {
            filter = FilterType::KMeans;
            filterChanged = true;
        } else if (key == '2' && filter != FilterType::EdgeDetection) {
            filter = FilterType::EdgeDetection;
            filterChanged = true;
        } else if (key == '3' && filter != FilterType::Otsu) {
            filter = FilterType::Otsu;
            filterChanged = true;
        } else if (key == '4' && filter != FilterType::GaussianBlur) { // Added key '4' for Gaussian Blur
            filter = FilterType::GaussianBlur;
            filterChanged = true;
        }

        if(filterChanged) {
            std::cout << "Current Filter: " << filter << '\n';
        }
    } // End while loop

    /*
    *  Free common/KMeans device memory
    * * * * */
    checkCudaErrors(cudaStreamSynchronize(stream)); // Ensure stream is finished before freeing
    checkCudaErrors(cudaFree(device_input));
    checkCudaErrors(cudaFree(device_output_kmeans));
    checkCudaErrors(cudaFree(device_centroids));
    checkCudaErrors(cudaFree(device_labels));
    checkCudaErrors(cudaFree(device_counts));
    checkCudaErrors(cudaFree(device_sums));
    checkCudaErrors(cudaStreamDestroy(stream));

    capture.release();
    cv::destroyAllWindows();
    checkCudaErrors(cudaDeviceReset()); // Clean up CUDA context

    return 0;
}
