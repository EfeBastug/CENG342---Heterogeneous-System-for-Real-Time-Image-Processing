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


#define K 6 // Number of clusters
#define MAX_ITER 10 // Maximum iterations
#define BLOCK_SIZE 256 // Threads per block
#define WARPS_PER_BLOCK  (BLOCK_SIZE / 32) // Number of warps per block
#define CONVERGENCE_THRESHOLD 0.01f // Convergence threshold


enum class FilterType {
    KMeans, 
    EdgeDetection, 
    Otsu
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
    }
    return os;
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
    output[idx] = make_uchar3(centroids[cluster].x, centroids[cluster].y, centroids[cluster].z);
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
    }
}

/*
*  K-means clustering kernel
*  This function performs the K-means clustering algorithm on the GPU.
*  It assigns clusters to each pixel, calculates the sums and counts of pixels
*  in each cluster, and updates the centroids.
* * * * */
void kmeansCUDA(uchar3* device_input, float3* device_centroids, int* device_labels,
    int* device_counts, float3* device_sums, int totalPixels, cudaStream_t stream) {

    dim3 block(BLOCK_SIZE); // Threads per block
    dim3 grid((totalPixels + block.x - 1) / block.x); // Blocks per grid

    // Assign clusters
    assignClusters << <grid, block, 0, stream >> > (device_input, device_centroids, device_labels, totalPixels);

    // Reset counts and sums
    cudaMemsetAsync(device_counts, 0, K * sizeof(int), stream);
    cudaMemsetAsync(device_sums, 0, K * sizeof(float3), stream);

    // Calculate sums and counts
    countAndSum << <grid, block, 0, stream >> > (device_input, device_labels, device_counts, device_sums, totalPixels);

    // Update centroids
    updateCentroids << <1, K, 0, stream >> > (device_centroids, device_sums, device_counts);
}




/********************************************************************
*
*                        EDGE DETECTION
*
*****************************************************************/

__global__ void sobelKernel(uchar3* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int Gx[3][3] = {
        { -1, 0, 1 },
        { -2, 0, 2 },
        { -1, 0, 1 }
    };

    int Gy[3][3] = {
        { -1, -2, -1 },
        {  0,  0,  0 },
        {  1,  2,  1 }
    };

    int sumX = 0, sumY = 0;

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            int nx = min(max(x + dx, 0), width - 1);
            int ny = min(max(y + dy, 0), height - 1);
            uchar3 pixel = input[ny * width + nx];
            int gray = (pixel.x + pixel.y + pixel.z) / 3;

            sumX += gray * Gx[dy + 1][dx + 1];
            sumY += gray * Gy[dy + 1][dx + 1];
        }
    }

    int magnitude = sqrtf((float)(sumX * sumX + sumY * sumY));
    magnitude = min(255, magnitude);
    output[y * width + x] = (unsigned char)magnitude;
}

void applyEdgeDetectionCUDA(const cv::Mat& input, cv::Mat& output) {
    int width = input.cols;
    int height = input.rows;
    size_t inputSize = width * height * sizeof(uchar3);
    size_t outputSize = width * height * sizeof(unsigned char);

    uchar3* d_input;
    unsigned char* d_output;

    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);

    cudaMemcpy(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    sobelKernel << <grid, block >> > (d_input, d_output, width, height);
    cudaDeviceSynchronize();

    output.create(height, width, CV_8UC1);
    cudaMemcpy(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);

    // Normalize the result to 0-255 range for better contrast
    cv::normalize(output, output, 0, 255, cv::NORM_MINMAX);

    cudaFree(d_input);
    cudaFree(d_output);
}


/**********************************************************************
*
*                      OTSU'S THRESHOLDING
*
**************************************************************************/


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

// --- CUDA Kernels ---

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
    float sum = 0;
    for (int i = 0; i < 256; ++i) {
        sum += (float)i * histogram[i];
    }

    float sumB = 0;          // Weighted sum for background
    int wB = 0;              // Weight (pixel count) for background
    int wF = 0;              // Weight (pixel count) for foreground
    float maxVariance = 0;
    int optimalThreshold = 0;

    for (int t = 0; t < 256; ++t) {
        wB += histogram[t]; // Add pixels with intensity t to background weight
        if (wB == 0) continue; // Skip if no pixels in background

        wF = totalPixels - wB; // Foreground weight
        if (wF == 0) break;    // Skip if no pixels in foreground (rest will be the same)

        sumB += (float)t * histogram[t]; // Add weighted intensity for background

        float mB = sumB / wB; // Mean of background
        float mF = (sum - sumB) / wF; // Mean of foreground

        // Calculate between-class variance
        float varianceBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);

        // Check if new maximum found
        if (varianceBetween > maxVariance) {
            maxVariance = varianceBetween;
            optimalThreshold = t;
        }
    }

    return optimalThreshold;
}


// --- Host Function to Orchestrate CUDA Otsu ---
void applyOtsuThresholdingCUDA(const cv::Mat& input, cv::Mat& output) {
    if (input.empty()) {
        std::cerr << "Input image is empty!" << std::endl;
        return;
    }
    if (input.channels() != 3 && input.channels() != 1) {
        std::cerr << "Input image must be 1-channel grayscale or 3-channel color." << std::endl;
        return;
    }

    cv::Mat inputGray;
    // Convert to grayscale if necessary
    if (input.channels() == 3) {
        cv::cvtColor(input, inputGray, cv::COLOR_BGR2GRAY);
    }
    else {
        inputGray = input;
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
    checkCudaErrors(cudaMemset(d_histogram, 0, histogramSize));

    // --- Data Transfer: Host -> Device ---
    checkCudaErrors(cudaMemcpy(d_inputGray, inputGray.ptr(), grayImageSize, cudaMemcpyHostToDevice));

    // --- Kernel Launch Configuration ---
    dim3 blockDim(16, 16); // 256 threads per block
    // Adjust grid dimensions to cover the entire image
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // --- Kernel 1: Calculate Histogram ---
    histogramKernel << <gridDim, blockDim >> > (d_inputGray, d_histogram, width, height);
    checkCudaErrors(cudaGetLastError()); // Check for kernel launch errors
    checkCudaErrors(cudaDeviceSynchronize()); // Wait for histogram kernel to finish

    // --- Data Transfer: Device -> Host (Histogram only) ---
    std::vector<unsigned int> h_histogram(256);
    checkCudaErrors(cudaMemcpy(h_histogram.data(), d_histogram, histogramSize, cudaMemcpyDeviceToHost));

    // --- CPU Calculation: Find Otsu Threshold ---
    int otsuThreshold = calculateOtsuThresholdCPU(h_histogram, totalPixels);
    // std::cout << "Calculated Otsu Threshold: " << otsuThreshold << std::endl; // Optional debug output

    // --- Kernel 2: Apply Threshold ---
    // Reuse the same grid/block dimensions
    applyThresholdKernel << <gridDim, blockDim >> > (d_inputGray, d_outputBinary, width, height, (unsigned char)otsuThreshold);
    checkCudaErrors(cudaGetLastError()); // Check for kernel launch errors
    checkCudaErrors(cudaDeviceSynchronize()); // Wait for threshold application kernel to finish

    // --- Data Transfer: Device -> Host (Final Binary Image) ---
    output.create(height, width, CV_8UC1); // Ensure output Mat is correctly sized and typed
    checkCudaErrors(cudaMemcpy(output.ptr(), d_outputBinary, grayImageSize, cudaMemcpyDeviceToHost));

    // --- Free Device Memory ---
    checkCudaErrors(cudaFree(d_histogram));
    checkCudaErrors(cudaFree(d_outputBinary));
    checkCudaErrors(cudaFree(d_inputGray));
}



int main() {

	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);// Suppress OpenCV logs
    cv::VideoCapture capture(0);// Open default camera
    if (!capture.isOpened()) return -1;

    cv::Mat frame, edges, combined, frameGray, otsuBinary, edgesColor;
    capture >> frame;// Capture frame
    const int totalPixels = frame.rows * frame.cols;// Total pixels in the frame

    cudaStream_t stream;// CUDA stream for asynchronous operations
    cudaStreamCreate(&stream);// Create the stream

    /*
    *  Allocate device (GPU) memory
    *  uchar3 and float3 are CUDA types for 3-channel uchar and float vectors
    * * * * */

    uchar3* device_input;// Input image
    uchar3* device_output;// Output image
    float3* device_centroids;// Centroids for K-means
    int* device_labels;// Cluster labels for each pixel
    int* device_counts;// Counts of pixels in each cluster
    float3* device_sums;// Sums of pixel colors for each cluster

    cudaMalloc(&device_input, totalPixels * sizeof(uchar3));
    cudaMalloc(&device_output, totalPixels * sizeof(uchar3));
    cudaMalloc(&device_centroids, K * sizeof(float3));
    cudaMalloc(&device_labels, totalPixels * sizeof(int));
    cudaMalloc(&device_counts, K * sizeof(int));
    cudaMalloc(&device_sums, K * sizeof(float3));

    /*
    *  Inıtialize centroids with random colors
    *  And copy them to device memory
    * * * * */

    float3 host_centroids[K];// Host array for centroids

    std::srand(std::time(nullptr));

    for (int k = 0; k < K; k++) {
        host_centroids[k].x = float(std::rand() % 256);
        host_centroids[k].y = float(std::rand() % 256);
        host_centroids[k].z = float(std::rand() % 256);
    }

    cudaMemcpy(device_centroids, host_centroids, K * sizeof(float3), cudaMemcpyHostToDevice);// Copy centroids to device

    FilterType filter = FilterType::KMeans;
    std::cout << "Press 1: KMeans, 2: Sobel, 3: Otsu, ESC: Exit\n";
    std::cout << "Current Filter: " << filter << '\n';
    // Main loop for processing frames
    while (true) {

        capture >> frame;
        if (frame.empty()) break;

        switch (filter) {
        case FilterType::KMeans: {
           
            if (!frame.isContinuous()) {            
                frame = frame.clone();         
            }
            // Copy frame's data to device
            cudaMemcpyAsync(device_input, frame.data, totalPixels * sizeof(uchar3), cudaMemcpyHostToDevice, stream);

            float3 previous_centroids[K], current_centroids[K];

            for (int iter = 0; iter < MAX_ITER; iter++) {

                // Copy before updating centroids
                cudaMemcpyAsync(previous_centroids, device_centroids, K * sizeof(float3), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);

                // Perform K-means clustering
                kmeansCUDA(device_input, device_centroids, device_labels, device_counts, device_sums, totalPixels, stream);

                // Copy updated centroids to host
                cudaMemcpyAsync(current_centroids, device_centroids, K * sizeof(float3), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);


                /*
                *  Check for convergence
                * * * * */
                float max_delta = 0;// Maximum change in centroids

                for (int k = 0; k < K; k++) {
                    float dx = current_centroids[k].x - previous_centroids[k].x;
                    float dy = current_centroids[k].y - previous_centroids[k].y;
                    float dz = current_centroids[k].z - previous_centroids[k].z;
                    float dist2 = dx * dx + dy * dy + dz * dz;// Squared distance
                    max_delta = fmaxf(max_delta, dist2); // Update max_delta
                }
                // Break if converged
                if (max_delta < CONVERGENCE_THRESHOLD) {
                    break;
                }
            }

            // Apply colors to output image
            applyColors << <(totalPixels + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream >> > (device_output, device_centroids, device_labels, totalPixels);

            // Copy output image back to host
            cudaMemcpyAsync(frame.data, device_output, totalPixels * sizeof(uchar3), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            cv::hconcat(originalFrame, frame, combined);

            break;
        }
		case FilterType::EdgeDetection:

			applyEdgeDetectionCUDA(frame, edges);

            // Convert grayscale edge image to color for concatenation
            cv::cvtColor(edges, edgesColor, cv::COLOR_GRAY2BGR);

            // Combine original and edge-detected images
            cv::hconcat(frame, edgesColor, combined);

			break;

		case FilterType::Otsu:

            // Apply Otsu thresholding using CUDA
            applyOtsuThresholdingCUDA(frame, otsuBinary);

            // Prepare for display
            // Convert binary image to color BGR to concatenate with the original color frame
            cv::Mat otsuColor;
            cv::cvtColor(otsuBinary, otsuColor, cv::COLOR_GRAY2BGR);

            // Combine original and Otsu images side-by-side
            cv::hconcat(frame, otsuColor, combined);
			break;
        }

        // Display the output image
        cv::imshow("Original | Filtered", combined);

        int key = cv::waitKey(1);  // waits 1 ms
        if (key == 27 || key == 'e') break; // Exit on ESC or 'e' key

        if (key == '1') {
            filter = FilterType::KMeans;
            std::cout << "Current Filter: KMeans\n";
        }
        else if (key == '2') {
            filter = FilterType::EdgeDetection;
            std::cout << "Current Filter: EdgeDetection\n";
        }
        else if (key == '3') {
            filter = FilterType::Otsu;
            std::cout << "Current Filter: Otsu\n";
        }
    }

    /*
    *  Free device memory
    * * * * */
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_centroids);
    cudaFree(device_labels);
    cudaFree(device_counts);
    cudaFree(device_sums);
    cudaStreamDestroy(stream);

    capture.release();
    cv::destroyAllWindows();
    cudaDeviceReset(); // Clean up CUDA context

    return 0;
}