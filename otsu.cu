#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <limits>

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

// --- Main Function (Similar to your example) ---
int main() {
    cv::VideoCapture cap(0); // Use 0 for default camera
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    cv::Mat frame, frameGray, otsuBinary, combined;

    std::cout << "Press ESC to quit." << std::endl;

    while (true) {
        cap >> frame; // Capture a new frame
        if (frame.empty()) {
            std::cerr << "Error: Captured empty frame." << std::endl;
            break;
        }

        // Apply Otsu thresholding using CUDA
        applyOtsuThresholdingCUDA(frame, otsuBinary);

        // Prepare for display
        // Convert binary image to color BGR to concatenate with the original color frame
        cv::Mat otsuColor;
        cv::cvtColor(otsuBinary, otsuColor, cv::COLOR_GRAY2BGR);

        // Combine original and Otsu images side-by-side
        cv::hconcat(frame, otsuColor, combined);

        // Display the combined image
        cv::imshow("Original | Otsu Thresholding (CUDA)", combined);

        // Exit loop if ESC key is pressed
        if (cv::waitKey(1) == 27) { // wait 1ms for key press
            break;
        }
    }

    // Release resources
    cap.release();
    cv::destroyAllWindows();
    cudaDeviceReset(); // Clean up CUDA context

    return 0;
}