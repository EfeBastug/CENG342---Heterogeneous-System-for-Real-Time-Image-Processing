#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

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

    sobelKernel<<<grid, block>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    output.create(height, width, CV_8UC1);
    cudaMemcpy(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);

    // Normalize the result to 0-255 range for better contrast
    cv::normalize(output, output, 0, 255, cv::NORM_MINMAX);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Could not open camera.\n";
        return -1;
    }

    cv::Mat frame, edges, combined;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        applyEdgeDetectionCUDA(frame, edges);

        // Convert grayscale edge image to color for concatenation
        cv::Mat edgesColor;
        cv::cvtColor(edges, edgesColor, cv::COLOR_GRAY2BGR);

        // Combine original and edge-detected images
        cv::hconcat(frame, edgesColor, combined);

        cv::imshow("Original | Edge Detection", combined);
        if (cv::waitKey(1) == 27) break; // ESC to quit
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}