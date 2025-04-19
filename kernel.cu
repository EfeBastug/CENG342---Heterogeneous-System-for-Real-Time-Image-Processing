#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cfloat>
#include <cstdlib>
#include <ctime>

#define K 8 // Number of clusters
#define MAX_ITER 10 // Maximum iterations
#define BLOCK_SIZE 256 // Threads per block
#define WARPS_PER_BLOCK  (BLOCK_SIZE / 32) // Number of warps per block
#define CONVERGENCE_THRESHOLD 0.01f // Convergence threshold



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
    assignClusters <<<grid, block, 0, stream >> > (device_input, device_centroids, device_labels, totalPixels);

    // Reset counts and sums
    cudaMemsetAsync(device_counts, 0, K * sizeof(int), stream);
    cudaMemsetAsync(device_sums, 0, K * sizeof(float3), stream);

    // Calculate sums and counts
    countAndSum <<<grid, block, 0, stream >>> (device_input, device_labels, device_counts, device_sums, totalPixels);

    // Update centroids
    updateCentroids <<<1, K, 0, stream >>> (device_centroids, device_sums, device_counts);
}

int main() {

    cv::VideoCapture capture(0);// Open default camera
    if (!capture.isOpened()) return -1;

    cv::Mat frame;
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

    // Main loop for processing frames
    while (true) {

        capture >> frame;
        if (frame.empty()) break;

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
                std::cout << "Converged after " << iter << " iterations." << std::endl;
                break;
            }
        }

		// Apply colors to output image
        applyColors <<<(totalPixels + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(device_output, device_centroids, device_labels, totalPixels);

		// Copy output image back to host
        cudaMemcpyAsync(frame.data, device_output, totalPixels * sizeof(uchar3), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

		// Display the output image
        cv::imshow("CUDA K-Means", frame);

        int key = cv::waitKey(1);  // waits 1 ms
		if (key == 27 || key == 'e') break; // Exit on ESC or 'e' key

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

    return 0;
}
