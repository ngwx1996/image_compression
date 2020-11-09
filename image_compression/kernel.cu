#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "opencv2\opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace std;

string type2str(int type) {
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

__global__ void setClusters(float* pointVec, int* cur_cluster, float* means,
                            float* sums, int vecSize, int k, int dimensions, int gridSize, int* counter, int* done) {
    extern __shared__ float shared[];
    int* shared_count = (int*)&shared[blockDim.x * dimensions];

    int localIdx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= vecSize)
        return;

    // Use shared mem to store means
    if (localIdx < k * dimensions) {
        // Put means into shared memory to reduce time to take from global memory
        shared[threadIdx.x] = means[threadIdx.x];
        //printf("Cluster %d, axis %d has mean val %f\n", localIdx % k, localIdx / k, means[localIdx]);
    }
    __syncthreads();

    float minDist = FLT_MAX;
    int bestCluster = INT_MAX;
    float distance;

    for (int i = 0; i < k; i++) {
        distance = 0;
        for (int j = 0; j < dimensions; j++)
        {
            distance += (pointVec[idx + vecSize * j] - shared[i + k * j]) * (pointVec[idx + vecSize * j] - shared[i + k * j]);
        }
        if (distance < minDist) {
            minDist = distance;
            bestCluster = i;
        }
    }
    if (cur_cluster[idx] != bestCluster) {
        cur_cluster[idx] = bestCluster;
        done[0] = 1;
    }
    
    __syncthreads();

    for (int j = 0; j < k; j++) {
        for (int curAxis = 0; curAxis < dimensions; curAxis++) {
            shared[localIdx + curAxis * blockDim.x] = (bestCluster == j) ? pointVec[idx + vecSize * curAxis] : 0;
        }
        shared_count[localIdx] = (bestCluster == j) ? 1 : 0;
        //printf("point %d at cluster %d has val %f , %f, %f with bestcluster %d actual %d\n", idx, j, shared[localIdx], shared[localIdx + blockDim.x], shared[localIdx + 2 * blockDim.x],shared_count[localIdx], bestCluster);
        __syncthreads();

        // Reduction to get sum at tid 0
        for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
            if (localIdx < d && idx + d < vecSize) {
                for (int curAxis = 0; curAxis < dimensions; curAxis++) {
                    shared[localIdx + curAxis * blockDim.x] += shared[localIdx + curAxis * blockDim.x + d];
                }
                shared_count[localIdx] += shared_count[localIdx + d];
            }
            __syncthreads();
        }

        // Value at tid 0
        if (localIdx == 0) {
            //printf("cluster %d has shared sum %f, %f and count %d\n", j, shared[localIdx], shared[localIdx + blockDim.x], shared_count[localIdx]);

            //printf("cluster %d total count %d\n", j, shared_count[localIdx]);
            int clusterIdx = j + blockIdx.x * k;
            for (int curAxis = 0; curAxis < dimensions; curAxis++) {
                sums[clusterIdx + curAxis * k * gridSize] = shared[localIdx + curAxis * blockDim.x];
            }
            counter[clusterIdx] = shared_count[localIdx];
            //printf("cluster %d has sum %f, %f and count %d\n", j, sums[clusterIdx], sums[clusterIdx + k * gridSize], counter[clusterIdx]);
        }
        __syncthreads();
    }
}

__global__ void getNewMeans(float* means, float* sums, int* counter, int k, int dimensions) {
    extern __shared__ float shared[];
    int* shared_count = (int*)&shared[dimensions * blockDim.x];


    int idx = threadIdx.x;
    int blocks = blockDim.x / k;
    for (int curAxis = 0; curAxis < dimensions; curAxis++) {
        shared[idx + blockDim.x * curAxis] = sums[idx + blockDim.x * curAxis];
    }
    shared_count[idx] = counter[idx];
    __syncthreads();

    //printf("idx %d for cluster %d has %f , %f with count %d\n", idx, idx % 5, shared[idx], shared[idx + blockDim.x], shared_count[idx]);
    if (idx < k) {
        for (int j = 1; j < blocks; j++) {
            for (int curAxis = 0; curAxis < dimensions; curAxis++) {
                shared[idx + blockDim.x * curAxis] += shared[idx + j * k + blockDim.x * curAxis];
            }
            shared_count[idx] += shared_count[idx + j * k];
        }
    }
    __syncthreads;

    if (idx < k) {
        int count = (shared_count[idx] > 0) ? shared_count[idx] : 1;
        //printf("%d has count %d\n", idx, count);
        for (int curAxis = 0; curAxis < dimensions; curAxis++) {
            means[idx + k * curAxis] = shared[idx + blockDim.x * curAxis] / count;
            
            //printf("idx %d has sum %f , %f and count %d\n", idx, shared[idx], shared[idx + blockDim.x], count);

            sums[idx + blockDim.x * curAxis] = 0;
        }
        //printf("idx %d has means %f , %f and count %d\n", idx, means[idx], means[idx + k], count);

        counter[idx] = 0;
    }
}

__global__ void updatePixels(unsigned char* image, int* cur_cluster, float* means, int k, int rowSize, int colSize, int step) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tidImage = row * step + (3 * col);
    int cur = row * colSize + col;

    if (col < colSize && row < rowSize) {
        int cluster = cur_cluster[cur];
        //if (col == 177 && row == 342) {
        //    printf("%d, %d, cluster %d, image %u,%u,%u\n", row, col, cluster, image[tidImage], image[tidImage + 1],image[tidImage + 2]);
        //}
        image[tidImage] = (unsigned char)means[cluster];
        image[tidImage + 1] = (unsigned char)means[cluster + k];
        image[tidImage + 2] = (unsigned char)means[cluster + 2 * k];
        //if (col == 177 && row == 342) {
        //    printf("new %d, %d, cluster %d, means %u, %u, %u\n", row, col, cluster, image[tidImage], image[tidImage + 1], image[tidImage + 2]);
        //}
    }
}


cv::Mat kmeans_cuda(cv::Mat image, int k, int iters) {
    cout << "---CUDA K-means---" << endl;
    int dimensions = 3;
    int vecSize = image.rows * image.cols;


    vector<cv::Mat> chans;
    split(image, chans);
    cv::Mat res;
    for (int i = 0; i < chans.size(); i++) {
        res.push_back(chans[i]);
    }

    res = res.reshape(1, 1);
    vector<unsigned char> uchar_pointVec;
    uchar_pointVec.assign(res.data, res.data + res.total() * res.channels());
    vector<float> h_pointVec(uchar_pointVec.begin(), uchar_pointVec.end());

    float val;
    char eater;
    int curPoint = 0;
    int offset;
    // Add point to vector
 
    float* d_pointVec;
    int* h_done = new int(0);

    cudaMalloc(&d_pointVec, dimensions * vecSize * sizeof(float));
    cudaMemcpy(d_pointVec, h_pointVec.data(), dimensions * vecSize * sizeof(float), cudaMemcpyHostToDevice);

    // each dimension has k size
    vector<float> h_means(k * dimensions);

    int check;
    srand(time(0));
    // Initialize clusters
    for (int i = 0; i < k; i++) {
        while (true) {
            int idx = rand() % vecSize;
            check = 0;
            for (int j = 0; j < dimensions; j++) {
                if (find(h_means.begin() + k * j, h_means.begin() + k * (j + 1), h_pointVec[idx + vecSize * j])
                    == h_means.begin() + k * (j + 1)) {
                    check++;
                }
                h_means[i + j * k] = h_pointVec[idx + vecSize * j];
            }
            if (check > 0) {
                break;
            }
        }
    }

    cout << k << " clusters initialized" << endl;

    cout << "Running K-means clustering" << endl;

    int blockSize = 1024;
    int gridSize = (vecSize - 1) / blockSize + 1;
    // shared mem has 3 layers
    int sharedSizeCluster = blockSize * (dimensions * sizeof(float) + sizeof(int));
    int sharedSizeMeans = k * gridSize * (dimensions * sizeof(float) + sizeof(int));

    int* d_cur_cluster;
    float* d_means;
    float* d_sums;
    int* d_counter;
    int* d_done;

    cudaMalloc(&d_cur_cluster, vecSize * sizeof(int));
    cudaMalloc(&d_means, k * dimensions * sizeof(float));
    cudaMemcpy(d_means, h_means.data(), k * dimensions * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_sums, gridSize * k * dimensions * sizeof(float));
    cudaMalloc(&d_counter, gridSize * k * sizeof(int));
    cudaMalloc(&d_done, sizeof(int));
    // Clear sum and counter array to prepare for iteration
    cudaMemset(d_sums, 0, gridSize * k * dimensions * sizeof(float));
    cudaMemset(d_counter, 0, gridSize * k * sizeof(int));

    auto start = chrono::high_resolution_clock::now();

    int iter;
    for (iter = 0; iter < iters; iter++) {
        cudaMemset(d_done, 0, sizeof(int));

        setClusters << <gridSize, blockSize, sharedSizeCluster >> > 
            (d_pointVec, d_cur_cluster, d_means, d_sums, vecSize, k, dimensions, gridSize, d_counter, d_done);

        getNewMeans << <1, k * gridSize, sharedSizeMeans >> > (d_means, d_sums, d_counter, k, dimensions);

        cudaMemcpy(h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_done[0] == 0)
            break;
    }

    auto end = chrono::high_resolution_clock::now();
    cout << "Clustering completed in iteration : " << iter << endl << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;

    cudaMemcpy(h_means.data(), d_means, k * dimensions * sizeof(float), cudaMemcpyDeviceToHost);

    //for (int i = 0; i < k; i++) {
    //    cout << "Centroid in cluster " << i << " : ";
    //    for (int j = 0; j < dimensions; j++) {
    //        cout << h_means[i + k * j] << " ";
    //    }
    //    cout << endl;
    //}

    unsigned char* d_image;
    int size = image.rows * image.step;
    cudaMalloc(&d_image, image.rows * image.step);
    cudaMemcpy(d_image, image.ptr(), size, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((image.cols + blockDim.x - 1) / blockDim.x, (image.rows + blockDim.y - 1) / blockDim.y);
    updatePixels <<< gridDim, blockDim >> > (d_image, d_cur_cluster, d_means, k, image.rows, image.cols, image.step);
    cudaMemcpy(image.ptr(), d_image, size, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_cur_cluster);
    cudaFree(d_means);
    cudaFree(d_sums);
    cudaFree(d_counter);
    cudaFree(d_done);

    cv::imshow("result", image);

    return image;
}