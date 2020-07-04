#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>

//лучший результат при соотношении 1 к 5

//#define N 500
//#define div 100

//#define N 1000
//#define div 200

#define N 2000
#define div 400

//#define N 5000
//#define div 1000

//алгоритм перемножения матриц
__global__ void mult_matrix(int* M1, int *M2, int *R, int i) {
	int k = blockIdx.x * (N / div) + threadIdx.x;
	int j = blockIdx.y * (N / div) + threadIdx.y;
	R[k * N + j] += M1[k * N + i] * M2[i * N + j];
}

int main() {

	printf("N = %d, div = %d \n", N, div);

	//заполнение матриц
	int* M1, *M2, *R;
	M1 = new int[N * N];
	M2 = new int[N * N];
	R = new int[N * N];
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			M1[i * N + j] = rand() % 10;
			M2[i * N + j] = rand() % 10;
			R[i * N + j] = 0;
		}
	}

	int* dev_m1, *dev_m2, *dev_r;
	cudaError_t cudaStatus;
	cudaMalloc((void**)&dev_m1, N * N * sizeof(int));
	cudaMalloc((void**)&dev_m2, N * N * sizeof(int));
	cudaMalloc((void**)&dev_r, N * N * sizeof(int));

	cudaError_t error;

	error = cudaMemcpy(dev_m1, M1, N * N * sizeof(int), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}
	error = cudaMemcpy(dev_m2, M2, N * N * sizeof(int), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}
	error = cudaMemcpy(dev_r, R, N * N * sizeof(int), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}

	dim3 grid(div, div);
	dim3 blocks(N / div, N / div);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	//запускаем алгоритм
	for (int i = 0; i < N; ++i) {
		mult_matrix << <grid, blocks >> > (dev_m1, dev_m2, dev_r, i);
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	//записываем время работы
	float time = 0;
	cudaEventElapsedTime(&time, start, stop);

	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}

	cudaDeviceSynchronize();

	error = cudaMemcpy(M1, dev_m1, N * N * sizeof(int), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}

	error = cudaMemcpy(M2, dev_m2, N * N * sizeof(int), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}

	error = cudaMemcpy(R, dev_r, N * N * sizeof(int), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}

	printf("\nTIME: \n");
	printf("%f ms\n", time);

	delete M1;
	delete M2;
	delete R;
	cudaFree(dev_m1);
	cudaFree(dev_m2);
	cudaFree(dev_r);
	return 0;
}