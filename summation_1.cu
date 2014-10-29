#include "utils.h"
#include <stdlib.h>

struct results
{
	float sum;
};

#include "summation_kernel.cu"



// CPU implementation
float log2_series(int n)
{    
    int i = 0;
    double sum = 0;
    for(i=n-1;i>=0;i--){
        sum += ((((i%2)-1.0) + (i%2)) * (-1.0)) / ((double)i + 1.0);
        //printf("%.9f\n",sum);
    }
    //printf("%.50f\n", sum);
    //printf("%.50e\n", sum - sum1);
    return sum;
}







int main(int argc, char ** argv)
{
    int data_size = 1024 * 1024 * 128;

    // Run CPU version
    double start_time = 0;
    //double start_time = getclock();
    float log2 = log2_series(data_size);
    double end_time = 1;
    //double end_time = getclock();
    
    printf("CPU result: %f\n", log2);
    printf(" log(2)=%f\n", log(2.0));
    printf(" time=%fs\n", end_time - start_time);





    
    // Parameter definition
    int threads_per_block = 4 * 32;
    int blocks_in_grid = 8;    
    int num_threads = threads_per_block * blocks_in_grid;

    // Timer initialization
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));

    int results_size = num_threads;
    results* data_out_cpu;
    
    // Allocating output data on CPU    
    data_out_cpu = (results *) malloc(sizeof(results) * results_size);

    // Allocating output data on GPU    
    results * data_out_gpu;
    cudaMalloc((void**)&data_out_gpu, sizeof(results) * results_size);

    // Start timer
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));
    
    

    // Execute kernel
    // summation_kernel<<<1, num_threads>>>(data_size / num_threads, data_out_gpu);
    
    // Number of blocks to use on kernel   
    int k_blocks = 1;    
    // Number of threads per block to use on kernel.
    int k_threads = 32;
    
    summation_kernel_1<<<k_blocks, k_threads>>>(data_size, data_out_gpu);

    
    
    // Stop timer
    CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));

    //return 0;
    
    // Get results back
    // TODO
    cudaMemcpy(data_out_cpu, data_out_gpu, sizeof(results) * results_size, cudaMemcpyDeviceToHost);
    
    // Finish reduction
    int i;
    float sum = 0.;
    for(i=0; i<num_threads; i++){
        sum += data_out_cpu[i].sum;        
    }
    
    // Cleanup
    // TODO
    cudaFree(data_out_gpu);
    free(data_out_cpu);
    
    printf("GPU results:\n");
    printf(" Sum: %f\n", sum);
    
    float elapsedTime;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));	// In ms

    double total_time = elapsedTime / 1000.;	// s
    double time_per_iter = total_time / (double)data_size;
    double bandwidth = sizeof(float) / time_per_iter; // B/s
    
    printf(" Total time: %g s,\n Per iteration: %g ns\n Throughput: %g GB/s\n",
    	total_time,
    	time_per_iter * 1.e9,
    	bandwidth / 1.e9);
  
    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));
    return 0;
}

