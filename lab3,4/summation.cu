#include "utils.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include "summation_kernel.cu"




// CPU implementation
float log2_series(int n){    
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
    
    if(argc < 4){                
        printf("\nYou must specify: kernel_id,  number of blocks, and number of threads per block.\n");
        return 1;
    }
    
    // Executes kernel, depending on input parameters...        
    int kernel_id = atoi(argv[1]);        
    
    
    int data_size = 1024 * 1024 * 128;

    
    // Run CPU version
    clock_t start_cpu = clock();    
    float log2 = log2_series(data_size);                    
    clock_t end_cpu = clock();    
    float seconds = (float)(end_cpu - start_cpu) / CLOCKS_PER_SEC;    
    printf("\nlog(2)    = %20.20f", log(2.0));    
    printf("\nCPU RESULT: %20.20f\n", log2);        
    printf(" Total time :%fs\n", seconds);

    // Parameter definition (original from example...)
    int blocks_in_grid = 8;    
    int threads_per_block = 4 * 32;

    // Modified parameters for testing purposes...
    // Some ideias about how to setup the block an thread number
    // http://stackoverflow.com/questions/4861244/how-many-threads-does-nvidia-gts-450-has
    blocks_in_grid = atoi(argv[2]);
    threads_per_block = atoi(argv[3]);

    
    int num_threads = threads_per_block * blocks_in_grid;
    

    // Timer initialization and configuration
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));

    
    int results_size;
    
    switch(kernel_id){
        case SUM:            
        case SUM_INTERLEAVED:
            // Each thread will return only one element as a result.
            results_size = num_threads;            
            break;
        
        case SUM_BLOCK:
        case SUM_GPU_ONLY:
            // Only one element will be returned per block
            results_size = blocks_in_grid;            
            break;                       
            
    }

    
    // data_out_cpu is a pointer of type results
    results* data_out_cpu;
    results* data_out_gpu;
    results* data_out_reduce;
    
    // Allocating output data on CPU
    // Cast necessary to ensure corret type on data_out_cpu
    data_out_cpu = (results *) malloc(sizeof(results) * results_size);

    // Allocating output data on GPU    
    cudaMalloc((void**)&data_out_gpu, sizeof(results) * results_size);

    
    if(kernel_id == SUM_GPU_ONLY){        
        // Allocating output data for reduce on GPU    
        cudaMalloc((void**)&data_out_reduce, sizeof(results) * results_size);        
    }
    
    
    // Start timer
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));

    
    switch(kernel_id){
        case SUM:
            summation_kernel_0<<<blocks_in_grid, threads_per_block>>>(data_size, data_out_gpu);
            break;
        case SUM_INTERLEAVED:
            summation_kernel_interleaved<<<blocks_in_grid, threads_per_block>>>(data_size, data_out_gpu);
            break;
        case SUM_BLOCK:            
            summation_kernel_value_per_block<<<blocks_in_grid, threads_per_block, threads_per_block*sizeof(float)>>>(data_size, data_out_gpu);
            break;
        case SUM_GPU_ONLY:            
            summation_kernel_value_per_block<<<blocks_in_grid, threads_per_block, threads_per_block*sizeof(float)>>>(data_size, data_out_gpu);
            reduce<<<blocks_in_grid, threads_per_block, threads_per_block*sizeof(float)>>>(blocks_in_grid, data_out_gpu, data_out_reduce);
            break;                        
    }
    
    
    // Stop timer
    CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));

    switch(kernel_id){
        case SUM:
        case SUM_INTERLEAVED:
        case SUM_BLOCK:            
            // Get results back from GPU to CPU memory
            cudaMemcpy(data_out_cpu, data_out_gpu, sizeof(results) * results_size, cudaMemcpyDeviceToHost);
            break;
        case SUM_GPU_ONLY:
            // Copy only one element back from GPU to CPU
            cudaMemcpy(data_out_cpu, data_out_reduce, sizeof(results) * 1, cudaMemcpyDeviceToHost);
            break;            
    }

    
    
    
    int i;
    float sum = 0.;    
    switch(kernel_id){
        // Finish reduction on CPU, adding all elements
        case SUM:
        case SUM_INTERLEAVED:
        case SUM_BLOCK:            
            printf("\n");
            for(i=0; i<results_size; i++){
                sum += data_out_cpu[i].sum;        
                #if 0
                if((i>0)&&(i<40)){
                    printf("Thread %d result: %20.20f\n" , i, data_out_cpu[i].sum);
                }
                #endif  
            }
            break;
        case SUM_GPU_ONLY:
            sum = data_out_cpu[0].sum;        
            break;
    }
            
    
    // Cleanup CPU and GPU memory.
    cudaFree(data_out_gpu);
    free(data_out_cpu);
    
    
    
    // Show timming statistics
    
    printf("\nlog(2)   = %20.20f", log(2.0));
    
    printf("\nGPU RESULT:%20.20f\n", sum);
    printf(" Kernel ID: %d\n", kernel_id);
    printf(" Blocks: %d\n", blocks_in_grid);
    printf(" Thread per block: %d\n", threads_per_block);
    printf(" Total threads: %d\n", threads_per_block * blocks_in_grid);
    
    float elapsedTime;
    // In ms
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));	
    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));
    
    double total_time = elapsedTime / 1000.;	// s
    double time_per_iter = total_time / (double)data_size;
    double bandwidth = sizeof(float) / time_per_iter; // B/s
    
    printf(" Per iteration: %g ns\n Throughput: %g GB/s\n Total time: %gs\n",    	
    	time_per_iter * 1.e9,
    	bandwidth / 1.e9,
        total_time);
  
    printf("\n Speedup CPU to GPU: %5.2fx" , ((double)seconds / total_time));
    
    return 0;
}

