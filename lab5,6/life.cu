
#include "utils.h"
#include <stdlib.h>

#include "life_kernel.cu"





// Without optimizing memory accessing...
int life0(){
    // Definition of parameters
    int domain_x = 128;	// Multiple of threads_per_block * cells_per_word
    int domain_y = 128;
    
    int cells_per_word = 1;
    
    int steps = 2;
    
    int threads_per_block = 128;
    int blocks_x = domain_x / (threads_per_block * cells_per_word);
    int blocks_y = domain_y;
    
    dim3  grid(blocks_x, blocks_y);	// CUDA grid dimensions
    dim3  threads(threads_per_block);	// CUDA block dimensions

    // Allocation of arrays
    int * domain_gpu[2] = {NULL, NULL};

	size_t pitch;
	CUDA_SAFE_CALL(cudaMallocPitch((void**)&domain_gpu[0], &pitch,
		domain_x / cells_per_word * sizeof(int),
		domain_y));
    CUDA_SAFE_CALL(cudaMallocPitch((void**)&domain_gpu[1], &pitch,
    	domain_x / cells_per_word * sizeof(int),
		domain_y));

	// Arrays of dimensions pitch * domain.y

	init_kernel0<<< grid, threads, 0 >>>(domain_gpu[0], pitch);

    // Timer initialization
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));

    // Start timer
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));

    // Kernel execution
    int shared_mem_size = 0;
    for(int i = 0; i < steps; i++) {
        life_kernel0<<< grid, threads, shared_mem_size >>>(domain_gpu[i%2], domain_gpu[(i+1)%2], domain_x, domain_y, pitch);
    }

    // Stop timer
    CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    
    float elapsedTime;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));	// In ms
    

    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));

    // Get results back
    int * domain_cpu = (int*)malloc(pitch * domain_y);
    CUDA_SAFE_CALL(cudaMemcpy(domain_cpu, domain_gpu[steps%2], pitch * domain_y, cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(domain_gpu[0]));
    CUDA_SAFE_CALL(cudaFree(domain_gpu[1]));
    

    // Count colors
    int red = 0;
    int blue = 0;
    for(int y = 0; y < domain_y; y++)
    {
    	for(int x = 0; x < domain_x; x++)
    	{
    		int cell = domain_cpu[y * pitch/sizeof(int) + x];
    		printf("%u", cell);
    		if(cell == 1) {
    			red++;
    		}
    		else if(cell == 2) {
    			blue++;
    		}
    	}
    	printf("\n");
    }
    
    printf("Red/Blue cells: %d/%d\n", red, blue);
    printf("GPU time: %f ms\n", elapsedTime);
    
    free(domain_cpu);
    
    return 0;
}













// Lab's question 5 and 6: using shared memory to remove duplicate reads
int life1(){
    // Definition of parameters
    
    // Multiple of threads_per_block * cell_per_word
    int domain_x = 128;
    int domain_y = 128;
    
    // Total of 128 x 128 cells = 16384 cells
    
    int cells_per_word = 1;
    int steps = 2;
    
    
    // CUDA grid dimensions    
    // Max dimension size of a grid size    (x,y,z): (65535, 65535, 65535)    
    int y_step = 4;
    
    int blocks_y = domain_y / y_step;
    int blocks_x = 1;    
    printf("\nGrid dimensions: %d x %d blocks\n", blocks_x, blocks_y);    
    dim3  grid(blocks_x  , blocks_y );
    
    
    // CUDA threads per block
    //Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
    int thr_x = domain_x;
    int thr_y = y_step;    

    printf("Block dimensions: %d x %d threads\n", thr_x , thr_y);
    dim3  threads(thr_x, thr_y);

    // Allocation of arrays
    int * domain_gpu[2] = {NULL, NULL};

    size_t pitch;
    CUDA_SAFE_CALL(cudaMallocPitch((void**)&domain_gpu[0], &pitch, domain_x / cells_per_word * sizeof(int), domain_y));
    CUDA_SAFE_CALL(cudaMallocPitch((void**)&domain_gpu[1], &pitch, domain_x / cells_per_word * sizeof(int), domain_y));

    //printf("%d", pitch);
    
    // Arrays of dimensions pitch * domain.y
    init_kernel1<<< grid, threads, 0 >>>(domain_gpu[0],  domain_x, domain_y, pitch);

    // Timer initialization
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));


    // Kernel execution
    
    int shared_mem_size = thr_x * (thr_y + 2) * sizeof(int) ;
    printf("Shared mem size: %d bytes\n", shared_mem_size);
    
    // Start timer
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));
    

    for(int i = 0; i < steps; i++) {
        life_kernel1<<< grid, threads, shared_mem_size >>>(domain_gpu[i%2], domain_gpu[(i+1)%2], domain_x, domain_y, pitch);        
    }

    // Stop timer
    CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));

    float elapsedTime;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));	// In ms
    

    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));

    // Get results back
    int * domain_cpu = (int*)malloc(pitch * domain_y);
    CUDA_SAFE_CALL(cudaMemcpy(domain_cpu, domain_gpu[steps%2], pitch * domain_y, cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(domain_gpu[0]));
    CUDA_SAFE_CALL(cudaFree(domain_gpu[1]));


    // Count colors
    int red = 0;
    int blue = 0;
    for(int y = 0; y < domain_y; y++){
        for(int x = 0; x < domain_x; x++){
            int cell = domain_cpu[y * pitch/sizeof(int) + x];
            if(cell < 0) cell = 9;
            printf("%u", cell % 10);
            if(cell == 1) {
                red++;
            }
            else if(cell == 2) {
                blue++;
            }
        }
        printf("\n");
    }

    printf("Red/Blue cells: %d/%d\n", red, blue);
    printf("GPU time: %f ms\n", elapsedTime);
    
    
    free(domain_cpu);

    return 0;
}





















// Labs question 8 (16 cells per thread)
int life2(){
    // Definition of parameters
    
    // Multiple of threads_per_block * cell_per_word
    int domain_x = 128;
    int domain_y = 128;
    
    // Total of 128 x 128 cells = 16384 cells
    
    int cells_per_word = 1;
    int steps = 2;
    
    
    // CUDA grid dimensions    
    // Max dimension size of a grid size    (x,y,z): (65535, 65535, 65535)    
    int y_step = 4;
    
    int blocks_y = domain_y / y_step;
    int blocks_x = 1;    
    printf("\nGrid dimensions: %d x %d blocks\n", blocks_x, blocks_y);    
    dim3  grid(blocks_x  , blocks_y );
    
    // CUDA threads per block
    //Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
    int thr_x = domain_x / CELLS_PER_THREAD;
    int thr_y = y_step;    

    printf("Block dimensions: %d x %d threads\n", thr_x , thr_y);
    dim3  threads(thr_x, thr_y);

    // Allocation of arrays
    int * domain_gpu[2] = {NULL, NULL};

    size_t pitch;
    CUDA_SAFE_CALL(cudaMallocPitch((void**)&domain_gpu[0], &pitch, domain_x / cells_per_word * sizeof(int), domain_y));
    CUDA_SAFE_CALL(cudaMallocPitch((void**)&domain_gpu[1], &pitch, domain_x / cells_per_word * sizeof(int), domain_y));

    //printf("%d", pitch);
    
    // Arrays of dimensions pitch * domain.y
    init_kernel2<<< grid, threads, 0 >>>(domain_gpu[0],  domain_x, domain_y, pitch);

    // Timer initialization
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));


    // Kernel execution
    
    int shared_mem_size = thr_x * CELLS_PER_THREAD * (thr_y + 2) * sizeof(int) ;
    printf("Shared mem size: %d bytes\n", shared_mem_size);
    
    // Start timer
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));
    

    for(int i = 0; i < steps; i++) {
        life_kernel2<<< grid, threads, shared_mem_size >>>(domain_gpu[i%2], domain_gpu[(i+1)%2], domain_x, domain_y, pitch);        
    }

    // Stop timer
    CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));

    float elapsedTime;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));	// In ms
    

    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));

    // Get results back
    int * domain_cpu = (int*)malloc(pitch * domain_y);
    CUDA_SAFE_CALL(cudaMemcpy(domain_cpu, domain_gpu[steps%2], pitch * domain_y, cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(domain_gpu[0]));
    CUDA_SAFE_CALL(cudaFree(domain_gpu[1]));


    // Count colors
    int red = 0;
    int blue = 0;
    for(int y = 0; y < domain_y; y++){
        for(int x = 0; x < domain_x; x++){
            int cell = domain_cpu[y * pitch/sizeof(int) + x];
            if(cell < 0) cell = 9;
            printf("%u", cell % 10);
            if(cell == 1) {
                red++;
            }
            else if(cell == 2) {
                blue++;
            }
        }
        printf("\n");
    }

    printf("Red/Blue cells: %d/%d\n", red, blue);
    printf("GPU time: %f ms\n", elapsedTime);
    
    
    free(domain_cpu);

    return 0;
}













// Labs question 9 (16 cells per thread, each thread using 2 bits of a 32 bits int)
int life3(){
    // Definition of parameters
    
    // Multiple of threads_per_block * cell_per_word
    int domain_x = 128;
    int domain_y = 128;
    
    // Total of 128 x 128 cells = 16384 cells
    
    int cells_per_word = 16;
    int steps = 2;
    
    
    // CUDA grid dimensions    
    // Max dimension size of a grid size    (x,y,z): (65535, 65535, 65535)    
    int y_step = 4;
    
    int blocks_y = domain_y / y_step;
    int blocks_x = 1;    
    printf("\nGrid dimensions: %d x %d blocks\n", blocks_x, blocks_y);    
    dim3  grid(blocks_x  , blocks_y );
    
    // CUDA threads per block
    //Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
    int thr_x = domain_x / CELLS_PER_THREAD;
    int thr_y = y_step;    

    printf("Block dimensions: %d x %d threads\n", thr_x , thr_y);
    dim3  threads(thr_x, thr_y);

    // Allocation of arrays
    int * domain_gpu[2] = {NULL, NULL};

    size_t pitch;
    CUDA_SAFE_CALL(cudaMallocPitch((void**)&domain_gpu[0], &pitch, domain_x / cells_per_word * sizeof(int), domain_y));
    CUDA_SAFE_CALL(cudaMallocPitch((void**)&domain_gpu[1], &pitch, domain_x / cells_per_word * sizeof(int), domain_y));

    //printf("%d", pitch);
    
    // Arrays of dimensions pitch * domain.y
    init_kernel3<<< grid, threads, 0 >>>(domain_gpu[0],  domain_x, domain_y, pitch);

    // Timer initialization
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));


    // Kernel execution
    
    int shared_mem_size = thr_x * CELLS_PER_THREAD * (thr_y + 2) * sizeof(int) / cells_per_word;
    printf("Shared mem size: %d bytes\n", shared_mem_size);
    
    
    
    // Start timer
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));
    

    for(int i = 0; i < steps; i++) {
        life_kernel3<<< grid, threads, shared_mem_size >>>(domain_gpu[i%2], domain_gpu[(i+1)%2], domain_x, domain_y, pitch);        
    }

    // Stop timer
    CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));

    float elapsedTime;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));	// In ms
    

    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));

    // Get results back
    int * domain_cpu = (int*)malloc(pitch * domain_y);
    CUDA_SAFE_CALL(cudaMemcpy(domain_cpu, domain_gpu[steps%2], pitch * domain_y, cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(domain_gpu[0]));
    CUDA_SAFE_CALL(cudaFree(domain_gpu[1]));


    // Count colors
    int red = 0;
    int blue = 0;
    for(int y = 0; y < domain_y; y++){
        for(int x = 0; x < domain_x / cells_per_word; x++){
            unsigned cells = (domain_cpu[y * pitch/sizeof(int) + x]);
            
            for(int c=0; c < cells_per_word; c++){
                
                unsigned cell = (cells >> (30 - c*2)) & 0x03;

                printf("%u", cell % 10);
                if(cell == 1) {
                    red++;
                }
                else if(cell == 2) {
                    blue++;
                }                
            }
            //printf(" ");

            
        }
        printf("\n");
    }

    printf("Red/Blue cells: %d/%d\n", red, blue);
    printf("GPU time: %f ms\n", elapsedTime);
    
    
    free(domain_cpu);

    return 0;
}

















int main(int argc, char ** argv){   
    if(argc == 1){                
        printf("You need to specify the implementation number!");
        return 1;
    }    
    int implementation = atoi(argv[1]);                
    switch(implementation){
        case 0: return life0();
        case 1: return life1();        
        case 2: return life2();        
        case 3: return life3();        
    }
    return 0;
}

