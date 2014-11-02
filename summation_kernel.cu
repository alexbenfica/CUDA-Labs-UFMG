


/*
  
http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
 
 B.4. Built-in Variables
Built-in variables specify the grid and block dimensions and the block and thread indices. They are only valid within functions that are executed on the device.

B.4.1. gridDim
This variable is of type dim3 (see dim3) and contains the dimensions of the grid.

B.4.2. blockIdx
This variable is of type uint3 (see char, short, int, long, longlong, float, double) and contains the block index within the grid.

B.4.3. blockDim
This variable is of type dim3 (see dim3) and contains the dimensions of the block.

B.4.4. threadIdx
This variable is of type uint3 (see char, short, int, long, longlong, float, double ) and contains the thread index within the block.

B.4.5. warpSize
This variable is of type int and contains the warp size in threads (see SIMT Architecture for the definition of a warp).

Read more at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ixzz3HRncyiwA 
Follow us: @GPUComputing on Twitter | NVIDIA on Facebook
 
 */


// GPU kernel 
// Each thread receives a contiguous block of data
__global__ void summation_kernel_0(int data_size, results* data_out)
{   
    
    int tot_thr = gridDim.x * blockDim.x;
    int thr_data_size = data_size / tot_thr;
    int thr_id_abs = blockIdx.x * blockDim.x + threadIdx.x;
    int thr_offset = thr_id_abs * thr_data_size;
    
    /*
    if(threadIdx.x >= 0){
        data_out[blockIdx.x * blockDim.x + threadIdx.x].sum = (float)blockIdx.x;        
    }
    */
    
    
    int i;
    float result = 0.0;        
    for(int j = 0; j < thr_data_size; j++){
        i = thr_offset + j;
        result +=  (float)(((((i%2)-1)+(i%2)))*-1.) / (float)(i+1);
    }
    data_out[thr_id_abs].sum = (float) result;
    
}




// GPU kernel 2 (sliced)
__global__ void summation_kernel_interleaved(int data_size, results* data_out)
{    
     int tot_thr = gridDim.x * blockDim.x;
    int thr_data_div = data_size / tot_thr;
    int thr_id_abs = blockIdx.x * blockDim.x + threadIdx.x;

    
    int i;
    float result = 0.0;        
    for(int j = 0; j < thr_data_div; j++){        
        i = j * tot_thr + thr_id_abs;
                
        result +=  (float)(((((i%2)-1)+(i%2)))*-1.) / (float)(i+1);
    }
    data_out[thr_id_abs].sum = (float) result;
    
}





__global__ void summation_kernel_value_per_block(int data_size, results* data_out)
{    
 
    // http://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/
    extern __shared__ float sum_threads[];

    int tot_thr = gridDim.x * blockDim.x;
    int thr_data_size = data_size / tot_thr;
    int thr_id_abs = blockIdx.x * blockDim.x + threadIdx.x;
    int thr_offset = thr_id_abs * thr_data_size;
    
    /*
    if(threadIdx.x >= 0){
        data_out[blockIdx.x * blockDim.x + threadIdx.x].sum = (float)blockIdx.x;        
    }
    */    
    
    int i;
    float result = 0.0;        
    for(int j = 0; j < thr_data_size; j++){
        i = thr_offset + j;
        result +=  (float)(((((i%2)-1)+(i%2)))*-1.) / (float)(i+1);
    }
    sum_threads[threadIdx.x] = result;
    
    __syncthreads();    
    
    // Use the thread 0 to sum the reasults of each thread in a block.    
    if(threadIdx.x == 0){
        float sum_block = 0.0;
        for(i=0;i<blockDim.x;i++){
            sum_block += sum_threads[i];
        }
        data_out[blockIdx.x].sum = sum_block;    
    }
    
    
}







