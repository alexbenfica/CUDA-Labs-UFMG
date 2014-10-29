


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


// GPU kernel blocks of data
__global__ void summation_kernel(int data_size, results* data_out)
{    
    
    float result = 0.;    
    int offset = threadIdx.x * data_size;
    
    for(int i = offset; i < offset + data_size; i++){
        result +=  (float)(((((i%2)-1)+(i%2)))*-1.) / (float)(i+1);
    }
    data_out[threadIdx.x].sum = result;
    
}




// GPU kernel 2 (sliced)
__global__ void summation_kernel_1(int data_size, results* data_out)
{    
    float result = 0.;    
    int i;
    
    int tot_thr = gridDim.x * blockDim.x;    
    int dt_sz_thr = data_size / tot_thr;
    
    int thr_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(int j = 0; j < tot_thr; j++){
        i = j * dt_sz_thr + thr_id;
        result +=  (float)(((((i%2)-1)+(i%2)))*-1.) / (float)(i+1);
    }
    data_out[thr_id].sum = result;    
}



