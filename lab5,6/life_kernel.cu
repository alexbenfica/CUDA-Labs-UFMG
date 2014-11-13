



// Reads a cell at (x+dx, y+dy)
__device__ int read_cell(int * source_domain, int x, int y, int dx, int dy, int domain_x, int domain_y, int pitch){
    
    
    /* 
    int i = -1;
    printf("%d\n", i % 128);
    i = -2;
    printf("%d\n", i % 128);    
    return 0;
    */
    
    
    
    x = (x + dx + domain_x) % domain_x;	// Wrap around
    y = (y + dy + domain_y) % domain_y;
    return source_domain[y * (pitch / sizeof(int)) + x];
}



// Writes cell at (x+dx, y+dy)
__device__ int write_cell(int * dest_domain, int x, int y, int dx, int dy, int domain_x, int domain_y, int pitch, int cell_value){
    x = (x + dx + domain_x) % domain_x;	// Wrap around
    y = (y + dy + domain_y) % domain_y;
    dest_domain[y * (pitch / sizeof(int)) + x] = cell_value;
    return 0;
}



__device__ int calc_color(int myself, int nb, int nr, int na){
    
    int color = myself;    
    int tot_neig = nb + nr;   
    
    // Survival conditions
    if(na == 1){
        color = 0;
    }
    
    if(tot_neig > 3){
        color = 0;
    }
    
    if(tot_neig == 3){    
        if(myself == 0){
            if(tot_neig == 3){
                if(nb >= 2){
                    color = 2;
                }
                else{
                    color = 1;
                }
            }
        }
    }
    return color;
}




// KERNEL 1


__global__ void init_kernel0(int * domain, int pitch){
    domain[blockIdx.y * pitch / sizeof(int) + blockIdx.x * blockDim.x + threadIdx.x] = (1664525ul * (blockIdx.x + threadIdx.y + threadIdx.x) + 1013904223ul) % 3;
}



// Compute kernel
__global__ void life_kernel0(int * source_domain, int * dest_domain,int domain_x, int domain_y, int pitch){
    
    
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y;
    
    
    int myself = 0;

    int nr = 0; // number of red
    int nb = 0; // number of blue    
    int na = 0; // number of adjacent neighbours
    
    int neig_value;
    
    for(int i=-1; i<2; i++){        
        for(int j=-1; j<2; j++){
            
            // skip the central element
            if((i==0) && (j==0)) {
                // Read cell
                myself = read_cell(source_domain, tx, ty, 0, 0, domain_x, domain_y, pitch);
                continue;            
            }
            
            neig_value = read_cell(source_domain, tx, ty, i, j, domain_x, domain_y, pitch);
            
            if(neig_value == 1) nr++;
            if(neig_value == 2) nb++;

            if((i == 0) || (j == 0)){
                if(neig_value > 0){
                    na++;
                }
            }                
        }        
    }
    
    int color = calc_color(myself, nb, nr, na);
    
    // TODO: Write it in dest_domain
    write_cell(dest_domain, tx, ty, 0, 0, domain_x, domain_y, pitch, color);    
}






// KERNEL 1



__global__ void init_kernel1(int * domain, int domain_x, int domain_y, int pitch){
    
    int tx = threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;    
    
    int value = threadIdx.x % 3;    
    if(value != 2) value ^= 1 << 0;
    write_cell(domain, tx, ty, 0 , 0 , domain_x, domain_y, pitch, value);                        
}







// Compute kernel
__global__ void life_kernel1(int * source_domain, int * dest_domain, int domain_x, int domain_y, int pitch){
    
    extern __shared__ int sha[];
    
    int ty, isha;        
    
    // Each thread copies one or two values form global to shared memory.
    // First blockDim.x bytes has the values from line before.
    // Last blockDim.x bytes has the values from line after.
    isha = threadIdx.x + (threadIdx.y + 1) * blockDim.x;    
    ty = blockIdx.y * blockDim.y + threadIdx.y;            
    
    sha[isha] = read_cell(source_domain, threadIdx.x, ty, 0, 0, domain_x, domain_y, pitch);            
    
    // copies the first line
    if(!threadIdx.y){                
        sha[threadIdx.x] = read_cell(source_domain, threadIdx.x, ty, 0, -1, domain_x, domain_y, pitch);            
    }

    // copies the last line
    if(threadIdx.y == blockDim.y-1){        
        sha[isha + blockDim.x] = read_cell(source_domain, threadIdx.x, ty, 0, 1, domain_x, domain_y, pitch);            
    }
    
    __syncthreads();        
    

            
#if 0
    // Debug copy from global to shared ...
    if(blockIdx.y==0){    
        if(threadIdx.x==0){    
            if(threadIdx.y==0){    
            
                //printf("\n%d", blockDim.y);
                //printf("\n%d\n", blockDim.x);

                for(int j=0;j<blockDim.y+2;j++){
                    for(int i=0;i<blockDim.x;i++){        
                        int value = sha[i + j*blockDim.x];
                        if((value < 3)&&(value >= 0)){                        
                            printf("%d", value);
                        }
                        else{
                            printf("-");
                        }

                    }
                    printf("\n");        
                }

                printf("\n");        
                printf("\n");        
            
            }
        }
    }    
    

    //write_cell(dest_domain, threadIdx.x, ty, 0 , 0 , domain_x, domain_y, pitch, sha[isha]);                        
    
#endif    


    int nr = 0; // number of red
    int nb = 0; // number of blue    
    int na = 0; // number of adjacent neighbours
    
    int myself;    

     
#if 1    
    for(int x = -1; x < 2; x++){                
        for(int y = -1; y < 2; y++){            
            
            int sx = (threadIdx.x + x + blockDim.x) % blockDim.x;            
            int sy = threadIdx.y + y + 1;                                    
            int sindex = sx + sy * blockDim.x;

            int neig_value = sha[sindex];            
            
            // The central element is the cell itself
            if((x==0) && (y==0)){
                myself = neig_value;
                continue;            
            } 
            
            nr += (neig_value & 1);
            nb += (neig_value & 1<<1);
            na += (!(x&y)) & neig_value;
            
            /*
            if((x == 0) || (y == 0)){
                if(neig_value > 0){
                    na++;
                }
            } 
            */
            
            #if 0
            if(blockIdx.y==0){    
                if(threadIdx.x==2){    
                    if(threadIdx.y==0){    
                        printf("\nsindex=%d \n" , sindex);
                        //printf("\nx=%d \ny=%d \nsx=%d \nsy=%d \nsindex=%d \nblockDim.x=%d \nblockDim.y=%d \nneig_value=%d\n" , x, y, sx, sy, sindex, blockDim.x, blockDim.y, neig_value);
                    }
                }
            }

            #endif            
        }        
    }
    
#endif                
    
    
    int color = calc_color(myself, nb, nr, na);
    
    ty = blockIdx.y * blockDim.y + threadIdx.y;        
    
    write_cell(dest_domain, threadIdx.x, ty, 0, 0, domain_x, domain_y, pitch, color);
}








