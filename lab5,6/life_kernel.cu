
__global__ void init_kernel(int * domain, int pitch){
	domain[blockIdx.y * pitch / sizeof(int) + blockIdx.x * blockDim.x + threadIdx.x] = (1664525ul * (blockIdx.x + threadIdx.y + threadIdx.x) + 1013904223ul) % 3;
}



// Reads a cell at (x+dx, y+dy)
__device__ int read_cell(int * source_domain, int x, int y, int dx, int dy, int domain_x, int domain_y, int pitch){
    x = (x + dx) % domain_x;	// Wrap around
    y = (y + dy) % domain_y;
    return source_domain[y * (pitch / sizeof(int)) + x];
}



// Writes cell at (x+dx, y+dy)
__device__ int write_cell(int * dest_domain, int x, int y, int dx, int dy, int domain_x, int domain_y, int pitch, int cell_value){
    x = (x + dx) % domain_x;	// Wrap around
    y = (y + dy) % domain_y;
    dest_domain[y * (pitch / sizeof(int)) + x] = cell_value;
    return 0;
}




// Compute kernel
__global__ void life_kernel(int * source_domain, int * dest_domain,int domain_x, int domain_y, int pitch){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y;
    
    // Read cell
    int myself = read_cell(source_domain, tx, ty, 0, 0, domain_x, domain_y, pitch);
    
    // TODO: Read the 8 neighbors and count number of blue and red
    int nr = 0; // number of red
    int nb = 0; // number of blue    
    int na = 0; // number of adjacent neighbours
    
    int neig_value;
    
    for(int i=-1; i<2; i++){        
        for(int j=-1; j<2; j++){
            
            // skip the central element
            if((i==0) && (j==0)) continue;            
            
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
    

    
    
    
    // TODO: Compute new value       
    // begins dead...
    int new_value = myself;
    
    int tot_neig = nb + nr;
    
    
    // Survival conditions
    if(na == 1){
        new_value = 0;
    }
    
    if(tot_neig > 3){
        new_value = 0;
    }
    
    if(tot_neig == 3){    
        if(myself == 0){
            if(tot_neig == 3){
                if(nb >= 2){
                    new_value = 2;
                }
                else{
                    new_value = 1;
                }
            }
        }
    }
    
    
  
    
    //new_value = na;
    
    // TODO: Write it in dest_domain
    write_cell(dest_domain, tx, ty, 0, 0, domain_x, domain_y, pitch, new_value);
    
}

