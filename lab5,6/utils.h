// Fonctions auxiliaires
#include <stdio.h>

const int CELLS_PER_THREAD = 16;

const int CELLS_PER_WORD = 16;


/* 
 * 102 102 102 102 102 1 
 * 010010 010010 010010 010010 010010 01
 * 0100 1001 0010 0100 1001 0010 0100 1001
 * 0x49249249
 
*/

const int CELL_INIT_PATTERN = 0x49249249;


// Depuis le SDK Cuda
#  define CUDA_SAFE_CALL_NO_SYNC( call) do {                                 \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)
    
/*#  define CUDA_SAFE_CALL( call) do {                                         \
    CUDA_SAFE_CALL_NO_SYNC(call);                                            \
    cudaError err = cudaThreadSynchronize();                                 \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)
*/

#define CUDA_SAFE_CALL(call) CUDA_SAFE_CALL_NO_SYNC(call)

float random_float(int emin, int emax, int pos_neg);

double getclock();
