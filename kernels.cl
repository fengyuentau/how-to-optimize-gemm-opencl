/* Row major GEMM kernels:

    Mat     Row     Col
    A       M       K
    B       K       N
    C       M       N

*/

#define DEBUG 0

#if OCL_GEMM_KERNEL == 1
// Naive implementation
__kernel void GEMM1(const int M, const int N, const int K,
                    const __global float *A,
                    const __global float *B,
                    __global float *C) {
    const int global_row_index = get_global_id(0);
    const int global_col_index = get_global_id(1);

#if DEBUG
    printf("global_row_index=%d, global_col_index=%d\n", global_row_index, global_col_index);
#endif

    float c = 0.f;
    for (int k = 0; k < K; k++) {
        c += A[global_row_index * K + k] * B[k * N + global_col_index];
    }

    C[global_row_index * N + global_col_index] = c;
}
#endif

#if OCL_GEMM_KERNEL == 2
// Tiling on the three dimensions using local memory
__kernel void GEMM2(const int M, const int N, const int K,
                    const __global float *A,
                    const __global float *B,
                    __global float *C) {
    const int tile_size = OCL_GEMM_KERNEL_TILE_SIZE;

    // Index inside a tile
    const int local_row_index = get_local_id(0);    // [0, OCL_GEMM_KERNEL_TILE_SIZE)
    const int local_col_index = get_local_id(1);    // [0, OCL_GEMM_KERNEL_TILE_SIZE)
    // Index in the C domain
    const int global_row_index = get_global_id(0);  // [0, M]
    const int global_col_index = get_global_id(1);  // [0, N]

    // Allocate local memory for tiling
    __local float A_tile[tile_size * tile_size];
    __local float B_tile[tile_size * tile_size];

    float c = 0.f;
    const int num_tiles = K / tile_size;
    for (int tile_index = 0; tile_index < num_tiles; tile_index++) {
        // Index from tiled domain to source domain
        const int tiled_A_col_index = tile_size * tile_index + local_col_index;
        const int tiled_B_row_index = tile_size * tile_index + local_row_index;
        // Load tile
        A_tile[local_row_index * tile_size + local_col_index] = A[global_row_index * K + tiled_A_col_index];
        B_tile[local_row_index * tile_size + local_col_index] = B[tiled_B_row_index * N + global_col_index];

        // Barrier to make sure the tile_index tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < tile_size; k++) {
            c += A_tile[local_row_index * tile_size + k] * B_tile[k * tile_size + local_col_index];
        }

        // Barrier to make sure works are done with this tile before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store result
    C[global_row_index * N + global_col_index] = c;
}
#endif
