// More works in a work-item
__kernel void GEMM(const int M, const int N, const int K,
                   const __global float *A,
                   const __global float *B,
                   __global float *C) {
    const int tile_size = OCL_GEMM_KERNEL_TILE_SIZE;
    const int work_per_thread = OCL_GEMM_KERNEL_WORK_PER_THREAD;
    const int reduced_tile_size = tile_size / work_per_thread;

    // Index inside a tile
    const int local_row_index = get_local_id(0);    // [0, OCL_GEMM_KERNEL_TILE_SIZE)
    const int local_col_index = get_local_id(1);    // [0, OCL_GEMM_KERNEL_TILE_SIZE / work_per_thread)
    // Index in the C domain
    const int global_row_index = get_global_id(0);  // [0, M)
    // const int global_col_index = get_global_id(1);  // [0, N / work_per_thread)
    const int global_col_index = tile_size * get_group_id(1) + local_col_index; // [0, N)

    // number of groups: (N / work_per_thread) / (tile_size / work_per_thread) = N / tile_size
    // index in global_size N: [0, N / work_per_thread)
    // index in local_size N: [0, tile_size / work_per_thread)

    // Allocate local memory for tiling
    __local float A_tile[tile_size * tile_size];
    __local float B_tile[tile_size * tile_size];

    __local float c[work_per_thread];
    for (int w = 0; w < work_per_thread; w++) {
        c[w] = 0.f;
    }
    const int num_tiles = K / tile_size;
    for (int tile_index = 0; tile_index < num_tiles; tile_index++) {
        // Load a tile
        for (int w = 0; w < work_per_thread; w++) {
            // Index from tiled domain to source domain
            const int tiled_A_col_index = tile_size * tile_index + local_col_index;
            const int tiled_B_row_index = tile_size * tile_index + local_row_index;
            A_tile[local_row_index * tile_size + w * reduced_tile_size + local_col_index] = A[global_row_index * K + w * reduced_tile_size + tiled_A_col_index];
            B_tile[local_row_index * tile_size + w * reduced_tile_size + local_col_index] = B[tiled_B_row_index * N + w * reduced_tile_size + global_col_index];
        }

        // Barrier to make sure the tile_index tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < tile_size; k++) {
            for (int w = 0; w < work_per_thread; w++) {
                c[w] += A_tile[local_row_index * tile_size + k] * B_tile[k * tile_size + w * reduced_tile_size + local_col_index];
            }
        }

        // Barrier to make sure works are done with this tile before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store result
    for (int w = 0; w < work_per_thread; w++) {
        C[global_row_index * N + w * reduced_tile_size + global_col_index] = c[w];
    }
}
