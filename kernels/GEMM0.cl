__kernel void GEMM(const int M, const int N, const int K,
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
