/* Row major GEMM kernels:

    A   M   K
    B   K   N
    C   M   N

*/

#define DEBUG 0

#if OCL_GEMM_KERNEL == 1
// naive
__kernel void GEMM1(const int M, const int N, const int K,
                    const __global float *A,
                    const __global float *B,
                    __global float *C) {
    const int global_row_index = get_global_id(0);
    const int global_col_index = get_global_id(1);

#if DEBUG
    printf("global_row_index=%d, global_col_index=%d\n", global_row_index, global_col_index);
#endif

    float acc = 0.f;
    for (int k = 0; k < K; k++) {
        acc += A[global_row_index * K + k] * B[k * N + global_col_index];
    }

    C[global_row_index * N + global_col_index] = acc;
}
#endif
