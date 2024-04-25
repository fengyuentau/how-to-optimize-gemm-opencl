# Timing cuBlas

Build separately:

```
cmake -B build
cmake --build build
```

Run:

```
# Set GPU id before running if there are multiple.
# Otherwise it leads to segmentation fault.
export CUDA_VISIBLE_DEVICES=0

./build/cublas_timing
```
