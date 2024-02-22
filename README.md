# how-to-optimize-gemm-opencl

Step-by-step row major GEMM optimization tutorial on OpenCL GPU platforms (OpenCL >= 1.2). Tested on Khadas VIM4 (A311D2).

## Environment

Ubuntu 22.04:

```shell
sudo apt install clinfo opencl-headers mesa-opencl-icd ocl-icd-opencl-dev
```

## Build

Linux:

```shell
cmake -B build .
cmake --build build
```
