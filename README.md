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

## Run

```shell
python3 run.py # Run all kernels in ./kernels
python3 run.py -k GEMM0 # Run ./kernels/GEMM0.cl
```

## FAQ

### Render permission denied

See https://dgpu-docs.intel.com/driver/installation.html#configuring-render-group-membership.

## License

[Apache 2.0 License](./LICENSE)
