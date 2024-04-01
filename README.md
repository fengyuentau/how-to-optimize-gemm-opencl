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

Build with CLBlast:

```
# Get and build CLBlast
cd $workspace
git clone https://github.com/CNugteren/CLBlast
cd CLBlast
cmake -B build -DCMAKE_INSTALL_PREFIX=build/install .
cmake --build build --target install

# Build this repo
cd $workspace
cd how-to-optimize-gemm-opencl
export CLBlast_DIR=/path/to/CLBlast/build/install/lib/cmake/CLBlast
cmake -B build -DWITH_CLBLAST=ON .
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
