import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("")
parser.add_argument("--load_dir", "-l", default="results")
parser.add_argument("--skip_my_gemm_kernels", action="store_true")
parser.add_argument("--save_dir", "-s", default="figures")
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

for result_file in os.listdir(args.load_dir):
    if not result_file.endswith(".txt"):
        continue
    if args.skip_my_gemm_kernels and "+GEMM" in result_file:
        continue

    platform_name, device_name, kernel_name = result_file[:-4].split("+")
    x = list()
    y = list()
    with open(os.path.join(args.load_dir, result_file)) as f:
        for line in f.readlines():
            line = line.strip()
            splits = line.split(",")
            scale = int(splits[1].replace("M=", ""))
            x.append(scale)
            gflops = float(splits[7].replace("gflops=", ""))
            y.append(gflops)

    plt.clf()
    plt.plot(x, y)
    plt.xlabel("Matrix Scale")
    plt.ylabel("GFLOPS")
    plt.savefig("{}/{}+{}.png".format(args.save_dir, device_name, kernel_name))
