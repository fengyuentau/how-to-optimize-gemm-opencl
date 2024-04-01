import os
import argparse
import subprocess

def main(args):
    all_kernels = dict()
    for k in os.listdir("kernels"):
        if k.endswith(".cl"):
            key = k[:-3].lower()
            all_kernels[key] = os.path.join("kernels", k)
    if args.with_clblast:
        all_kernels["clblast"] = "clblast"
    print(all_kernels)

    selected_kernel = list(all_kernels.values()) if args.kernel == "all" else [all_kernels[args.kernel.lower()]]
    selected_kernel = sorted(selected_kernel)

    for k in selected_kernel:
        output = subprocess.run(["./build/main", k], stdout=subprocess.PIPE)
        if output.stderr is not None:
            print(k, output.stderr.decode("utf-8"))
        output = output.stdout.decode("utf-8")
        output_lines = output.splitlines()

        platform_name = output_lines[0]
        device_name = output_lines[1]
        kernel_name = k if k == "clblast" else k.split("/")[-1][:-3]
        log_name = "{}+{}+{}.txt".format(platform_name, device_name, kernel_name)
        with open(os.path.join(args.save_dir, log_name), "w") as f:
            for line in output_lines[2:]:
                f.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--kernel", "-k", default="all")
    parser.add_argument("--save_dir", "-s", default="results")
    parser.add_argument("--with_clblast", action="store_true")
    args = parser.parse_args()
    main(args)
