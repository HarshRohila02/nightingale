"""Check that PyTorch can use a CUDA GPU. For the owner to run by hand (docs/11 §2).

    .\\.venv-gpu\\Scripts\\python.exe scripts\\check_gpu.py                # report only
    .\\.venv-gpu\\Scripts\\python.exe scripts\\check_gpu.py --device cuda   # plus a short GPU test

Project rule (CLAUDE.md, docs/11 §1): the laptop GPU is for short tests that the owner runs
manually; heavy work goes to the cloud or the university GPU. This script never chooses the
GPU by itself. Without ``--device cuda`` it only reports how PyTorch was built and what it can
see; with it, it runs about a second of matrix multiplication and releases the GPU.
"""

from __future__ import annotations

import argparse
import sys
import time

BLACKWELL_ARCH = "sm_120"  # RTX 50-series, including the RTX 5060 Laptop GPU
MATRIX_SIZE = 4096
REPEATS = 10


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="cuda runs a short matrix-multiply on the GPU; the default only reports",
    )
    args = parser.parse_args(argv)

    try:
        import torch
    except ImportError:
        print(
            "torch is not installed in this environment. GPU tests run from the separate "
            ".venv-gpu environment: see docs/11 §2."
        )
        return 1

    print(f"python            : {sys.version.split()[0]}")
    print(f"torch             : {torch.__version__}")
    print(f"torch build CUDA  : {torch.version.cuda}")
    if torch.version.cuda is None:
        print("This is the CPU build of torch, which cannot use a GPU (docs/11 §2, step 2).")
        return 1 if args.device == "cuda" else 0

    available = torch.cuda.is_available()
    print(f"CUDA available    : {available}")
    if not available:
        print("The driver is too old or no NVIDIA GPU is visible (docs/11 §2, step 1).")
        return 1

    arch_list = torch.cuda.get_arch_list()
    has_blackwell = BLACKWELL_ARCH in arch_list
    print(f"built for         : {' '.join(arch_list)}")
    print(f"{BLACKWELL_ARCH} in build   : {'yes' if has_blackwell else 'NO: reinstall from cu128'}")
    if args.device == "cpu":
        print("Report only. Add --device cuda to run the short GPU test.")
        return 0

    major, minor = torch.cuda.get_device_capability(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"device            : {torch.cuda.get_device_name(0)}")
    print(f"compute capability: {major}.{minor}")
    print(f"memory            : {memory_gb:.1f} GB")

    a = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device="cuda")
    b = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device="cuda")
    (a @ b).sum().item()  # warm-up: the first call also initialises cuBLAS
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(REPEATS):
        c = a @ b
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    tflops = REPEATS * 2 * MATRIX_SIZE**3 / seconds / 1e12
    print(f"matmul test       : OK, {tflops:.1f} TFLOP/s in float32")

    del a, b, c
    torch.cuda.empty_cache()
    print("GPU released. Paste this output to Claude.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
