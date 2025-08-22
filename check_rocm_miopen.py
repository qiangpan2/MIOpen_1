#!/usr/bin/env python3
# check_miopen_torch.py

import subprocess
import os
import sys

def check_pytorch_installation():
    """Checks if PyTorch is installed and prints its version."""
    try:
        import torch
        print(f"[1/4] PyTorch is installed. Version: {torch.__version__}")
        return True
    except ImportError:
        print("[1/4] Error: PyTorch is not installed.")
        return False

def check_rocm_support():
    """Checks if the installed PyTorch was built with ROCm support."""
    try:
        import torch
        # For PyTorch built with ROCm, this often contains ROCm/HIP related info
        config_str = str(torch.__config__.show())
        if "USE_ROCM" in config_str or "HIP" in config_str:
            print("[2/4] PyTorch build supports ROCm (likely includes MIOpen).")
            return True
        else:
            print("[2/4] Warning: PyTorch build info does not obviously indicate ROCm support.")
            print(f"     Build info snippet: {config_str[:200]}...")
            return False
    except Exception as e:
        print(f"[2/4] Error checking PyTorch ROCm support: {e}")
        return False

def check_cuda_device_availability():
    """Checks if CUDA (ROCm) devices are available to PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"[3/4] CUDA (ROCm) is available to PyTorch. Number of devices: {device_count}")
            for i in range(device_count):
                print(f"      Device {i}: {torch.cuda.get_device_properties(i).name}")
            return True
        else:
            print("[3/4] CUDA (ROCm) is NOT available to PyTorch. No GPU devices found.")
            return False
    except Exception as e:
        print(f"[3/4] Error checking CUDA availability: {e}")
        return False

def check_miopen_library_link():
    """Attempts to infer if MIOpen library is linked by checking PyTorch's core library dependencies."""
    try:
        import torch
        # Find the main PyTorch shared library
        # A common one is libtorch_python.so, but libtorch.so is more core.
        # We'll try to find a core library in the torch lib directory.
        torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
        core_lib_name = "libtorch.so" # This is a key library
        core_lib_path = os.path.join(torch_lib_dir, core_lib_name)

        if not os.path.exists(core_lib_path):
            # Fallback: try libtorch_python.so
            core_lib_name = "libtorch_python.so"
            core_lib_path = os.path.join(torch_lib_dir, core_lib_name)
        
        if not os.path.exists(core_lib_path):
            print(f"[4/4] Could not find a core PyTorch library ({core_lib_name}) to check dependencies.")
            return False

        # Use ldd to list dynamic dependencies and search for MIOpen
        result = subprocess.run(["ldd", core_lib_path], capture_output=True, text=True, check=True)
        if "libMIOpen" in result.stdout:
            # Extract the line for clarity
            miopen_line = [line.strip() for line in result.stdout.splitlines() if "libMIOpen" in line][0]
            print(f"[4/4] MIOpen library is linked. Found in dependencies:\
      {miopen_line}")
            return True
        else:
            print("[4/4] MIOpen library is NOT found in PyTorch's core library dependencies (checked via ldd).")
            return False

    except subprocess.CalledProcessError as e:
        print(f"[4/4] Error running 'ldd' on PyTorch library: {e}")
        return False
    except Exception as e:
        print(f"[4/4] An unexpected error occurred while checking library links: {e}")
        return False

def main():
    """Main function to run all checks."""
    print("Checking PyTorch and MIOpen linkage...)
    
    checks = [
        check_pytorch_installation,
        check_rocm_support,
        check_cuda_device_availability,
        check_miopen_library_link
    ]
    
    results = []
    for check in checks:
        results.append(check())
        print() # Add a blank line for readability between checks

    # Summary
    print("----- Summary -----")
    if all(results):
        print("All checks passed. PyTorch is likely correctly linked with MIOpen.")
    else:
        print("One or more checks failed or gave warnings. Please review the output above.")

if __name__ == "__main__":
    main()