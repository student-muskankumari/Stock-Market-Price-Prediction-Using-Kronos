# setup_checker.py
"""
Interactive requirements checker/installer.
- Detects installed pip packages.
- If missing, asks user (y/n) to install via pip.
- For PyTorch (GPU), suggests the conda install command and offers
  to install CPU-only torch via pip automatically if user agrees.
"""

import subprocess
import sys
import pkgutil
import shutil

REQUIRED = [
    "streamlit", "pandas", "numpy", "yfinance",
    "transformers", "accelerate", "sentencepiece", "matplotlib", "pyyaml", "tqdm"
]

def is_installed(pkg):
    return pkgutil.find_loader(pkg) is not None

def pip_install(pkg):
    print(f"> pip install {pkg} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def main():
    print("Checking required Python packages...")
    missing = [p for p in REQUIRED if not is_installed(p)]
    if missing:
        print("Missing packages detected:")
        for m in missing:
            print(" -", m)
        ans = input("Install missing packages via pip now? [y/n]: ").strip().lower()
        if ans == "y":
            for m in missing:
                try:
                    pip_install(m)
                except Exception as e:
                    print(f"Failed to install {m} via pip: {e}")
            print("Pip installs attempted. Some packages may require additional system tools.")
        else:
            print("Skipped pip installs. You can install them later with pip.")
    else:
        print("All basic Python packages appear installed.")

    # PyTorch special-case
    try:
        import torch
        print(f"PyTorch version found: {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    except Exception:
        print("PyTorch not found.")
        ans = input("Install CPU-only PyTorch via pip now? (recommended to get started) [y/n]: ").strip().lower()
        if ans == "y":
            # CPU-only pip install
            try:
                pip_install("torch")
                pip_install("torchvision")
                print("Installed CPU-only torch. If you want GPU support, follow the instructions at https://pytorch.org/get-started/locally/")
            except Exception as e:
                print("Failed to install CPU-only torch via pip:", e)
                print("If you want GPU-enabled torch, please install via conda as described in README.")
        else:
            print("Skipped PyTorch install. If you want GPU support, use conda command in README.")

    # Kronos tokenizer (optional) — try to detect Kronos-specific package
    kronos_pkg_names = ["kronos", "Kronos", "kronos_tokenizer", "kronos_tokeniser"]
    found = any(is_installed(n) for n in kronos_pkg_names)
    if not found:
        print("\nKronos tokenizer package not detected.")
        print("If you plan to use Kronos tokenizer from GitHub, you can install it with:")
        print("  pip install git+https://github.com/shiyu-coder/Kronos.git")
        ans = input("Install Kronos repo (pip from GitHub) now? [y/n]: ").strip().lower()
        if ans == "y":
            try:
                pip_install("git+https://github.com/shiyu-coder/Kronos.git")
            except Exception as e:
                print("Failed to pip install Kronos repo:", e)
                print("You can install it manually or use the Hugging Face tokenizer if available.")
        else:
            print("Skipped Kronos repo install. The app will attempt to load tokenizer from Hugging Face if available.")

    print("\nSetup checker finished.")
    print("If you plan to use GPU (GTX1650), see README for the recommended conda command to install torch with CUDA support.")
    print("Done.")

if __name__ == "__main__":
    main()
