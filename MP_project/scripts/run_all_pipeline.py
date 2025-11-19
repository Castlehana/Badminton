# scripts/run_all_pipeline.py
import subprocess
import sys
import os

STEPS = [
    ("03_all",         ["python", "scripts/run_03_all.py"]),
    ("check_counts",   ["python", "scripts/check_counts.py"]),
    ("split_dataset",  ["python", "scripts/04_split_dataset.py"]),
    ("train_tcn",      ["python", "scripts/05_train_tcn.py"]),
    ("export_onnx",    ["python", "scripts/06_export_onnx.py"]),
]

def run_step(name, cmd):
    print(f"\n==============================")
    print(f"[RUN] Step: {name}")
    print(f"==============================\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"[OK] Step '{name}' completed.\n")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Step '{name}' failed with error:")
        print(e)
        sys.exit(1)

def main():
    print("\n###########################################")
    print("#   FULL PIPELINE EXECUTION STARTED       #")
    print("###########################################\n")

    for name, cmd in STEPS:
        run_step(name, cmd)

    print("\n###########################################")
    print("#   ALL STEPS COMPLETED SUCCESSFULLY      #")
    print("###########################################\n")

if __name__ == "__main__":
    main()
