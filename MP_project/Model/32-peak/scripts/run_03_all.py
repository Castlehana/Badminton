# landmarks/*.parquet ↔ events/*.csv 이름 매칭 후
# 03_refine_and_make_clips.py를 일괄 실행해 T=33 클립 생성

import os, glob, subprocess, sys

LAND_DIR = "dataset/landmarks"
EVT_DIR  = "dataset/events"
OUT_DIR  = "dataset/clips/raw"
SCRIPT   = "scripts/03_refine_and_make_clips.py"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    lands = sorted(glob.glob(os.path.join(LAND_DIR, "*.parquet")))
    if not lands:
        print(f"[run_03_all] No parquet files in {LAND_DIR}")
        sys.exit(1)

    failed, skipped = [], []
    for lp in lands:
        base = os.path.splitext(os.path.basename(lp))[0]
        ev = os.path.join(EVT_DIR, f"{base}.csv")
        if not os.path.exists(ev):
            print(f"[SKIP] events csv not found for {base}: {ev}")
            skipped.append(base)
            continue

        cmd = ["python", SCRIPT, lp, ev, OUT_DIR]
        print(f"[run_03_all] {base}: {os.path.relpath(lp)} + {os.path.relpath(ev)} -> {OUT_DIR}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {base} failed: {e}")
            failed.append(base)

    print("\n[SUMMARY]")
    print(f" - total landmarks: {len(lands)}")
    print(f" - skipped (no csv): {len(skipped)}")
    if skipped:
        for b in skipped: print(f"   * {b}")
    print(f" - failed: {len(failed)}")
    if failed:
        for b in failed: print(f"   * {b}")
    if failed:
        sys.exit(1)

if __name__ == "__main__":
    main()
