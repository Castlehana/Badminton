import os, glob, re, json, numpy as np

ROOT = "dataset/clips/raw"
CLS_NAMES = ["Clear","Drive","Drop","Under","Hairpin","Idle"]

def subject_from_path(p:str):
    m = re.search(r"(S\d{3})", p)
    return m.group(1) if m else "UNK"

def main():
    per_class = {c:0 for c in CLS_NAMES}
    per_subject = {}
    files = sorted(glob.glob(os.path.join(ROOT, "class=*","*.npz")))
    if not files:
        print(f"[WARN] No clips found under {ROOT}")
        return

    for p in files:
        m = re.search(r"class=([^/\\]+)", p)
        cls = m.group(1) if m else "UNK"
        per_class.setdefault(cls, 0)
        per_class[cls] += 1

        s = subject_from_path(p)
        per_subject.setdefault(s, {c:0 for c in CLS_NAMES})
        per_subject[s].setdefault(cls, 0)
        per_subject[s][cls] += 1

    total = sum(per_class.values())
    print("per-class counts:", per_class, " total:", total)

    subs = sorted(per_subject.keys())
    print("\nby-subject (counts per class):")
    header = "subject".ljust(8) + "".join(c.center(10) for c in CLS_NAMES) + " | total"
    print(header)
    print("-"*len(header))
    for s in subs:
        row_total = sum(per_subject[s].get(c,0) for c in CLS_NAMES)
        row = s.ljust(8) + "".join(str(per_subject[s].get(c,0)).center(10) for c in CLS_NAMES) + f" | {row_total}"
        print(row)

    avg = total / max(len(CLS_NAMES),1)
    for c, n in per_class.items():
        if n == 0:
            print(f"[ALERT] class '{c}' has 0 samples")
        elif n < 0.5 * avg:
            print(f"[WARN] class '{c}' count is low ({n} < 50% of mean {avg:.1f})")

    events_refined = sorted(glob.glob("dataset/events/*_refined.csv"))
    if not events_refined:
        print("\n[INFO] No refined CSV found in dataset/events; run 03_refine_and_make_clips.py first.")
    else:
        print(f"\n[OK] refined CSV files: {len(events_refined)}")

if __name__ == "__main__":
    main()
