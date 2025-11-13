# scripts/04_split_dataset.py
# - 영상 베이스이름(예: S001_side) 단위로 분할 생성
# - video_lovo(Leave-One-Video-Out) + video_fixed(8:1:1) 동시 저장
# - clips가 만들어진 뒤 실행

import os, json, re, glob, random
from typing import List, Dict
random.seed(42)

def video_from_path(p: str) -> str:
    # class=*/S001_side_123.npz -> S001_side
    base = os.path.splitext(os.path.basename(p))[0]
    m = re.match(r'([A-Za-z0-9]+_[A-Za-z0-9]+)', base)  # S001_side
    return m.group(1) if m else base

def discover_videos() -> List[str]:
    paths = glob.glob('dataset/clips/raw/class=*/*.npz')
    vids = sorted({video_from_path(p) for p in paths})
    return vids

def make_video_fixed(vids: List[str]) -> Dict[str, List[str]]:
    vids = vids[:]
    random.shuffle(vids)
    n = len(vids)
    if n == 0:
        raise SystemExit("[ERR] no videos found under dataset/clips/raw")
    if n == 1:
        return {'train': [vids[0]], 'val': [], 'test': []}
    if n == 2:
        return {'train': [vids[0]], 'val': [], 'test': [vids[1]]}
    n_tr = max(1, int(n*0.8))
    n_val = max(1, int(n*0.1))
    n_te = n - n_tr - n_val
    if n_te <= 0:
        if n_val >= 2: n_val -= 1; n_te = 1
        else: n_tr = max(1, n_tr-1); n_te = 1
    fixed = {
        'train': vids[:n_tr],
        'val':   vids[n_tr:n_tr+n_val],
        'test':  vids[n_tr+n_val:n_tr+n_val+n_te]
    }
    if len(fixed['test']) == 0:
        n_tr = max(1, int(round(n*0.7)))
        fixed = {'train': vids[:n_tr], 'val': [], 'test': vids[n_tr:]}
    return fixed

def make_video_lovo(vids: List[str]) -> List[Dict]:
    folds = []
    for i, v_test in enumerate(vids):
        train = [x for x in vids if x != v_test]
        v_val = train[i % len(train)] if train else v_test
        folds.append({
            'name': v_test,
            'train': [x for x in train if x != v_val],
            'val':   [v_val],
            'test':  [v_test]
        })
    return folds

def main():
    vids = discover_videos()
    video_fixed = make_video_fixed(vids)
    video_lovo  = make_video_lovo(vids) if len(vids) >= 2 else []

    os.makedirs('dataset/meta', exist_ok=True)
    out = {'video_fixed': video_fixed}
    if video_lovo:
        out['video_lovo'] = video_lovo

    with open('dataset/meta/splits.json','w',encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("[OK] videos:", vids)
    print("[OK] video_fixed:", video_fixed)
    if video_lovo:
        print(f"[OK] video_lovo: {len(video_lovo)} folds")

if __name__ == "__main__":
    main()
