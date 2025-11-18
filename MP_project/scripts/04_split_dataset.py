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

    
    # ============================================
    # 2) clips → train/val/test .npz 생성
    # ============================================
    import numpy as np

    # --- clip npz 로더 (너의 구조에 정확히 맞춘 버전) ---
    def load_clip_npz(path):
        data = np.load(path, allow_pickle=True)

        # 너의 clip 구조는 ['X', 'y', 'meta']
        # X: (T, D),  y: scalar, meta: scalar
        if 'X' not in data or 'y' not in data:
            raise KeyError(f"Unsupported clip format {path}, keys={data.files}")

        x = data['X']              # (T, D)
        y = int(data['y'])         # numpy scalar → int 변환
        return x, y


    # --- split 리스트에 따라 clip들을 하나로 모아주는 함수 ---
    def collect_clips(video_list):
        xs, ys = [], []

        target_T = None   # 기준 시퀀스 길이(T)
        feat_dim = None   # 기준 피처 차원(D)

        for video in video_list:
            # 예: S001_side → S001_side_*.npz
            pattern = f"dataset/clips/raw/class=*/{video}_*.npz"
            for path in glob.glob(pattern):
                x, y = load_clip_npz(path)    # x: (T,D)
                x = x.astype(np.float32)

                if x.ndim != 2:
                    # 혹시 모양이 이상한 경우 방어
                    print(f"[WARN] invalid clip shape {x.shape} in {path}, skip")
                    continue

                T, D = x.shape

                # 첫 샘플에서 기준 T, D를 잡는다
                if target_T is None:
                    target_T = T
                    feat_dim = D
                    print(f"[INFO] collect_clips: base shape set to T={target_T}, D={feat_dim}")

                # ---- 시간 길이(T) 보정 ----
                if T < target_T:
                    pad = np.zeros((target_T - T, D), dtype=np.float32)
                    x = np.concatenate([x, pad], axis=0)
                elif T > target_T:
                    x = x[:target_T, :]

                # ---- 피처 차원(D) 보정 ----
                if x.shape[1] < feat_dim:
                    pad = np.zeros((target_T, feat_dim - x.shape[1]), dtype=np.float32)
                    x = np.concatenate([x, pad], axis=1)
                elif x.shape[1] > feat_dim:
                    x = x[:, :feat_dim]

                xs.append(x)
                ys.append(y)

        if len(xs) == 0:
            return None, None

        xs = np.stack(xs)                     # (N, target_T, feat_dim)
        ys = np.array(ys, dtype=np.int64)     # (N,)
        return xs, ys



    # === video_fixed 기반 train/val/test 데이터 생성 ===
    tr_list = video_fixed['train']
    va_list = video_fixed['val']
    te_list = video_fixed['test']

    train_x, train_y = collect_clips(tr_list)
    val_x,   val_y   = collect_clips(va_list)
    test_x,  test_y  = collect_clips(te_list)

    # === npz 저장 ===
    if train_x is not None:
        np.savez("dataset/train.npz", x=train_x, y=train_y)
        print("[OK] train.npz saved:", train_x.shape)

    if val_x is not None:
        np.savez("dataset/val.npz", x=val_x, y=val_y)
        print("[OK] val.npz saved:", val_x.shape)

    if test_x is not None:
        np.savez("dataset/test.npz", x=test_x, y=test_y)
        print("[OK] test.npz saved:", test_x.shape)


if __name__ == "__main__":
    main()
