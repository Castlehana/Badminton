# scripts/03_refine_and_make_clips.py
# - 누락 랜드마크 안전 처리(x11 등)
# - 빈/짧은 시퀀스 skip
# - 토르소(12~24) 정규화
# - 16D 특징: [x,y, vx,vy, v, ax,ay, theta, front, d_ws,d_we,d_wh, dtheta, phi_shoulder, theta_rel, ang_elbow]
# - 디스크 증강 OFF (학습 시 온라인 증강만 사용)
# - meta에 subject(S###), video(베이스파일명), peak_pos="center" 포함
# - ★ 변경점: 피크 프레임을 "가운데"로 포함하는 센터-정렬(center-aligned) 슬라이스 [k-16, k+16], 총 T=33

import os, sys, json, re
import numpy as np
import pandas as pd

# --------------------
# 하이퍼파라미터
# --------------------
T = 33  # ← 32 → 33 (피크를 중심으로 이전/이후 16프레임)
SEARCH_SEC_DEFAULT = 0.7
SEARCH_SEC_FALLBACK = 1.5
VIS_THR = 0.5
WRIST_BAD_THR = 0.5
DO_AUG_SHIFT = False   # ← 디스크 증강 끔
AUG_SHIFT = 2

# MediaPipe Pose 인덱스
IDX_NOSE = 0
IDX_L_SHOULDER = 11
IDX_R_SHOULDER = 12
IDX_R_ELBOW = 14
IDX_R_WRIST = 16
IDX_L_HIP = 23
IDX_R_HIP = 24

CLASSES = ['Clear', 'Drive', 'Drop', 'Under', 'Hairpin']
CLS2ID = {c:i for i,c in enumerate(CLASSES)}

# --------------------
# 유틸
# --------------------
def _interp(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return a
    s = pd.Series(a).interpolate(limit_direction='both')
    if s.isna().all():
        return np.zeros_like(a, dtype='float64')
    return s.to_numpy()

def _ensure_monotonic_t(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t).astype('float64')
    if t.size >= 2 and (np.diff(t) <= 0).any():
        t = t + np.linspace(0, 1e-6, len(t))
    return t

def _speed(t, x, y):
    t = np.asarray(t)
    if t.size < 2: return np.array([]), np.array([]), np.array([])
    xi, yi = _interp(x), _interp(y)
    t = _ensure_monotonic_t(t)
    vx = np.gradient(xi, t, edge_order=1)
    vy = np.gradient(yi, t, edge_order=1)
    v = np.hypot(vx, vy)
    return v, vx, vy

def _accel(t, vx, vy):
    t = np.asarray(t)
    if t.size < 2: return np.array([]), np.array([])
    t = _ensure_monotonic_t(t)
    ax = np.gradient(vx, t, edge_order=1)
    ay = np.gradient(vy, t, edge_order=1)
    return ax, ay

def _wrap_pi(a): return (a + np.pi) % (2*np.pi) - np.pi

def _get(df, key, fill=0.0):
    return df[key] if key in df.columns else pd.Series(fill, index=df.index, dtype='float64')

def _base_video_name(path: str) -> str:
    # landmarks parquet 파일명(확장자 제외)이 영상 베이스이름과 동일하다고 가정 (예: S001_side)
    return os.path.splitext(os.path.basename(path))[0]

def _subject_from_name(name: str) -> str:
    m = re.search(r'(S\d{3})', name) or re.search(r'(S\d+)', name)
    return m.group(1) if m else 'UNK'

# --------------------
# 정규화(토르소 기준)
# --------------------
def _torso_norm(df: pd.DataFrame) -> pd.DataFrame:
    x12 = _get(df, 'x12', np.nan); y12 = _get(df, 'y12', np.nan)
    x24 = _get(df, 'x24', np.nan); y24 = _get(df, 'y24', np.nan)
    cx = (x12 + x24) / 2.0
    cy = (y12 + y24) / 2.0
    scale = np.hypot(x12 - x24, y12 - y24) + 1e-6

    out = {}
    need = [IDX_NOSE, IDX_L_SHOULDER, IDX_R_SHOULDER, IDX_R_ELBOW, IDX_R_WRIST, IDX_L_HIP, IDX_R_HIP]
    for j in need:
        xj = _get(df, f'x{j}', np.nan)
        yj = _get(df, f'y{j}', np.nan)
        vj = _get(df, f'v{j}', 1.0)
        out[f'x{j}n'] = (xj - cx) / scale
        out[f'y{j}n'] = (yj - cy) / scale
        out[f'v{j}']  = vj

    if 't' not in df.columns:
        raise RuntimeError('parquet에 t 컬럼이 필요합니다.')
    out['t'] = df['t']
    return pd.DataFrame(out)

def _compute_phi_shoulder(arr):
    x11n = arr.get('x11n'); y11n = arr.get('y11n')
    x12n = arr.get('x12n'); y12n = arr.get('y12n')
    x24n = arr.get('x24n'); y24n = arr.get('y24n')
    if x11n is not None and y11n is not None:
        if not (np.all(np.isnan(x11n)) or np.all(np.isnan(y11n))):
            dx_sh = x12n - x11n; dy_sh = y12n - y11n
            return np.arctan2(dy_sh, dx_sh)
    dx_torso = x24n - x12n; dy_torso = y24n - y12n
    return np.arctan2(-dy_torso, dx_torso)

# --------------------
# 16D 특징
# --------------------
def _make_features(seg_n: pd.DataFrame) -> np.ndarray:
    arr = {k: seg_n[k].to_numpy() for k in seg_n.columns}
    t = _ensure_monotonic_t(arr['t'])

    x = arr['x16n']; y = arr['y16n']
    v, vx, vy = _speed(t, x, y)
    if v.size == 0:
        return np.zeros((len(seg_n), 16), dtype='float32')
    ax, ay = _accel(t, vx, vy)

    dx = arr['x16n'] - arr['x12n']; dy = arr['y16n'] - arr['y12n']
    theta = np.arctan2(dy, dx); dtheta = np.gradient(theta, t, edge_order=1)

    # 어깨 중점 기준 좌우: 왼/오 어깨 평균 사용
    x_mid_sh = (arr['x11n'] + arr['x12n']) * 0.5
    front = (arr['x16n'] >= x_mid_sh).astype(float)

    d_ws = np.hypot(arr['x16n']-arr['x12n'], arr['y16n']-arr['y12n'])
    d_we = np.hypot(arr['x16n']-arr['x14n'], arr['y16n']-arr['y14n'])
    d_wh = np.hypot(arr['x16n']-arr['x24n'], arr['y16n']-arr['y24n'])

    phi_shoulder = _compute_phi_shoulder(arr)
    theta_rel = _wrap_pi(theta - phi_shoulder)

    # 팔꿈치 관절각(12-14-16)
    sx, sy = arr['x12n'], arr['y12n']
    ex, ey = arr['x14n'], arr['y14n']
    wx, wy = arr['x16n'], arr['y16n']
    v_se = np.stack([ex - sx, ey - sy], axis=1)
    v_we = np.stack([wx - ex, wy - ey], axis=1)
    dot = np.sum(v_se * v_we, axis=1)
    n1 = np.linalg.norm(v_se, axis=1) + 1e-6
    n2 = np.linalg.norm(v_we, axis=1) + 1e-6
    cosang = np.clip(dot/(n1*n2), -1.0, 1.0)
    ang_elbow = np.arccos(cosang)

    feats = np.stack([
        x, y, vx, vy, v, ax, ay, theta, front, d_ws, d_we, d_wh,
        dtheta, phi_shoulder, theta_rel, ang_elbow
    ], axis=1).astype('float32')
    return feats

# --------------------
# 피크 탐색/절단
# --------------------
def _pick_peak(t, v_wrist, v_elbow, vis_wrist, t0):
    """t0(라벨 근처)에서 wrist 우선/가시도 불량 시 elbow로 속도 피크 선택"""
    def _search(win):
        lo, hi = t0 - win, t0 + win
        return np.where((t >= lo) & (t <= hi))[0]
    idx = _search(SEARCH_SEC_DEFAULT)
    use_elbow = False
    if len(idx) == 0:
        idx = _search(SEARCH_SEC_FALLBACK)
        if len(idx) == 0: return None, None
    if vis_wrist is not None and len(idx) > 0:
        bad = (vis_wrist[idx] < VIS_THR).mean()
        if bad >= WRIST_BAD_THR:
            use_elbow = True
    k = idx[np.nanargmax((v_elbow if use_elbow else v_wrist)[idx])]
    return k, use_elbow

def _slice_T_centered_at_peak(peak_idx, total_len, T):
    """피크 프레임을 가운데로 포함하는 센터-정렬 구간 [k-16, k+16] (총 33프레임)"""
    half = T // 2  # 16
    start = peak_idx - half
    end   = peak_idx + half + 1  # 포함 끝(+1)
    if start < 0 or end > total_len: return None
    return np.arange(start, end)

def _save_npz(out_dir, cls, lmk_name, start_idx, X, y, meta, tag=''):
    cls_dir = os.path.join(out_dir, f'class={cls}')
    os.makedirs(cls_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(lmk_name))[0]
    out_name = f'{base}_{start_idx}{tag}.npz'
    np.savez_compressed(os.path.join(cls_dir, out_name), X=X, y=y, meta=json.dumps(meta))

# --------------------
# 메인 처리
# --------------------
def process_pair(lmk_path, evt_csv, out_dir):
    df = pd.read_parquet(lmk_path)
    if 't' not in df.columns or len(df) == 0:
        print(f'[WARN] empty/invalid parquet -> skip: {os.path.basename(lmk_path)}'); return
    if len(df['t'].unique()) < 2:
        print(f'[WARN] too few timestamps (<2) -> skip: {os.path.basename(lmk_path)}'); return

    # 관측 품질 확인
    wr_ok = (~_get(df, 'x16', np.nan).isna()) & (~_get(df, 'y16', np.nan).isna())
    el_ok = (~_get(df, 'x14', np.nan).isna()) & (~_get(df, 'y14', np.nan).isna())
    if (wr_ok | el_ok).mean() < 0.05:
        print(f'[WARN] almost all wrist/elbow NaN -> skip: {os.path.basename(lmk_path)}'); return

    # 이벤트 로드
    ev = pd.read_csv(evt_csv)
    if not {'timestamp_sec','class'}.issubset(ev.columns) or len(ev) == 0:
        print(f'[WARN] invalid/empty events -> skip: {os.path.basename(evt_csv)}'); return

    # 정규화
    dfn = _torso_norm(df)
    t = dfn['t'].to_numpy()

    # 원 좌표(속도 계산용)
    x16 = _get(df, 'x16', np.nan).to_numpy()
    y16 = _get(df, 'y16', np.nan).to_numpy()
    v16 = _get(df, 'v16', 1.0).to_numpy()
    x14 = _get(df, 'x14', np.nan).to_numpy()
    y14 = _get(df, 'y14', np.nan).to_numpy()

    v_wrist, _, _ = _speed(t, x16, y16)
    v_elbow, _, _ = _speed(t, x14, y14)
    if v_wrist.size == 0 and v_elbow.size == 0:
        print(f'[WARN] cannot compute speed (len<2) -> skip: {os.path.basename(lmk_path)}'); return

    base = _base_video_name(lmk_path)
    subject = _subject_from_name(base)

    refined_rows = [('t_label','class','t_peak','used_joint')]
    for _, row in ev.iterrows():
        cls = str(row['class'])
        if cls not in CLS2ID: continue
        t0 = float(row['timestamp_sec'])

        k, used_elbow = _pick_peak(t, v_wrist, v_elbow, v16, t0)
        if k is None: continue

        # ★ 피크를 가운데로 포함하는 센터-정렬 슬라이스
        idxs = _slice_T_centered_at_peak(k, len(t), T)
        if idxs is None:
            continue

        seg_n = dfn.iloc[idxs].reset_index(drop=True)
        X = _make_features(seg_n)                 # (T=33, 16)
        y = np.int64(CLS2ID[cls])
        meta = {
            'class': cls,
            'class_id': int(y),
            'lmk_path': os.path.basename(lmk_path),
            'event_csv': os.path.basename(evt_csv),
            'subject': subject,
            'video': base,          # 영상 베이스이름
            't_label': float(t0),
            't_peak': float(t[k]),
            'start_idx': int(idxs[0]),
            'used_joint': 'elbow' if used_elbow else 'wrist',
            'T': int(T),
            'feat_dim': int(X.shape[1]),
            'peak_pos': 'center'    # ← 학습/추론 정렬 확인용 메타
        }

        _save_npz(out_dir, cls, lmk_path, int(idxs[0]), X.astype('float32'), y, meta)
        refined_rows.append((round(t0,3), cls, round(float(t[k]),3), 'elbow' if used_elbow else 'wrist'))

        # 디스크 증강 OFF (학습시 온라인 증강 사용)

    if len(refined_rows) > 1:
        pd.DataFrame(refined_rows[1:], columns=refined_rows[0]).to_csv(
            evt_csv.replace('.csv','_refined.csv'),
            index=False, encoding='utf-8'
        )

def main():
    if len(sys.argv) < 4:
        print('Usage: python scripts/03_refine_and_make_clips.py <landmarks.parquet> <events.csv> <out_dir>')
        sys.exit(1)
    lmk_path, evt_csv, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    process_pair(lmk_path, evt_csv, out_dir)
    print('done:', out_dir)

if __name__ == '__main__':
    main()
