# scripts/03_refine_and_make_clips.py
# (T=25, asymmetric window = 16(before) + 1(peak) + 8(after), PEAK INCLUDED)
# - 스윙: 이벤트 구간 내부(또는 포인트 라벨 주변)에서 속도 피크 k를 찾고
#         [k-16 ... k ... k+8] 총 25프레임을 잘라 특징을 생성
# - Idle: 보완(complement) 구간의 저속 프레임 k를 골라 동일한 비율(16|1|8)로 슬라이싱
# - 정규화: 중심=R-Shoulder(12), 스케일=|Shoulder(12)-Elbow(14)|
# - 특징 16D: [x,y, vx,vy, v, ax,ay, theta, front, d_ws, d_we, d_wl, dtheta, phi_shoulder(0), theta_rel, ang_elbow]
# - 출력: dataset/clips/raw/class=<CLS>/<base>_<startidx>.npz (X:(T,16), y, meta(json))

import os, sys, json, re
import numpy as np
import pandas as pd

PRE_FR  = 16
POST_FR = 8
T = PRE_FR + 1 + POST_FR  # 25

SEARCH_SEC_DEFAULT  = 0.7
SEARCH_SEC_FALLBACK = 1.5
VIS_THR        = 0.5
WRIST_BAD_THR  = 0.5

IDLE_ENABLED = True
IDLE_RATIO   = 1.0
MARGIN_SEC   = 0.40
W_LOW        = 5
V_IDLE_MAX_W = 0.25
V_IDLE_MAX_E = 0.20

IDX_R_SHOULDER = 12
IDX_R_ELBOW    = 14
IDX_R_WRIST    = 16
IDX_L_WRIST    = 15

CLASSES = ['Clear', 'Drive', 'Drop', 'Under', 'Hairpin', 'Idle']
CLS2ID  = {c:i for i,c in enumerate(CLASSES)}

def _interp(a):
    if a.size == 0:
        return a
    s = pd.Series(a).interpolate(limit_direction='both')
    if s.isna().all():
        return np.zeros_like(a, dtype='float64')
    return s.to_numpy()

def _ensure_monotonic_t(t):
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
    v  = np.hypot(vx, vy)
    return v, vx, vy

def _accel(t, vx, vy):
    t = np.asarray(t)
    if t.size < 2: return np.array([]), np.array([])
    t = _ensure_monotonic_t(t)
    ax = np.gradient(vx, t, edge_order=1)
    ay = np.gradient(vy, t, edge_order=1)
    return ax, ay

def _wrap_pi(a): 
    return (a + np.pi) % (2*np.pi) - np.pi

def _get(df, key, fill=0.0):
    return df[key] if key in df.columns else pd.Series(fill, index=df.index, dtype='float64')

def _base_video_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def _subject_from_name(name: str) -> str:
    m = re.search(r'(S\d{3})', name) or re.search(r'(S\d+)', name)
    return m.group(1) if m else 'UNK'

def _shoulder_norm(df: pd.DataFrame) -> pd.DataFrame:
    x12 = _get(df, 'x12', np.nan); y12 = _get(df, 'y12', np.nan)
    x14 = _get(df, 'x14', np.nan); y14 = _get(df, 'y14', np.nan)

    cx, cy = x12, y12
    scale = np.hypot(x12 - x14, y12 - y14) + 1e-6

    out = {}
    need = [IDX_R_SHOULDER, IDX_R_ELBOW, IDX_R_WRIST, IDX_L_WRIST]
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

def _make_features(seg_n: pd.DataFrame) -> np.ndarray:
    arr = {k: seg_n[k].to_numpy() for k in seg_n.columns}
    t = _ensure_monotonic_t(arr['t'])

    x = arr['x16n']; y = arr['y16n']
    v, vx, vy = _speed(t, x, y)
    if v.size == 0:
        return np.zeros((len(seg_n), 16), dtype='float32')
    ax, ay = _accel(t, vx, vy)

    dx = arr['x16n'] - arr['x12n']; dy = arr['y16n'] - arr['y12n']
    theta  = np.arctan2(dy, dx)
    dtheta = np.gradient(theta, t, edge_order=1)
    front = (arr['x16n'] >= 0).astype(float)

    d_ws = np.hypot(arr['x16n']-arr['x12n'], arr['y16n']-arr['y12n'])
    d_we = np.hypot(arr['x16n']-arr['x14n'], arr['y16n']-arr['y14n'])
    d_wl = np.hypot(arr['x16n']-arr['x15n'], arr['y16n']-arr['y15n'])

    phi_shoulder = np.zeros_like(theta)
    theta_rel = _wrap_pi(theta - phi_shoulder)

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
        x, y, vx, vy, v, ax, ay, theta, front, d_ws, d_we, d_wl,
        dtheta, phi_shoulder, theta_rel, ang_elbow
    ], axis=1).astype('float32')
    return feats

def _read_events(evt_csv):
    ev = pd.read_csv(evt_csv)
    if {'start_sec','end_sec','class'}.issubset(ev.columns) and len(ev)>0:
        mode = 'interval'
        events = [
            {'class': str(r['class']), 'start': float(r['start_sec']), 'end': float(r['end_sec'])}
            for _, r in ev.iterrows() if float(r['end_sec']) > float(r['start_sec'])
        ]
    elif {'timestamp_sec','class'}.issubset(ev.columns) and len(ev)>0:
        mode = 'point'
        events = [
            {'class': str(r['class']), 't0': float(r['timestamp_sec'])}
            for _, r in ev.iterrows()
        ]
    else:
        raise RuntimeError(f'invalid events CSV schema: {evt_csv}')
    return mode, events

def _pick_peak(t, v_wrist, v_elbow, vis_wrist, t0):
    def _search(win):
        lo, hi = t0 - win, t0 + win
        return np.where((t >= lo) & (t <= hi))[0]
    idx = _search(SEARCH_SEC_DEFAULT)
    use_elbow = False
    if len(idx) == 0:
        idx = _search(SEARCH_SEC_FALLBACK)
        if len(idx) == 0: 
            return None, None
    if vis_wrist is not None and len(idx) > 0:
        bad = (vis_wrist[idx] < VIS_THR).mean()
        if bad >= WRIST_BAD_THR:
            use_elbow = True
    seq = v_elbow if use_elbow else v_wrist
    k = idx[np.nanargmax(seq[idx])]
    return k, use_elbow

def _slice_asym_around_peak(peak_idx, total_len, pre=PRE_FR, post=POST_FR, include_peak=True):
    if include_peak:
        start = peak_idx - pre
        end   = peak_idx + post + 1
    else:
        start = peak_idx - pre
        end   = peak_idx + post
    if start < 0 or end > total_len:
        return None
    return np.arange(start, end)

def _save_npz(out_dir, cls, lmk_name, start_idx, X, y, meta, tag=''):
    cls_dir = os.path.join(out_dir, f'class={cls}')
    os.makedirs(cls_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(lmk_name))[0]
    out_name = f'{base}_{start_idx}{tag}.npz'
    np.savez_compressed(os.path.join(cls_dir, out_name), X=X, y=y, meta=json.dumps(meta))

def _windows_from_complement(t, rec_list):
    if len(t) == 0:
        return []
    T0, T1 = float(t[0]), float(t[-1])
    rec = sorted(rec_list)
    out = []
    cur = T0
    for s, e in rec:
        s2 = max(T0, s - MARGIN_SEC)
        e2 = min(T1, e + MARGIN_SEC)
        if cur < s2:
            out.append((cur, s2))
        cur = max(cur, e2)
    if cur < T1:
        out.append((cur, T1))
    return [(a, b) for a, b in out if (b - a) > 2 * MARGIN_SEC]

def _low_motion_mask(vw, ve, k, w=W_LOW, vmax_w=V_IDLE_MAX_W, vmax_e=V_IDLE_MAX_E):
    if k+1 < w:
        return False
    mw = float(np.mean(vw[k-w+1:k+1]))
    me = float(np.mean(ve[k-w+1:k+1]))
    return (mw <= vmax_w) and (me <= vmax_e)

def _gen_idle_clips(dfn, t, v_wrist, v_elbow, rec_intervals, n_target, out_dir, lmk_path, evt_csv, meta_base):
    idle_count = 0
    comp = _windows_from_complement(t, rec_intervals)
    if not comp:
        return 0

    idx_comp = np.zeros(len(t), dtype=bool)
    for a, b in comp:
        idx_comp |= ((t >= a) & (t <= b))

    low_idx = [i for i in range(len(t)) if idx_comp[i] and _low_motion_mask(v_wrist, v_elbow, i)]
    if len(low_idx) == 0:
        return 0

    step = max(1, len(low_idx) // max(1, n_target))
    picked = low_idx[::step][:max(1, n_target)]

    for k in picked:
        idxs = _slice_asym_around_peak(k, len(t), pre=PRE_FR, post=POST_FR, include_peak=True)
        if idxs is None:
            continue
        seg_n = dfn.iloc[idxs].reset_index(drop=True)
        X = _make_features(seg_n).astype('float32')
        y = np.int64(CLS2ID['Idle'])

        meta = dict(meta_base)
        meta.update({
            'class': 'Idle',
            'class_id': int(y),
            't_label': float(t[k]),
            't_peak': float(t[k]),
            'start_idx': int(idxs[0]),
            'used_joint': 'none',
            'T': int(T),
            'feat_dim': int(X.shape[1]),
            'peak_pos': 'asymmetric(16|1|8)',
            'pre': int(PRE_FR),
            'post': int(POST_FR),
            'causal': False,
            'include_peak': True
        })
        _save_npz(out_dir, 'Idle', lmk_path, int(idxs[0]), X, y, meta)
        idle_count += 1
        if idle_count >= n_target:
            break
    return idle_count

def process_pair(lmk_path, evt_csv, out_dir):
    df = pd.read_parquet(lmk_path)
    if 't' not in df.columns or len(df) == 0:
        print(f'[WARN] empty/invalid parquet -> skip: {os.path.basename(lmk_path)}'); return
    if len(df['t'].unique()) < 2:
        print(f'[WARN] too few timestamps (<2) -> skip: {os.path.basename(lmk_path)}'); return

    try:
        mode, evs = _read_events(evt_csv)
    except Exception as e:
        print(f'[WARN] read events failed: {evt_csv} -> {e}')
        return
    if not evs:
        print(f'[WARN] no valid events -> skip: {os.path.basename(evt_csv)}')
        return

    dfn = _shoulder_norm(df)
    t = dfn['t'].to_numpy()

    x16 = _get(df, 'x16', np.nan).to_numpy()
    y16 = _get(df, 'y16', np.nan).to_numpy()
    v16 = _get(df, 'v16', 1.0).to_numpy()
    x14 = _get(df, 'x14', np.nan).to_numpy()
    y14 = _get(df, 'y14', np.nan).to_numpy()

    v_wrist, _, _ = _speed(t, x16, y16)
    v_elbow, _, _ = _speed(t, x14, y14)
    if v_wrist.size == 0 and v_elbow.size == 0:
        print(f'[WARN] cannot compute speed (len<2) -> skip: {os.path.basename(lmk_path)}')
        return

    base = _base_video_name(lmk_path)
    subject = _subject_from_name(base)

    refined_rows = [('t_label_or_start','class','t_peak','used_joint')]
    made_swing = 0

    for e in evs:
        cls = e['class']
        if cls not in CLS2ID or cls == 'Idle':
            continue

        if mode == 'interval':
            start, end = float(e['start']), float(e['end'])
            idx = np.where((t >= start) & (t <= end))[0]
            if len(idx) == 0:
                continue
            use_elbow = False
            bad = (v16[idx] < VIS_THR).mean() if v16 is not None and len(idx)>0 else 0.0
            if bad >= WRIST_BAD_THR:
                use_elbow = True
            seq = v_elbow if use_elbow else v_wrist
            k = idx[np.nanargmax(seq[idx])]
            t0_for_log = start
        else:
            t0 = float(e['t0'])
            k, use_elbow = _pick_peak(t, v_wrist, v_elbow, v16, t0)
            if k is None:
                continue
            t0_for_log = t0

        idxs = _slice_asym_around_peak(k, len(t), pre=PRE_FR, post=POST_FR, include_peak=True)
        if idxs is None:
            continue

        seg_n = dfn.iloc[idxs].reset_index(drop=True)
        X = _make_features(seg_n)
        y = np.int64(CLS2ID[cls])
        meta = {
            'class': cls,
            'class_id': int(y),
            'lmk_path': os.path.basename(lmk_path),
            'event_csv': os.path.basename(evt_csv),
            'subject': subject,
            'video': base,
            't_label': float(t0_for_log),
            't_peak': float(t[k]),
            'start_idx': int(idxs[0]),
            'used_joint': 'elbow' if use_elbow else 'wrist',
            'T': int(T),
            'feat_dim': int(X.shape[1]),
            'peak_pos': 'asymmetric(16|1|8)',
            'pre': int(PRE_FR),
            'post': int(POST_FR),
            'causal': False,
            'include_peak': True
        }
        _save_npz(out_dir, cls, lmk_path, int(idxs[0]), X.astype('float32'), y, meta)
        refined_rows.append((round(t0_for_log,3), cls, round(float(t[k]),3),
                             'elbow' if use_elbow else 'wrist'))
        made_swing += 1

    if IDLE_ENABLED:
        rec_intervals = []
        if mode == 'interval':
            for e in evs:
                if e['class'] != 'Idle':
                    rec_intervals.append((float(e['start']), float(e['end'])))
        target_idle = max(1, int(round(made_swing * IDLE_RATIO))) if made_swing > 0 else 0
        if target_idle > 0:
            meta_base = {
                'lmk_path': os.path.basename(lmk_path),
                'event_csv': os.path.basename(evt_csv),
                'subject': subject,
                'video': base,
                'T': int(T),
                'feat_dim': 16,
                'peak_pos': 'asymmetric(16|1|8)',
                'pre': int(PRE_FR),
                'post': int(POST_FR),
                'causal': False,
                'include_peak': True
            }
            rc = _gen_idle_clips(dfn, t, v_wrist, v_elbow, rec_intervals, target_idle,
                                  out_dir, lmk_path, evt_csv, meta_base)
            print(f"[INFO] Idle clips: {rc} / target {target_idle}")

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
