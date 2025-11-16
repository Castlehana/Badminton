# scripts/03_refine_and_make_clips.py  (T=16, causal right-aligned, peak EXCLUDED)
# - 이벤트 CSV가 start/end 구간을 포함하면: "해당 구간 내부"에서만 피크 탐색 → 스윙 5종 클립 생성
# - REC 바깥(complement) 구간에서 저속/안정 구간을 자동 샘플 → Idle 클립 생성
# - 관절: 16(R-Wrist), 14(R-Elbow), 12(R-Shoulder), 15(L-Wrist)
# - 정규화: 중심=R-Shoulder(12), 스케일=|Shoulder(12)-Elbow(14)| (상완 길이)
# - 특징 16D: [x,y, vx,vy, v, ax,ay, theta, front, d_ws, d_we, d_wl, dtheta, phi_shoulder(0), theta_rel, ang_elbow]
# - 출력: dataset/clips/raw/class=<CLS>/<base>_<startidx>.npz (X:(T,16), y, meta(json))

import os, sys, json, re
import numpy as np
import pandas as pd

# =========================
# 하이퍼파라미터 / 상수
# =========================
# (변경) T=16: 피크 '직전' 16프레임만 사용(피크 프레임 미포함)
T = 16
SEARCH_SEC_DEFAULT = 0.7
SEARCH_SEC_FALLBACK = 1.5
VIS_THR = 0.5           # mediapipe visibility 임계
WRIST_BAD_THR = 0.5     # 구간 내 wrist visibility < VIS_THR 비율이 이 값 이상이면 elbow 속도 사용

# Idle 자동 생성 파라미터
IDLE_ENABLED   = True
IDLE_RATIO     = 1.0      # Idle 개수 ≈ (스윙 개수 * 비율)
MARGIN_SEC     = 0.40     # REC 경계로부터 띄우는 마진
W_LOW          = 5        # 저속 판정 평균창 길이(프레임)
V_IDLE_MAX_W   = 0.25     # 손목 속도 상한(정규화/초)
V_IDLE_MAX_E   = 0.20     # 팔꿈치 속도 상한(정규화/초)

# Mediapipe Pose 인덱스
IDX_R_SHOULDER = 12
IDX_R_ELBOW    = 14
IDX_R_WRIST    = 16
IDX_L_WRIST    = 15

# 클래스(Idle 포함)
CLASSES = ['Clear', 'Drive', 'Drop', 'Under', 'Hairpin', 'Idle']
CLS2ID = {c:i for i,c in enumerate(CLASSES)}

# =========================
# 유틸리티
# =========================
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

def _wrap_pi(a): 
    return (a + np.pi) % (2*np.pi) - np.pi

def _get(df, key, fill=0.0):
    return df[key] if key in df.columns else pd.Series(fill, index=df.index, dtype='float64')

def _base_video_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def _subject_from_name(name: str) -> str:
    m = re.search(r'(S\d{3})', name) or re.search(r'(S\d+)', name)
    return m.group(1) if m else 'UNK'

# =========================
# 정규화(어깨 중심, 상완 길이)
# =========================
def _shoulder_norm(df: pd.DataFrame) -> pd.DataFrame:
    x12 = _get(df, 'x12', np.nan); y12 = _get(df, 'y12', np.nan)
    x14 = _get(df, 'x14', np.nan); y14 = _get(df, 'y14', np.nan)

    cx, cy = x12, y12
    scale = np.hypot(x12 - x14, y12 - y14) + 1e-6  # 상완 길이

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

# =========================
# 특징 16D 생성
# =========================
def _make_features(seg_n: pd.DataFrame) -> np.ndarray:
    arr = {k: seg_n[k].to_numpy() for k in seg_n.columns}
    t = _ensure_monotonic_t(arr['t'])

    # 기준: 오른손목
    x = arr['x16n']; y = arr['y16n']
    v, vx, vy = _speed(t, x, y)
    if v.size == 0:
        return np.zeros((len(seg_n), 16), dtype='float32')
    ax, ay = _accel(t, vx, vy)

    # 각도(어깨→손목)
    dx = arr['x16n'] - arr['x12n']; dy = arr['y16n'] - arr['y12n']
    theta = np.arctan2(dy, dx)
    dtheta = np.gradient(theta, t, edge_order=1)

    # front: 어깨 중심 정규화 → 손목 x가 0 이상이면 전방(오른쪽)
    front = (arr['x16n'] >= 0).astype(float)

    # 거리들
    d_ws = np.hypot(arr['x16n']-arr['x12n'], arr['y16n']-arr['y12n'])  # wrist-shoulder
    d_we = np.hypot(arr['x16n']-arr['x14n'], arr['y16n']-arr['y14n'])  # wrist-elbow
    d_wl = np.hypot(arr['x16n']-arr['x15n'], arr['y16n']-arr['y15n'])  # wrist-left_wrist

    # phi_shoulder 사용하지 않음 → 0, 상대각 = theta
    phi_shoulder = np.zeros_like(theta)
    theta_rel = _wrap_pi(theta - phi_shoulder)

    # 팔꿈치 내각(12-14-16)
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

# =========================
# 이벤트 로더 (구간/포인트 지원)
# =========================
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

# =========================
# 피크 탐색(포인트 라벨용)
# =========================
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
    k = idx[np.nanargmax((v_elbow if use_elbow else v_wrist)[idx])]
    return k, use_elbow

# =========================
# (변경) 피크로 '끝나는' 우측정렬 슬라이스 (피크 미포함 기본)
# =========================
def _slice_T_ending_at_peak(peak_idx, total_len, T, include_peak=False):
    """
    include_peak=False → [peak-T .. peak-1] 총 T개 (피크 미포함, 권장 설정)
    include_peak=True  → [peak-(T-1) .. peak] 총 T개 (피크 포함)
    """
    if include_peak:
        start = peak_idx - (T - 1)
        end   = peak_idx + 1
    else:
        start = peak_idx - T
        end   = peak_idx
    if start < 0 or end > total_len:
        return None
    return np.arange(start, end)

def _save_npz(out_dir, cls, lmk_name, start_idx, X, y, meta, tag=''):
    cls_dir = os.path.join(out_dir, f'class={cls}')
    os.makedirs(cls_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(lmk_name))[0]
    out_name = f'{base}_{start_idx}{tag}.npz'
    np.savez_compressed(os.path.join(cls_dir, out_name), X=X, y=y, meta=json.dumps(meta))

# =========================
# Idle 보완 구간 처리
# =========================
def _windows_from_complement(t, rec_list):
    """REC 구간(start,end)들의 보완(complement) 구간 반환"""
    if len(t) == 0:
        return []
    T0, T1 = float(t[0]), float(t[-1])
    rec = sorted(rec_list)  # [(s,e), ...]
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
    """최근 w프레임 평균 속도가 모두 낮으면 Idle 후보"""
    if k+1 < w:
        return False
    mw = float(np.mean(vw[k-w+1:k+1]))
    me = float(np.mean(ve[k-w+1:k+1]))
    return (mw <= vmax_w) and (me <= vmax_e)

def _gen_idle_clips(dfn, t, v_wrist, v_elbow, rec_intervals, n_target, out_dir, lmk_path, evt_csv, meta_base):
    idle_count = 0
    comp = _windows_from_complement(t, rec_intervals)   # [(start,end), ...]
    if not comp:
        return 0

    # 보완 구간에서 프레임 인덱스 마스크
    idx_comp = np.zeros(len(t), dtype=bool)
    for a, b in comp:
        idx_comp |= ((t >= a) & (t <= b))

    # 저속 프레임 인덱스 후보
    low_idx = [i for i in range(len(t)) if idx_comp[i] and _low_motion_mask(v_wrist, v_elbow, i)]
    if len(low_idx) == 0:
        return 0

    # 간격을 두고 고르게 샘플링
    step = max(1, len(low_idx) // max(1, n_target))
    picked = low_idx[::step][:max(1, n_target)]

    for k in picked:
        # (변경) 피크와 동일한 우측정렬(피크 미포함) 슬라이스 재사용
        idxs = _slice_T_ending_at_peak(k, len(t), T, include_peak=False)
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
            't_peak': float(t[k]),           # Idle은 의미적 피크 없음 → 중심시간 기록
            'start_idx': int(idxs[0]),
            'used_joint': 'none',
            # (추가) 정렬 정보
            'T': int(T),
            'feat_dim': int(X.shape[1]),
            'peak_pos': 'right-aligned',
            'causal': True,
            'include_peak': False
        })
        _save_npz(out_dir, 'Idle', lmk_path, int(idxs[0]), X, y, meta)
        idle_count += 1
        if idle_count >= n_target:
            break
    return idle_count

# =========================
# 메인 처리
# =========================
def process_pair(lmk_path, evt_csv, out_dir):
    df = pd.read_parquet(lmk_path)
    if 't' not in df.columns or len(df) == 0:
        print(f'[WARN] empty/invalid parquet -> skip: {os.path.basename(lmk_path)}'); return
    if len(df['t'].unique()) < 2:
        print(f'[WARN] too few timestamps (<2) -> skip: {os.path.basename(lmk_path)}'); return

    # 이벤트 로드
    try:
        mode, evs = _read_events(evt_csv)
    except Exception as e:
        print(f'[WARN] read events failed: {evt_csv} -> {e}')
        return
    if not evs:
        print(f'[WARN] no valid events -> skip: {os.path.basename(evt_csv)}')
        return

    # 정규화 좌표
    dfn = _shoulder_norm(df)
    t = dfn['t'].to_numpy()

    # 원 좌표(속도/가시도 계산용)
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

    # ===== 스윙(REC) 처리 =====
    for e in evs:
        cls = e['class']
        if cls not in CLS2ID or cls == 'Idle':
            continue

        # 구간 기반
        if mode == 'interval':
            start, end = float(e['start']), float(e['end'])
            idx = np.where((t >= start) & (t <= end))[0]
            if len(idx) == 0:
                continue
            # wrist 가시도 품질로 elbow 대체 여부 판단
            use_elbow = False
            bad = (v16[idx] < VIS_THR).mean() if v16 is not None and len(idx)>0 else 0.0
            if bad >= WRIST_BAD_THR:
                use_elbow = True
            seq = v_elbow if use_elbow else v_wrist
            k = idx[np.nanargmax(seq[idx])]
            t0_for_log = start

        # 포인트 라벨 모드
        else:
            t0 = float(e['t0'])
            k, use_elbow = _pick_peak(t, v_wrist, v_elbow, v16, t0)
            if k is None:
                continue
            t0_for_log = t0

        # (변경) 피크 직전 T=16 슬라이스 (피크 미포함)
        idxs = _slice_T_ending_at_peak(k, len(t), T, include_peak=False)
        if idxs is None:
            continue

        seg_n = dfn.iloc[idxs].reset_index(drop=True)
        X = _make_features(seg_n)                 # (T=16, 16)
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
            # (추가) 정렬/사양 정보
            'T': int(T),
            'feat_dim': int(X.shape[1]),
            'peak_pos': 'right-aligned',
            'causal': True,
            'include_peak': False
        }

        _save_npz(out_dir, cls, lmk_path, int(idxs[0]), X.astype('float32'), y, meta)
        refined_rows.append((round(t0_for_log,3), cls, round(float(t[k]),3),
                             'elbow' if use_elbow else 'wrist'))
        made_swing += 1

    # ===== Idle 자동 생성 =====
    if IDLE_ENABLED:
        rec_intervals = []
        if mode == 'interval':
            for e in evs:
                if e['class'] != 'Idle':
                    rec_intervals.append((float(e['start']), float(e['end'])))
        # target 계산
        target_idle = max(1, int(round(made_swing * IDLE_RATIO))) if made_swing > 0 else 0
        if target_idle > 0:
            meta_base = {
                'lmk_path': os.path.basename(lmk_path),
                'event_csv': os.path.basename(evt_csv),
                'subject': subject,
                'video': base,
                # (추가) 정렬/사양 기본값
                'T': int(T),
                'feat_dim': 16,
                'peak_pos': 'right-aligned',
                'causal': True,
                'include_peak': False
            }
            rc = _gen_idle_clips(dfn, t, v_wrist, v_elbow, rec_intervals, target_idle,
                                  out_dir, lmk_path, evt_csv, meta_base)
            print(f"[INFO] Idle clips: {rc} / target {target_idle}")

    # ===== 정제 로그 저장 =====
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
