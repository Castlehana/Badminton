# scripts/pose_infer_onnx_videotest.py
# (Updated for causal right-aligned windows; T=16, peak EXCLUDED)
# - 학습/클립 규칙: "피크 직전 16프레임만" 사용(피크 미포함), target_T=16
# - 클래스: 학습 메타(tcn_meta.json)의 "classes"를 그대로 따름
#   · 보통 ['Clear','Drive','Drop','Under','Hairpin','Idle']
#   · 과거 Ready 사용 버전 호환: 'Ready'가 없으면 'Idle'을 '비피크 상태'로 간주
# - 결정 로직(하이브리드):
#     · 피크일 때 → 스윙 5종만 확정(softmax max ≥ --th & cooldown 충족) → UDP
#     · 피크가 아닐 때 → Ready/Idle 추정치가 임계 이상이면 상태 업데이트(UDP는 선택적)
# - 메타(tcn_meta.json): {"classes":[...],"feat_dim":16,"target_T":16,"input_name":"clips","output_name":"logits","zscore_mu":[...],"zscore_std":[...]}

import argparse, time, json, socket, math, sys, os, glob
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort

# ===================== 상수/기본값 =====================
DEF_T    = 16              # 학습/클립 길이(03/05와 일치; ckpt 메타가 우선)
DEF_TH   = 0.80            # 스윙 확신도 임계
DEF_CD   = 0.80            # 스윙 확정 쿨다운(초)
UDP_IP   = "127.0.0.1"
UDP_PORT = 5052
EPS      = 1e-6

# 비피크(준비/대기) 관련
TH_READY        = 0.60     # 비피크 시 Ready/Idle 확정 최소 확률
SEND_READY_UDP  = False    # True면 Ready/Idle도 UDP로 알림(swing=False)

# 실시간 dt 안정화(웹캠 권장)
TARGET_FPS = 30.0
TARGET_DT  = 1.0 / TARGET_FPS
EMA_ALPHA  = 0.20
DT_MIN, DT_MAX = 1/90.0, 1/20.0  # 90~20fps

# 피크 검출 파라미터(정규화 좌표계 기준)
PEAK_WIN        = 5          # 최근 5프레임 창(센터 = 최신 프레임)
V_MIN_WRIST     = 0.80       # 손목 속도 최소(정규화/초)
V_MIN_ELBOW     = 0.50       # 팔꿈치 속도 최소
PROM_MIN        = 0.05       # 돌출도(피크 - 인접값)
VIS_THR         = 0.60       # 가시도 임계
USE_ELBOW_RATIO = 0.5        # 창 내 손목 가시도 비율이 낮으면 팔꿈치 사용

# ===================== 유틸 =====================
def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z); p /= (p.sum(axis=1, keepdims=True) + EPS)
    return p

def to_3d_btd(x: np.ndarray, T: int, D: int) -> np.ndarray:
    if x.ndim == 3: return x
    if x.ndim == 2: return x[None, ...]
    x2 = np.squeeze(x)
    if x2.ndim == 3: return x2
    return x.reshape(1, T, -1)

def auto_pick_from_dir(video_dir: str, idx: int = 0, pat: str = "*.mp4") -> str:
    cands = sorted(glob.glob(os.path.join(video_dir, pat))) \
          + sorted(glob.glob(os.path.join(video_dir, pat.upper())))
    return cands[idx] if len(cands) > idx else None

# ===================== 특징 빌더 (03과 정렬) =====================
# 16D: [x,y, vx,vy, v, ax,ay, theta, front, d_ws, d_we, d_wl, dtheta, phi_shoulder(0), theta_rel, ang_elbow]
class FeatureBuilder:
    def __init__(self, T: int, D: int):
        self.T, self.D = T, D
        self.hist = {k: deque(maxlen=T) for k in
                     ['t',
                      'x16','y16','x14','y14','x12','y12','x15','y15',
                      'v_w','v_e','vis16','vis14']}
        self.t_accum = 0.0

    @staticmethod
    def _grad(a, t):
        n = len(a)
        if n < 2: return np.zeros(n, dtype=np.float32)
        a = np.asarray(a, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        out = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i == 0:
                dt = (t[i+1]-t[i]) + EPS; out[i] = (a[i+1]-a[i]) / dt
            elif i == n-1:
                dt = (t[i]-t[i-1]) + EPS; out[i] = (a[i]-a[i-1]) / dt
            else:
                dt = (t[i+1]-t[i-1]) + EPS; out[i] = (a[i+1]-a[i-1]) / dt
        return out.astype(np.float32)

    def push_and_get(self, lm, dt: float):
        # 원 좌표 & 가시도
        def P(i): return (lm[i].x, lm[i].y)
        def V(i): return getattr(lm[i], 'visibility', 1.0)
        r_wrist = P(16); r_elbow = P(14); r_sh = P(12); l_wrist = P(15)
        vis16 = float(V(16)); vis14 = float(V(14))

        # 어깨 중심 정규화 + 상완 길이 스케일
        cx, cy = r_sh
        scale = max(math.hypot(r_sh[0]-r_elbow[0], r_sh[1]-r_elbow[1]), 1e-6)
        def norm(p): return ((p[0]-cx)/scale, (p[1]-cy)/scale)
        nw = norm(r_wrist); ne = norm(r_elbow); ns = norm(r_sh); nlw = norm(l_wrist)

        # 누Accum 시간/버퍼
        self.t_accum += dt
        H = self.hist
        H['t'].append(self.t_accum)
        H['x16'].append(nw[0]); H['y16'].append(nw[1])
        H['x14'].append(ne[0]); H['y14'].append(ne[1])
        H['x12'].append(ns[0]); H['y12'].append(ns[1])
        H['x15'].append(nlw[0]); H['y15'].append(nlw[1])
        H['vis16'].append(vis16); H['vis14'].append(vis14)

        # 즉시 특징 생성 조건(우측정렬: 최신 프레임이 윈도우의 끝)
        if len(H['t']) < self.T: return None, None

        # 배열화
        t  = np.asarray(H['t'], dtype=np.float32)
        xw = np.asarray(H['x16'], dtype=np.float32); yw = np.asarray(H['y16'], dtype=np.float32)
        xe = np.asarray(H['x14'], dtype=np.float32); ye = np.asarray(H['y14'], dtype=np.float32)
        xs = np.asarray(H['x12'], dtype=np.float32); ys = np.asarray(H['y12'], dtype=np.float32)
        xl = np.asarray(H['x15'], dtype=np.float32); yl = np.asarray(H['y15'], dtype=np.float32)

        # 속도/가속(손목)
        vx = self._grad(xw, t); vy = self._grad(yw, t)
        v  = np.hypot(vx, vy).astype(np.float32)
        ax = self._grad(vx, t); ay = self._grad(vy, t)

        # 팔꿈치 속도(피크 대체용)
        vxe = self._grad(xe, t); vye = self._grad(ye, t)
        v_e = np.hypot(vxe, vye).astype(np.float32)

        # 각도/거리/상대각
        dx = xw - xs; dy = yw - ys
        theta = np.arctan2(dy, dx).astype(np.float32)
        dtheta = self._grad(theta, t)

        # front: 어깨 기준 → 손목 x가 0 이상이면 전방(오른쪽)
        front = (xw >= 0).astype(np.float32)

        # 거리들
        d_ws = np.hypot(xw - xs, yw - ys).astype(np.float32)   # wrist-shoulder
        d_we = np.hypot(xw - xe, yw - ye).astype(np.float32)   # wrist-elbow
        d_wl = np.hypot(xw - xl, yw - yl).astype(np.float32)   # wrist-left_wrist

        # phi_shoulder 사용하지 않음 → 0, 상대각 = theta
        phi_shoulder = np.zeros_like(theta, dtype=np.float32)
        theta_rel = theta.copy()

        # 팔꿈치 내각(12-14-16)
        v_se_x, v_se_y = (xe - xs), (ye - ys)
        v_we_x, v_we_y = (xw - xe), (yw - ye)
        dot = (v_se_x * v_we_x + v_se_y * v_we_y)
        n1 = np.hypot(v_se_x, v_se_y) + 1e-6
        n2 = np.hypot(v_we_x, v_we_y) + 1e-6
        cosang = np.clip(dot/(n1*n2), -1.0, 1.0)
        ang_elbow = np.arccos(cosang).astype(np.float32)

        feats = np.stack([
            xw, yw, vx, vy, v, ax, ay, theta, front, d_ws, d_we, d_wl,
            dtheta, phi_shoulder, theta_rel, ang_elbow
        ], axis=1).astype(np.float32)

        # 속도/가시도 기록(피크 검사용)
        H['v_w'].append(float(v[-1]))
        H['v_e'].append(float(v_e[-1]))

        diag = {
            'v_w_hist': list(H['v_w']),
            'v_e_hist': list(H['v_e']),
            'vis16_hist': list(H['vis16']),
            'vis14_hist': list(H['vis14']),
        }

        # D 정합
        if feats.shape[1] < self.D:
            pad = np.zeros((self.T, self.D - feats.shape[1]), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=1)
        elif feats.shape[1] > self.D:
            feats = feats[:, :self.D]

        return feats, diag

# ===================== 피크 트리거 (최신 프레임 중심) =====================
def is_recent_peak(seq, v_min, prom_min):
    """최근 5프레임에서 '마지막 프레임'이 국소 최대 & 속도/돌출도 충족"""
    if len(seq) < PEAK_WIN: return False
    w = seq[-PEAK_WIN:]                  # [ -5, -4, -3, -2, -1 ]
    c = w[-1]                            # 최신 프레임
    if not (c > w[-2] and c > w[-3] and c > w[-4] and c > w[-5]):
        return False
    if c < v_min: return False
    prom = c - max(w[-2], w[-3])         # 직전들 대비 돌출
    if prom < prom_min: return False
    return True

def decide_peak(diag):
    vw = diag['v_w_hist']; ve = diag['v_e_hist']
    visw = diag['vis16_hist']
    # 손목 가시도 비율이 낮으면 팔꿈치로 대체
    use_elbow = False
    if len(visw) >= PEAK_WIN:
        win_w = visw[-PEAK_WIN:]
        if sum(1 for v in win_w if v >= VIS_THR) / len(win_w) < USE_ELBOW_RATIO:
            use_elbow = True
    if not use_elbow:
        if is_recent_peak(vw, V_MIN_WRIST, PROM_MIN): return True
        if is_recent_peak(ve, V_MIN_ELBOW, PROM_MIN): return True
        return False
    else:
        return is_recent_peak(ve, V_MIN_ELBOW, PROM_MIN)

# ===================== 시각화 =====================
def draw_body_ui(frame, lm, w, h, classes, prob=None, last_detected="None", last_conf=0.0):
    key_map = {16:"R-Wrist",14:"R-Elbow",12:"R-Shoulder",15:"L-Wrist"}
    trail_joints = (16, 14, 12, 15)
    joint_colors = {16:(0,200,255),14:(255,180,0),12:(0,140,255),15:(180,0,200)}
    default_color = (200,200,200)

    for idx, name in key_map.items():
        x, y = int(lm[idx].x*w), int(lm[idx].y*h)
        cv2.circle(frame, (x,y), 6, (0,255,255), -1)
        cv2.putText(frame, f"{name}({idx})", (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    if not hasattr(draw_body_ui, "_trails"): draw_body_ui._trails = {}
    trails = draw_body_ui._trails; trail_len = 25
    for j in trail_joints:
        if j not in trails or trails[j].maxlen != trail_len:
            trails[j] = deque(maxlen=trail_len)
    for j in list(trails.keys()):
        if j not in trail_joints: del trails[j]
    for j in trail_joints:
        xj, yj = int(lm[j].x*w), int(lm[j].y*h)
        trails[j].append((xj, yj))
    for j, q in trails.items():
        color = joint_colors.get(j, default_color)
        for i in range(1, len(q)):
            cv2.line(frame, q[i-1], q[i], color, 2)

    if prob is not None:
        base_x, base_y = 16, 140
        bar_w, bar_h = 220, 18
        gap = 8
        for i, cls in enumerate(classes):
            p = float(prob[0, i]); x1, y1 = base_x, base_y + i*(bar_h+gap)
            x2 = x1 + int(bar_w * max(0.0, min(1.0, p)))
            cv2.rectangle(frame, (x1, y1), (x1+bar_w, y1+bar_h), (50,50,50), 1)
            cv2.rectangle(frame, (x1, y1), (x2, y1+bar_h), (60,180,75), -1)
            cv2.putText(frame, f"{cls}: {p:.2f}", (x1+6, y1+bar_h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

# ===================== 파일 탐색기(옵션) =====================
def pick_file_dialog(initial_dir: str = None) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        print(f"[WARN] tkinter 불가: {e}", file=sys.stderr); return None
    root = tk.Tk(); root.withdraw(); root.update_idletasks()
    fp = filedialog.askopenfilename(
        title="테스트할 영상(.mp4) 선택",
        initialdir=initial_dir if initial_dir and os.path.isdir(initial_dir) else os.getcwd(),
        filetypes=[("MP4 files","*.mp4;*.MP4"), ("All files","*.*")]
    )
    root.destroy(); return fp if fp else None

# ===================== 메인 =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, default="tcn.onnx")
    ap.add_argument("--meta", type=str, default="tcn_meta.json")
    ap.add_argument("--webcam", action="store_true", help="웹캠(0) 입력 사용")
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--video_dir", type=str, default="testvideo")
    ap.add_argument("--video_index", type=int, default=0)
    ap.add_argument("--th", type=float, default=DEF_TH)
    ap.add_argument("--cooldown", type=float, default=DEF_CD)
    ap.add_argument("--ip", type=str, default=UDP_IP)
    ap.add_argument("--port", type=int, default=UDP_PORT)
    ap.add_argument("--show_landmarks", action="store_true", default=False)
    ap.add_argument("--no-filepicker", action="store_true", default=False)
    args = ap.parse_args()

    # 메타
    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    classes = meta["classes"]            # 예: ['Clear','Drive','Drop','Under','Hairpin','Idle']
    D       = int(meta["feat_dim"])
    T       = int(meta.get("target_T", DEF_T))  # 일반적으로 16
    in_name = meta.get("input_name", "clips")
    out_name= meta.get("output_name", "logits")
    mu  = np.asarray(meta["zscore_mu"],  dtype=np.float32)
    std = np.asarray(meta["zscore_std"], dtype=np.float32)
    std = np.where(np.abs(std) < 1e-8, 1.0, std)  # 0 나눗셈 방지

    # 입력 소스
    video_path = None
    if args.webcam:
        cap = cv2.VideoCapture(0)
    else:
        if args.video and os.path.exists(args.video):
            video_path = args.video
        elif not args.no_filepicker:
            video_path = pick_file_dialog(initial_dir=args.video_dir) or auto_pick_from_dir(args.video_dir, args.video_index)
        else:
            video_path = auto_pick_from_dir(args.video_dir, args.video_index)
        if not video_path or not os.path.exists(video_path):
            print("[ERROR] 사용할 영상이 없습니다.", file=sys.stderr); sys.exit(1)
        cap = cv2.VideoCapture(video_path)
        print(f"[INFO] Using video: {video_path}")
    if not cap or not cap.isOpened():
        print("[ERROR] 입력 소스를 열 수 없습니다.", file=sys.stderr); sys.exit(1)

    # ONNX & UDP
    try:
        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"[ERROR] ONNX 로드 실패: {e}", file=sys.stderr); sys.exit(1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Mediapipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=0, smooth_landmarks=True,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    fb = FeatureBuilder(T=T, D=D)
    last_fire_ts = 0.0
    last_detected, last_conf, last_prob = "None", 0.0, None
    last_wrist_speed = 0.0  # 현재 손목 속도(정규화/초)

    # 시간/표시
    t_prev = time.perf_counter(); dt_ema = TARGET_DT
    printed_io_spec = False
    fps_vis = 0.0; fps_prev_wall = time.perf_counter()

    # Ready/Idle 클래스 인덱스(둘 다 지원)
    ready_idx = classes.index('Ready') if 'Ready' in classes else None
    idle_idx  = classes.index('Idle')  if 'Idle'  in classes else None
    ready_like_idx = ready_idx if ready_idx is not None else idle_idx  # 하나라도 있으면 사용
    ready_like_name= 'Ready' if ready_idx is not None else ('Idle' if idle_idx is not None else None)

    # 스윙 클래스(Ready/Idle 제외)
    swing_ids = [i for i,c in enumerate(classes) if c not in ('Ready','Idle')]

    while True:
        ok, frame = cap.read()
        if not ok: break

        # dt(실시간)
        t_now = time.perf_counter()
        dt_raw = t_now - t_prev; t_prev = t_now
        dt_ema = EMA_ALPHA*dt_raw + (1-EMA_ALPHA)*dt_ema
        dt_ema = max(DT_MIN, min(DT_MAX, dt_ema))
        dt_for_model = TARGET_DT  # 학습 스케일 고정

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            feats, diag = fb.push_and_get(lm, dt=dt_for_model)

            # 최신 손목 속도 업데이트(가능할 때만)
            if diag and diag.get('v_w_hist'):
                last_wrist_speed = float(diag['v_w_hist'][-1])

            # UI
            draw_body_ui(frame, lm, w, h, classes, prob=last_prob,
                         last_detected=last_detected, last_conf=last_conf)
            if args.show_landmarks:
                mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # ===== 항상 추론(하이브리드 결정) =====
            if feats is not None and diag is not None:
                X = feats.astype(np.float32)            # (T,D)
                Xn = (X - mu) / (std + EPS)
                Xn = to_3d_btd(Xn, T, D)               # (1,T,D)

                if not printed_io_spec:
                    ins = sess.get_inputs()[0]
                    print("[INFO] ONNX input spec:", ins.name, ins.shape, ins.type)
                    print("[INFO] Feeding shape:", Xn.shape, Xn.dtype)
                    printed_io_spec = True

                logits = sess.run([out_name], {in_name: Xn})[0]
                p = softmax(logits)
                last_prob = p
                cls_idx = int(np.argmax(p, axis=1)[0])
                conf = float(p[0, cls_idx])
                cls_name = classes[cls_idx] if 0 <= cls_idx < len(classes) else str(cls_idx)

                # 피크 판정(최신 프레임 기준)
                peak_now = decide_peak(diag)

                if peak_now:
                    # 스윙 시점: Ready/Idle 제외, 스윙 클래스만 확정/UDP
                    if cls_idx in swing_ids:
                        now = time.perf_counter()
                        if conf >= args.th and (now - last_fire_ts) >= args.cooldown:
                            pkt = {"swing": True, "class": cls_name, "conf": round(conf,4), "ts": round(now,3)}
                            try:
                                sock.sendto(json.dumps(pkt).encode("utf-8"), (args.ip, args.port))
                            except Exception as e:
                                print(f"[WARN] UDP send failed: {e}", file=sys.stderr)
                            last_fire_ts = now
                            last_detected, last_conf = cls_name, conf
                else:
                    # 비피크 시점: Ready/Idle만 확정 후보
                    if ready_like_idx is not None and cls_idx == ready_like_idx and conf >= TH_READY:
                        last_detected, last_conf = ready_like_name, conf
                        if SEND_READY_UDP:
                            now = time.perf_counter()
                            pkt = {"swing": False, "class": ready_like_name, "conf": round(conf,4), "ts": round(now,3)}
                            try:
                                sock.sendto(json.dumps(pkt).encode("utf-8"), (args.ip, args.port))
                            except Exception as e:
                                print(f"[WARN] UDP send failed: {e}", file=sys.stderr)

        # 오버레이 (상태)
        cv2.putText(frame, f"Last: {last_detected} ({last_conf:.2f})",
                    (16,48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,220,255), 2)

        # FPS
        now_wall = time.perf_counter()
        fps_dt = now_wall - fps_prev_wall
        if fps_dt > 0: fps_vis = 1.0 / fps_dt
        fps_prev_wall = now_wall
        cv2.putText(frame, f"FPS: {fps_vis:.1f}", (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # 현재 손목 속도(정규화/초) 표시 — FPS 아래 줄
        cv2.putText(frame, f"Wrist speed: {last_wrist_speed:.3f} (norm/s)",
                    (16, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

        title = f"Pose Inference (Hybrid: Swing/{'Ready' if 'Ready' in classes else 'Idle'}) - {'Webcam' if args.webcam else os.path.basename(video_path) if 'video_path' in locals() and video_path else 'Video'}"
        cv2.imshow(title, frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
