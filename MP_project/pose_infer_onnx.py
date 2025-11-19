# scripts/pose_infer_onnx.py
# (CLI Camera Picker + Webcam + Hybrid Decision + Jump Detector, Optimized)
# - 실행하면 먼저 CMD에 "카메라 목록"이 뜹니다.
#   · 번호 입력: 카메라 선택
#   · r: 다시 스캔
#   · q/엔터: 종료
# - 선택 후 실시간 추론(스윙 + 점프 감지)
# - UDP로 스윙/점프 이벤트 전송
#
# 최적화 사항
# - 캡처 해상도 기본 960x540 (옵션으로 변경 가능)
# - Mediapipe Pose 입력 프레임 다운샘플링 (기본 0.5배)
# - ONNX 추론을 프레임 건너뛰며 수행 (기본 2프레임마다 1회)
# - 궤적 trail 길이 축소 (25 → 15)
# - 확률 바를 표시/비표시 옵션 (--no_probbar)
# - CUDA 사용 옵션 (--use_cuda, 실패 시 자동 CPU fallback)

import argparse, time, json, socket, math, sys
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import os


# ===================== PyInstaller 안전 경로 =====================
def resource_path(relative_path: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# ===================== 설정 상수 =====================
DEF_T    = 33
DEF_TH   = 0.80
DEF_CD   = 0.0
UDP_IP   = "127.0.0.1"
UDP_PORT = 5052
EPS      = 1e-6

TH_READY        = 0.60
SEND_READY_UDP  = False  # 현재는 사용 안 함

TARGET_FPS = 30.0
TARGET_DT  = 1.0 / TARGET_FPS
EMA_ALPHA  = 0.20
DT_MIN, DT_MAX = 1/90.0, 1/20.0

PEAK_WIN        = 5
V_MIN_WRIST     = 0.80
V_MIN_ELBOW     = 0.50
PROM_MIN        = 0.05
VIS_THR         = 0.60
USE_ELBOW_RATIO = 0.5

# 최적화 관련 기본값
CAP_WIDTH_DEFAULT   = 1280
CAP_HEIGHT_DEFAULT  = 720
POSE_DOWNSCALE_DEF  = 0.5   # Pose 입력 이미지 다운샘플 비율
INFER_STRIDE_DEF    = 2     # ONNX 추론을 N프레임마다 수행
TRAIL_MAXLEN        = 15    # 관절 궤적 trail 길이


# ===================== util =====================
def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= (p.sum(axis=1, keepdims=True) + EPS)
    return p

def to_3d_btd(x, T, D):
    if x.ndim == 3:
        return x
    if x.ndim == 2:
        return x[None, ...]
    x2 = np.squeeze(x)
    if x2.ndim == 3:
        return x2
    return x.reshape(1, T, D)


# ===================== 카메라 스캔 =====================
BACKENDS = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]

def try_open_once(dev_id):
    for be in BACKENDS:
        cap = cv2.VideoCapture(dev_id, be)
        if not cap.isOpened():
            try:
                cap.release()
            except:
                pass
            continue

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            continue

        try:
            name = cap.getBackendName()
        except:
            name = "UNKNOWN"

        return cap, name
    return None, None

def scan_cameras(max_dev=10):
    found = []
    for dev_id in range(max_dev):
        cap, be = try_open_once(dev_id)
        if cap is not None:
            ok, f = cap.read()
            if ok and f is not None:
                h, w = f.shape[:2]
                found.append((dev_id, be, (w, h)))
            cap.release()
    return found

def choose_camera_cli(max_dev=10):
    while True:
        found = scan_cameras(max_dev)
        print("========== Camera Picker ==========")
        if not found:
            print("사용 가능한 카메라 없음")
            print("'r' 재스캔")
        else:
            for idx, (dev, be, wh) in enumerate(found):
                print(f" [{idx}] Device {dev} ({be}, {wh[0]}x{wh[1]})")

        print("-----------------------------------")
        sel = input("번호 입력 (r=다시, q=종료): ").strip()

        if sel == "" or sel.lower() == "q":
            return None
        if sel.lower() == "r":
            continue

        try:
            idx = int(sel)
        except:
            print("잘못된 입력")
            continue

        if 0 <= idx < len(found):
            return found[idx][0]

        print("범위 초과")


def open_camera(dev_id, width=CAP_WIDTH_DEFAULT, height=CAP_HEIGHT_DEFAULT):
    for be in BACKENDS:
        cap = cv2.VideoCapture(dev_id, be)
        if not cap.isOpened():
            try:
                cap.release()
            except:
                pass
            continue

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        t0 = time.time()
        ok = False
        while time.time() - t0 < 2.0:
            ok, f = cap.read()
            if ok and f is not None:
                break
            time.sleep(0.03)

        if ok:
            return cap
        cap.release()

    raise RuntimeError(f"Camera open failed {dev_id}")


# ===================== FeatureBuilder =====================
class FeatureBuilder:
    def __init__(self, T, D):
        self.T, self.D = T, D
        self.hist = {
            k: deque(maxlen=T) for k in [
                't',
                'x16','y16','x14','y14','x12','y12','x15','y15',
                'v_w','v_e','vis16','vis14'
            ]
        }
        self.t_accum = 0.0

    @staticmethod
    def _grad(a, t):
        n = len(a)
        if n < 2:
            return np.zeros(n, np.float32)
        a = np.asarray(a, float)
        t = np.asarray(t, float)
        out = np.zeros(n, float)
        for i in range(n):
            if i == 0:
                dt = t[i+1] - t[i]
                out[i] = (a[i+1] - a[i]) / (dt + EPS)
            elif i == n-1:
                dt = t[i] - t[i-1]
                out[i] = (a[i] - a[i-1]) / (dt + EPS)
            else:
                dt = t[i+1] - t[i-1]
                out[i] = (a[i+1] - a[i-1]) / (dt + EPS)
        return out.astype(np.float32)

    def push_and_get(self, lm, dt):
        def P(i): return (lm[i].x, lm[i].y)
        def V(i): return getattr(lm[i], 'visibility', 1.0)

        r_w = P(16)
        r_e = P(14)
        r_s = P(12)
        l_w = P(15)
        vis16 = V(16)
        vis14 = V(14)

        cx, cy = r_s
        scale = max(math.hypot(r_s[0] - r_e[0], r_s[1] - r_e[1]), 1e-6)

        def norm(p): return ((p[0] - cx) / scale, (p[1] - cy) / scale)

        nw = norm(r_w)
        ne = norm(r_e)
        ns = norm(r_s)
        nlw = norm(l_w)

        H = self.hist
        self.t_accum += dt
        H['t'].append(self.t_accum)
        H['x16'].append(nw[0]); H['y16'].append(nw[1])
        H['x14'].append(ne[0]); H['y14'].append(ne[1])
        H['x12'].append(ns[0]); H['y12'].append(ns[1])
        H['x15'].append(nlw[0]); H['y15'].append(nlw[1])
        H['vis16'].append(vis16); H['vis14'].append(vis14)

        if len(H['t']) < self.T:
            return None, None

        # 배열화
        t  = np.asarray(H['t'],  np.float32)
        xw = np.asarray(H['x16'])
        yw = np.asarray(H['y16'])
        xe = np.asarray(H['x14'])
        ye = np.asarray(H['y14'])
        xs = np.asarray(H['x12'])
        ys = np.asarray(H['y12'])
        xl = np.asarray(H['x15'])
        yl = np.asarray(H['y15'])

        vx = self._grad(xw, t)
        vy = self._grad(yw, t)
        v  = np.hypot(vx, vy).astype(np.float32)
        ax = self._grad(vx, t)
        ay = self._grad(vy, t)

        vxe = self._grad(xe, t)
        vye = self._grad(ye, t)
        v_e = np.hypot(vxe, vye).astype(np.float32)

        dx = xw - xs
        dy = yw - ys
        theta = np.arctan2(dy, dx).astype(np.float32)
        dtheta = self._grad(theta, t)

        front = (xw >= 0).astype(np.float32)
        d_ws = np.hypot(xw - xs, yw - ys).astype(np.float32)
        d_we = np.hypot(xw - xe, yw - ye).astype(np.float32)
        d_wl = np.hypot(xw - xl, yw - yl).astype(np.float32)

        phi_shoulder = np.zeros_like(theta, np.float32)
        theta_rel    = theta.copy()

        v_se_x, v_se_y = (xe - xs), (ye - ys)
        v_we_x, v_we_y = (xw - xe), (yw - ye)
        dot = (v_se_x * v_we_x + v_se_y * v_we_y)
        n1 = np.hypot(v_se_x, v_se_y) + 1e-6
        n2 = np.hypot(v_we_x, v_we_y) + 1e-6
        cosang = np.clip(dot / (n1 * n2), -1, 1)
        ang_elbow = np.arccos(cosang).astype(np.float32)

        feats = np.stack([
            xw, yw, vx, vy, v, ax, ay, theta,
            front, d_ws, d_we, d_wl,
            dtheta, phi_shoulder, theta_rel, ang_elbow
        ], axis=1).astype(np.float32)

        H['v_w'].append(float(v[-1]))
        H['v_e'].append(float(v_e[-1]))

        diag = {
            'v_w_hist':  list(H['v_w']),
            'v_e_hist':  list(H['v_e']),
            'vis16_hist': list(H['vis16']),
            'vis14_hist': list(H['vis14']),
        }

        if feats.shape[1] < self.D:
            pad = np.zeros((self.T, self.D - feats.shape[1]), np.float32)
            feats = np.concatenate([feats, pad], axis=1)
        elif feats.shape[1] > self.D:
            feats = feats[:, :self.D]

        return feats, diag


# ===================== Peak 기반 결정 =====================
def is_recent_peak(seq, v_min, prom_min):
    if len(seq) < PEAK_WIN:
        return False
    w = seq[-PEAK_WIN:]
    c = w[-1]
    if not (c > w[-2] and c > w[-3] and c > w[-4] and c > w[-5]):
        return False
    if c < v_min:
        return False
    prom = c - max(w[-2], w[-3])
    return prom >= prom_min

def decide_peak(diag):
    vw   = diag['v_w_hist']
    ve   = diag['v_e_hist']
    visw = diag['vis16_hist']

    use_elb = False
    if len(visw) >= PEAK_WIN:
        win = visw[-PEAK_WIN:]
        if sum(1 for v in win if v >= VIS_THR) / len(win) < USE_ELBOW_RATIO:
            use_elb = True

    if not use_elb:
        if is_recent_peak(vw, V_MIN_WRIST, PROM_MIN):
            return True
        if is_recent_peak(ve, V_MIN_ELBOW, PROM_MIN):
            return True
        return False
    else:
        return is_recent_peak(ve, V_MIN_ELBOW, PROM_MIN)


# ===================== UI =====================
def draw_body_ui(frame, lm, w, h, classes, prob, last_detected, last_conf):
    key_map = {16: "R-Wrist", 14: "R-Elbow", 12: "R-Shoulder", 15: "L-Wrist"}
    for idx, name in key_map.items():
        x = int(lm[idx].x * w)
        y = int(lm[idx].y * h)
        cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
        cv2.putText(frame, f"{name}({idx})", (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 궤적 trail
    if not hasattr(draw_body_ui, "_trails"):
        draw_body_ui._trails = {}
    trails = draw_body_ui._trails
    trail_j = [16, 14, 12, 15]

    for j in trail_j:
        trails.setdefault(j, deque(maxlen=TRAIL_MAXLEN))

    for j in trail_j:
        xj = int(lm[j].x * w)
        yj = int(lm[j].y * h)
        trails[j].append((xj, yj))

    colors = {16: (0, 200, 255), 14: (255, 180, 0), 12: (0, 140, 255), 15: (180, 0, 200)}
    for j, q in trails.items():
        col = colors.get(j, (200, 200, 200))
        for i in range(1, len(q)):
            cv2.line(frame, q[i-1], q[i], col, 2)

    # 클래스 확률 바 (옵션)
    if prob is not None:
        base_x, base_y = 16, 140
        bar_w, bar_h = 220, 18
        gap = 8
        for i, cls in enumerate(classes):
            p = float(prob[0, i])
            x1 = base_x
            y1 = base_y + i * (bar_h + gap)
            x2 = x1 + int(bar_w * max(0, min(1, p)))
            cv2.rectangle(frame, (x1, y1), (x1 + bar_w, y1 + bar_h), (50, 50, 50), 1)
            cv2.rectangle(frame, (x1, y1), (x2, y1 + bar_h), (60, 180, 75), -1)
            cv2.putText(frame, f"{cls}: {p:.2f}",
                        (x1 + 6, y1 + bar_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


# ===================== 메인 =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, default="tcn.onnx")
    ap.add_argument("--meta", type=str, default="tcn_meta.json")
    ap.add_argument("--device", type=int, default=-1)
    ap.add_argument("--ip", type=str, default=UDP_IP)
    ap.add_argument("--port", type=int, default=UDP_PORT)
    ap.add_argument("--th", type=float, default=DEF_TH)
    ap.add_argument("--cooldown", type=float, default=DEF_CD)
    ap.add_argument("--show_landmarks", action="store_true")

    # 점프 관련
    ap.add_argument("--jump_thr", type=float, default=2.0)
    ap.add_argument("--jump_hold", type=float, default=0.5)
    ap.add_argument("--jump_send_cooldown", type=float, default=0.3)

    # 기타 옵션
    ap.add_argument("--no_picker", action="store_true")

    # 최적화 관련 옵션
    ap.add_argument("--cap_width", type=int, default=CAP_WIDTH_DEFAULT,
                    help="카메라 캡처 폭(px)")
    ap.add_argument("--cap_height", type=int, default=CAP_HEIGHT_DEFAULT,
                    help="카메라 캡처 높이(px)")
    ap.add_argument("--pose_downscale", type=float, default=POSE_DOWNSCALE_DEF,
                    help="Mediapipe Pose 입력 다운샘플 비율 (0<r<=1), 예: 0.5")
    ap.add_argument("--infer_stride", type=int, default=INFER_STRIDE_DEF,
                    help="ONNX 추론 프레임 간격 (>=1)")
    ap.add_argument("--no_probbar", action="store_true",
                    help="클래스 확률 바 UI 표시하지 않음")
    ap.add_argument("--use_cuda", action="store_true",
                    help="가능하면 CUDAExecutionProvider 사용")

    args = ap.parse_args()

    # 카메라 선택
    if args.no_picker and args.device >= 0:
        chosen = args.device
    else:
        chosen = choose_camera_cli(10)
        if chosen is None:
            print("종료")
            return

    # meta 로드
    meta_path = resource_path(args.meta)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    classes = meta["classes"]
    D = int(meta["feat_dim"])
    T = int(meta.get("target_T", DEF_T))
    in_name = meta.get("input_name", "clips")
    out_name = meta.get("output_name", "logits")

    mu = np.asarray(meta["zscore_mu"], np.float32)
    std = np.asarray(meta["zscore_std"], np.float32)
    std = np.where(np.abs(std) < 1e-8, 1.0, std)

    # ONNX 세션 생성 (CUDA 옵션)
    onnx_path = resource_path(args.onnx)
    providers = ["CPUExecutionProvider"]

    if args.use_cuda:
        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess = ort.InferenceSession(onnx_path, providers=providers)
            print("[INFO] CUDAExecutionProvider 사용")
        except Exception as e:
            print(f"[WARN] CUDAExecutionProvider 사용 실패, CPU로 fallback: {e}")
            providers = ["CPUExecutionProvider"]
            sess = ort.InferenceSession(onnx_path, providers=providers)
    else:
        sess = ort.InferenceSession(onnx_path, providers=providers)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Mediapipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=0,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    # 카메라 열기 (해상도 옵션 반영)
    cap = open_camera(chosen, width=args.cap_width, height=args.cap_height)

    fb = FeatureBuilder(T=T, D=D)

    last_fire_ts = 0.0
    last_detected = "None"
    last_conf = 0.0
    last_prob = None
    last_wrist_speed = 0.0

    # ★ 최근 스윙 5개 저장용 큐
    swing_queue = deque(maxlen=5)

    t_prev = time.perf_counter()
    dt_ema = TARGET_DT

    printed = False

    fps_prev = time.perf_counter()
    fps_vis = 0.0

    frame_idx = 0
    infer_stride = max(1, int(args.infer_stride))

    try:
        ready_idx = classes.index("Ready")
    except:
        ready_idx = None

    prev_nose_y = None
    last_jump_ts = -1e9
    last_jump_send_ts = -1e9

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1

        t_now = time.perf_counter()
        dt_raw = t_now - t_prev
        t_prev = t_now

        dt_ema = EMA_ALPHA * dt_raw + (1 - EMA_ALPHA) * dt_ema
        dt_ema = max(DT_MIN, min(DT_MAX, dt_ema))

        dt_for_model = TARGET_DT  # 모델 시계는 고정 30fps로 가정

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose 입력 다운샘플링
        scale = max(0.1, float(args.pose_downscale))
        if scale < 1.0:
            small = cv2.resize(rgb, None, fx=scale, fy=scale)
            res = pose.process(small)
        else:
            res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            # 점프 감지
            nose_y = float(lm[0].y)
            if prev_nose_y is not None:
                dy = nose_y - prev_nose_y
                if dt_ema > 0:
                    dydt = dy / dt_ema
                    if (-dydt) >= args.jump_thr:
                        last_jump_ts = t_now
                        if (t_now - last_jump_send_ts) >= args.jump_send_cooldown:
                            pkt = {"jump": True, "speed": round(-dydt, 4), "ts": round(t_now, 3)}
                            try:
                                sock.sendto(json.dumps(pkt).encode(), (args.ip, args.port))
                            except:
                                pass
                            last_jump_send_ts = t_now
            prev_nose_y = nose_y

            feats, diag = fb.push_and_get(lm, dt=dt_for_model)

            if diag and diag.get("v_w_hist"):
                last_wrist_speed = float(diag["v_w_hist"][-1])

            # UI 그리기 (확률 바는 옵션)
            prob_for_ui = None if args.no_probbar else last_prob
            draw_body_ui(frame, lm, w, h, classes,
                         prob=prob_for_ui,
                         last_detected=last_detected,
                         last_conf=last_conf)

            if args.show_landmarks:
                mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # ONNX 추론 + 스윙 판정 (stride 적용)
            if feats is not None and diag is not None:
                if (frame_idx % infer_stride) == 0:
                    X = feats.astype(np.float32)
                    Xn = (X - mu) / (std + EPS)
                    Xn = to_3d_btd(Xn, T, D)

                    if not printed:
                        ins = sess.get_inputs()[0]
                        print("[INFO] ONNX input:", ins.name, ins.shape, ins.type)
                        printed = True

                    logits = sess.run([out_name], {in_name: Xn})[0]
                    p = softmax(logits)
                    last_prob = p

                    cls_idx = int(np.argmax(p, axis=1)[0])
                    conf = float(p[0, cls_idx])
                    cls_name = classes[cls_idx] if 0 <= cls_idx < len(classes) else str(cls_idx)

                    # hybrid peak
                    peak_now = decide_peak(diag)

                    if peak_now:
                        if cls_name != "Idle":
                            if conf >= args.th and (t_now - last_fire_ts) >= args.cooldown:
                                pkt = {
                                    "swing": True,
                                    "class": cls_name,
                                    "conf": round(conf, 4),
                                    "ts": round(t_now, 3)
                                }
                                try:
                                    sock.sendto(json.dumps(pkt).encode(), (args.ip, args.port))
                                except:
                                    pass

                                last_fire_ts = t_now
                                last_detected = cls_name
                                last_conf = conf

                                # ★ 스윙 확정 시 큐에 추가
                                swing_queue.append((cls_name, conf, round(t_now, 2)))
                    else:
                        if (ready_idx is not None and
                            cls_idx == ready_idx and
                            conf >= TH_READY):
                            last_detected = "Ready"
                            last_conf = conf

        # 텍스트 UI
        cv2.putText(frame,
                    f"Last: {last_detected} ({last_conf:.2f})",
                    (16, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 220, 255), 2)

        now = time.perf_counter()
        fps = 1 / (now - fps_prev) if now != fps_prev else 0.0
        fps_prev = now
        fps_vis = fps

        cv2.putText(frame,
                    f"Wrist speed: {last_wrist_speed:.3f}",
                    (16, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # ★ 최근 스윙 큐 (왼쪽 위, 빨간 글씨)
        base_x, base_y = w-360, 150
        line_h = 24
        for i, (cname, cconf, cts) in enumerate(swing_queue):
            txt = f"[{i+1}] {cname} ({cconf:.2f}) t={cts}"
            cv2.putText(frame, txt,
                        (base_x, base_y + i * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)


        cv2.putText(frame, f"FPS: {fps_vis:.1f}",
                    (16, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame,
                    f"Wrist speed: {last_wrist_speed:.3f}",
                    (16, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # 점프 텍스트
        if (time.perf_counter() - last_jump_ts) <= args.jump_hold:
            text = "JUMP!"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cx = w // 2 - tw // 2
            cy = h - 20
            cv2.putText(frame, text, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Pose Inference (Webcam)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
