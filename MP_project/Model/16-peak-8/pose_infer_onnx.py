# scripts/pose_infer_onnx.py
# (CLI Camera Picker + Webcam + Softmax-based Swing Decision + Jump Detector)
# - 실행하면 먼저 CMD에 "카메라 목록"이 뜹니다.
#   · 번호 입력: 해당 번호 카메라 선택
#   · r: 다시 스캔
#   · q 또는 빈 입력: 종료
# - 선택 후 실시간 추론(스윙 + 점프 감지) 시작
# - UDP로 스윙/점프 이벤트 전송
#
# 변경사항(softmax 기반 스윙 확정 방식):
# - 더 이상 "속도 피크(16|peak|8) 검출"로 스윙을 확정하지 않음
# - 대신 softmax 확률 + 클래스 + 상태 머신으로 스윙 확정:
#   · cls_name != 'Ready'
#   · conf ≥ th_up (기본 0.80)
#   · 직전 프레임은 swing_state == False (스윙 상태 아님)
#   · 쿨다운(cooldown 초) 경과
#   → 위 조건 만족 시 스윙 이벤트 1회 발생 후 swing_state=True
#   · conf ≤ th_down (기본 0.60) 이거나 Ready로 돌아오면 swing_state=False 로 복귀
# - 이렇게 하면 "한 번 스윙 = 한 번 이벤트"에 더 가깝게 유지하면서
#   속도 피크 조건 때문에 생기던 미검출을 없앰.

import argparse, time, json, socket, math, sys
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import os

# ===================== PyInstaller 리소스 경로 유틸 =====================
def resource_path(relative_path: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ===================== 상수/기본값 =====================
PRE      = 16   # (이제 피크 검출에는 쓰지 않지만, T=25 고정용으로 유지)
POST     = 8
DEF_T    = PRE + 1 + POST   # 25
DEF_TH   = 0.80             # swing 진입 상한(th_up) 기본값
DEF_TH_DOWN = 0.60          # swing 종료 하한(th_down) 기본값
DEF_CD   = 0.80
UDP_IP   = "127.0.0.1"
UDP_PORT = 5052
EPS      = 1e-6

# Ready 관련
TH_READY        = 0.60
SEND_READY_UDP  = False

# 실시간 dt 안정화(웹캠 권장)
TARGET_FPS = 30.0
TARGET_DT  = 1.0 / TARGET_FPS
EMA_ALPHA  = 0.20
DT_MIN, DT_MAX = 1/90.0, 1/20.0

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

# ===================== 카메라 스캔/CLI 선택 =====================
BACKENDS = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]

def try_open_once(dev_id: int):
    for be in BACKENDS:
        cap = cv2.VideoCapture(dev_id, be)
        if not cap.isOpened():
            try: cap.release()
            except: pass
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release(); continue
        try: name = cap.getBackendName()
        except Exception: name = "UNKNOWN"
        return cap, name
    return None, None

def scan_cameras(max_dev: int = 10):
    found = []
    for dev_id in range(max_dev):
        cap, be_name = try_open_once(dev_id)
        if cap is not None:
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                found.append((dev_id, be_name, (w, h)))
            cap.release()
    return found

def choose_camera_cli(max_dev: int = 10):
    while True:
        found = scan_cameras(max_dev=max_dev)
        print("========== Camera Picker (CLI) ==========")
        if not found:
            print("사용 가능한 카메라가 없습니다.")
            print("장치를 연결한 뒤 'r'을 입력하면 다시 스캔합니다.")
        else:
            print(f"검색된 장치 수: {len(found)}")
            for idx, (dev, be, wh) in enumerate(found):
                print(f"  [{idx}] Device {dev}  ({be}, {wh[0]}x{wh[1]})")
        print("----------------------------------------")
        print("번호 입력 → 해당 카메라 선택")
        print("'r' → 다시 스캔, 'q' 또는 빈 입력 → 종료")
        sel = input("카메라 번호를 입력하세요: ").strip()
        if sel == "" or sel.lower() == "q":
            print("[INFO] 사용자 종료 선택."); return None
        if sel.lower() == "r":
            print("[INFO] 다시 스캔합니다.\n"); continue
        try:
            idx = int(sel)
        except ValueError:
            print(f"[WARN] 잘못된 입력입니다: {sel}\n"); continue
        if not found:
            print("[WARN] 현재 사용 가능한 카메라가 없습니다.\n"); continue
        if 0 <= idx < len(found):
            dev_id = found[idx][0]
            print(f"[INFO] 선택된 카메라: 리스트[{idx}] → Device {dev_id}\n")
            return dev_id
        else:
            print(f"[WARN] 범위를 벗어난 번호입니다: {idx}\n")

def open_camera(dev_id: int):
    for be in BACKENDS:
        cap = cv2.VideoCapture(dev_id, be)
        if not cap.isOpened():
            try: cap.release()
            except: pass
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        t0 = time.time(); ok = False
        while time.time() - t0 < 2.0:
            ok, frame = cap.read()
            if ok and frame is not None: break
            time.sleep(0.03)
        if ok:
            try: name = cap.getBackendName()
            except Exception: name = "UNKNOWN"
            print(f"[INFO] Camera opened: device={dev_id}, backend={name}")
            return cap
        cap.release()
    raise RuntimeError(f"Camera open failed for device {dev_id}")

# ===================== 특징 빌더 (03과 정렬) =====================
# 16D: [x,y, vx,vy, v, ax,ay, theta, front, d_ws, d_we, d_wl, dtheta, phi_shoulder(0), theta_rel, ang_elbow]
class FeatureBuilder:
    def __init__(self, T: int, D: int):
        self.T, self.D = T, D
        self.hist = {k: deque(maxlen=T) for k in [
            't','x16','y16','x14','y14','x12','y12','x15','y15',
            'v_w','v_e','vis16','vis14'
        ]}
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
        def P(i): return (lm[i].x, lm[i].y)
        def V(i): return getattr(lm[i], 'visibility', 1.0)
        r_wrist = P(16); r_elbow = P(14); r_sh = P(12); l_wrist = P(15)
        vis16 = float(V(16)); vis14 = float(V(14))

        cx, cy = r_sh
        scale = max(math.hypot(r_sh[0]-r_elbow[0], r_sh[1]-r_elbow[1]), 1e-6)
        def norm(p): return ((p[0]-cx)/scale, (p[1]-cy)/scale)
        nw = norm(r_wrist); ne = norm(r_elbow); ns = norm(r_sh); nlw = norm(l_wrist)

        self.t_accum += dt
        H = self.hist
        H['t'].append(self.t_accum)
        H['x16'].append(nw[0]); H['y16'].append(nw[1])
        H['x14'].append(ne[0]); H['y14'].append(ne[1])
        H['x12'].append(ns[0]); H['y12'].append(ns[1])
        H['x15'].append(nlw[0]); H['y15'].append(nlw[1])
        H['vis16'].append(vis16); H['vis14'].append(vis14)

        if len(H['t']) < self.T: return None, None

        t  = np.asarray(H['t'], dtype=np.float32)
        xw = np.asarray(H['x16'], dtype=np.float32); yw = np.asarray(H['y16'], dtype=np.float32)
        xe = np.asarray(H['x14'], dtype=np.float32); ye = np.asarray(H['y14'], dtype=np.float32)
        xs = np.asarray(H['x12'], dtype=np.float32); ys = np.asarray(H['y12'], dtype=np.float32)
        xl = np.asarray(H['x15'], dtype=np.float32); yl = np.asarray(H['y15'], dtype=np.float32)

        vx = self._grad(xw, t); vy = self._grad(yw, t)
        v  = np.hypot(vx, vy).astype(np.float32)
        ax = self._grad(vx, t); ay = self._grad(vy, t)

        vxe = self._grad(xe, t); vye = self._grad(ye, t)
        v_e = np.hypot(vxe, vye).astype(np.float32)

        dx = xw - xs; dy = yw - ys
        theta = np.arctan2(dy, dx).astype(np.float32)
        dtheta = self._grad(theta, t)

        front = (xw >= 0).astype(np.float32)
        d_ws = np.hypot(xw - xs, yw - ys).astype(np.float32)
        d_we = np.hypot(xw - xe, yw - ye).astype(np.float32)
        d_wl = np.hypot(xw - xl, yw - yl).astype(np.float32)

        phi_shoulder = np.zeros_like(theta, dtype=np.float32)
        theta_rel = theta.copy()

        v_se_x, v_se_y = (xe - xs), (ye - ys)
        v_we_x, v_we_y = (xw - xe), (yw - ye)
        dot = (v_se_x * v_we_x + v_se_y * v_we_y)
        n1 = np.hypot(v_se_x, v_se_y) + 1e-6
        n2 = np.hypot(v_we_x, v_we_y) + 1e-6
        cosang = np.clip(dot/(n1*n2), -1.0, 1.0)
        ang_elbow = np.arccos(cosang).astype(np.float32)

        feats = np.stack([
            xw, yw, vx, vy, v, ax, ay, theta,
            front, d_ws, d_we, d_wl,
            dtheta, phi_shoulder, theta_rel, ang_elbow
        ], axis=1).astype(np.float32)

        H['v_w'].append(float(v[-1]))
        H['v_e'].append(float(v_e[-1]))

        diag = {
            'v_w_hist':   list(H['v_w']),
            'v_e_hist':   list(H['v_e']),
            'vis16_hist': list(H['vis16']),
            'vis14_hist': list(H['vis14']),
        }

        if feats.shape[1] < self.D:
            pad = np.zeros((self.T, self.D - feats.shape[1]), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=1)
        elif feats.shape[1] > self.D:
            feats = feats[:, :self.D]
        return feats, diag

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

# ===================== 메인 =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, default="tcn.onnx")
    ap.add_argument("--meta", type=str, default="tcn_meta.json")
    ap.add_argument("--device", type=int, default=-1)
    ap.add_argument("--ip", type=str, default=UDP_IP)
    ap.add_argument("--port", type=int, default=UDP_PORT)
    # softmax 임계값 (상한/하한)
    ap.add_argument("--th", type=float, default=DEF_TH, help="스윙 진입 상한 임계값 (softmax ≥ th_up)")
    ap.add_argument("--th_down", type=float, default=DEF_TH_DOWN, help="스윙 종료 하한 임계값 (softmax ≤ th_down)")
    ap.add_argument("--cooldown", type=float, default=DEF_CD, help="스윙 이벤트 최소 간격(초)")
    ap.add_argument("--show_landmarks", action="store_true", default=False)
    # 점프 감지
    ap.add_argument("--jump_thr", type=float, default=2.00, help="점프 임계값( -dy/dt >= jump_thr, 단위 1/s )")
    ap.add_argument("--jump_hold", type=float, default=0.50, help="점프 표시 유지 시간(초)")
    ap.add_argument("--jump_send_cooldown", type=float, default=0.30, help="점프 UDP 연속 전송 쿨다운(초)")
    ap.add_argument("--no_picker", action="store_true", help="CMD 카메라 선택 없이 --device 값으로 바로 열기")
    args = ap.parse_args()

    # 카메라 선택
    if args.no_picker and args.device >= 0:
        chosen_dev = args.device
        print(f"[INFO] --no_picker, device={chosen_dev}")
    else:
        chosen_dev = choose_camera_cli(max_dev=10)
        if chosen_dev is None:
            print("[INFO] 프로그램 종료."); return
    print(f"[INFO] 최종 선택 카메라: Device {chosen_dev}")

    # 메타 로드
    meta_path = resource_path(args.meta)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    classes = meta["classes"]
    D       = int(meta["feat_dim"])
    # 입력 길이는 16|peak|8 학습과 동일하게 25로 유지
    T       = DEF_T
    in_name = meta.get("input_name", "clips")
    out_name= meta.get("output_name", "logits")
    mu  = np.asarray(meta["zscore_mu"],  dtype=np.float32)
    std = np.asarray(meta["zscore_std"], dtype=np.float32)
    std = np.where(np.abs(std) < 1e-8, 1.0, std)

    # ONNX & UDP
    try:
        onnx_path = resource_path(args.onnx)
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"[ERROR] ONNX 로드 실패: {e}", file=sys.stderr); sys.exit(1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Mediapipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=0, smooth_landmarks=True,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    # 카메라 오픈
    cap = open_camera(chosen_dev)

    fb = FeatureBuilder(T=T, D=D)
    last_fire_ts   = 0.0
    last_detected  = "None"
    last_conf      = 0.0
    last_prob      = None
    last_wrist_speed = 0.0

    t_prev = time.perf_counter(); dt_ema = TARGET_DT
    printed_io_spec = False
    fps_vis = 0.0; fps_prev_wall = time.perf_counter()

    try:
        ready_idx = classes.index('Ready')
    except ValueError:
        ready_idx = None

    # 점프 상태
    prev_nose_y       = None
    last_jump_ts      = -1e9
    last_jump_send_ts = -1e9

    # 스윙 상태 머신 (softmax 기반)
    swing_state = False         # 현재 "스윙 중"인지 여부
    swing_class = None          # 현재 스윙 클래스
    th_up   = float(args.th)
    th_down = float(args.th_down)

    # 안전장치: th_down은 th_up보다 작게
    if th_down >= th_up:
        print(f"[WARN] th_down({th_down}) >= th_up({th_up}) 이므로, th_down을 th_up-0.1로 조정합니다.")
        th_down = max(0.0, th_up - 0.1)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok or frame is None: break

        t_now  = time.perf_counter()
        dt_raw = t_now - t_prev; t_prev = t_now
        dt_ema = EMA_ALPHA*dt_raw + (1-EMA_ALPHA)*dt_ema
        dt_ema = max(DT_MIN, min(DT_MAX, dt_ema))
        dt_for_model = TARGET_DT

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            # 점프 감지 (nose y: 감소=위로 이동)
            nose_y = float(lm[0].y)
            if prev_nose_y is not None and dt_ema > 0:
                dy = nose_y - prev_nose_y
                dydt = dy / dt_ema
                if (-dydt) >= args.jump_thr:
                    last_jump_ts = t_now
                    if (t_now - last_jump_send_ts) >= args.jump_send_cooldown:
                        jump_pkt = {"jump": True, "speed": round(-dydt,4), "ts": round(t_now,3)}
                        try: sock.sendto(json.dumps(jump_pkt).encode("utf-8"), (args.ip, args.port))
                        except Exception as e: print(f"[WARN] UDP send failed (jump): {e}", file=sys.stderr)
                        last_jump_send_ts = t_now
            prev_nose_y = nose_y

            feats, diag = fb.push_and_get(lm, dt=dt_for_model)

            if diag and diag.get('v_w_hist'):
                last_wrist_speed = float(diag['v_w_hist'][-1])

            draw_body_ui(frame, lm, w, h, classes, prob=last_prob, last_detected=last_detected, last_conf=last_conf)
            if args.show_landmarks:
                mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if feats is not None and diag is not None:
                X = feats.astype(np.float32)
                Xn = (X - mu) / (std + EPS)
                Xn = to_3d_btd(Xn, T, D)

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

                # ======== softmax 기반 스윙 상태 머신 ========
                if cls_name != 'Ready':
                    # Ready가 아닌 스윙 클래스
                    if not swing_state:
                        # 스윙이 아니었다가 → 스윙 진입을 시도
                        if (conf >= th_up) and ((t_now - last_fire_ts) >= args.cooldown):
                            # 새 스윙 확정
                            pkt = {
                                "swing": True,
                                "class": cls_name,
                                "conf": round(conf,4),
                                "ts": round(t_now,3)
                            }
                            try:
                                sock.sendto(json.dumps(pkt).encode("utf-8"), (args.ip, args.port))
                            except Exception as e:
                                print(f"[WARN] UDP send failed: {e}", file=sys.stderr)
                            last_fire_ts  = t_now
                            last_detected = cls_name
                            last_conf     = conf

                            swing_state = True
                            swing_class = cls_name
                    else:
                        # 이미 스윙 상태인 경우
                        # - 다른 클래스로 바뀌거나
                        # - conf가 충분히 떨어지면 스윙 종료
                        if (cls_name != swing_class) or (conf <= th_down):
                            swing_state = False
                            swing_class = None
                else:
                    # Ready 클래스인 경우
                    if (ready_idx is not None) and (cls_idx == ready_idx) and (conf >= TH_READY):
                        last_detected = 'Ready'
                        last_conf     = conf
                        if SEND_READY_UDP:
                            pkt = {
                                "swing": False,
                                "class": "Ready",
                                "conf": round(conf,4),
                                "ts": round(t_now,3)
                            }
                            try:
                                sock.sendto(json.dumps(pkt).encode("utf-8"), (args.ip, args.port))
                            except Exception as e:
                                print(f"[WARN] UDP send failed: {e}", file=sys.stderr)
                    # Ready가 나오면 스윙 상태는 무조건 초기화
                    swing_state = False
                    swing_class = None

        # 상태 표시
        cv2.putText(frame, f"Last confirmed: {last_detected} ({last_conf:.2f})", (16,48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,220,255), 2)

        now_wall = time.perf_counter()
        fps_dt = now_wall - fps_prev_wall
        if fps_dt > 0: fps_vis = 1.0 / fps_dt
        fps_prev_wall = now_wall
        cv2.putText(frame, f"FPS: {fps_vis:.1f}", (16,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.putText(frame, f"Wrist speed: {last_wrist_speed:.3f} (norm/s)", (16,110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

        # 점프 표시
        if (time.perf_counter() - last_jump_ts) <= args.jump_hold:
            text = "JUMP!"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cx = w//2 - tw//2; cy = h - 20
            cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        cv2.imshow(f"Pose Inference (Webcam, softmax-based, T={DEF_T})", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
