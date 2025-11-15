# scripts/pose_infer_onnx.py
# (CLI Camera Picker + Webcam + Hybrid Decision + Jump Detector)
# - 실행하면 먼저 CMD에 "카메라 목록"이 뜹니다.
#   · 번호 입력: 해당 번호 카메라 선택
#   · r: 다시 스캔
#   · q 또는 빈 입력: 종료
# - 선택 후 실시간 추론(스윙 + 점프 감지) 시작
# - UDP로 스윙/점프 이벤트 전송
#
# 변경사항(기능 유지 + 개선):
# - 피크 임계 '적응형(평균+표준편차)' 도입: ADAPT_R, ADAPT_K
# - 고정 임계 기본값 상향(더 빠른 속도에서만 피크로 인정)
# - CLI 옵션으로 임계값/적응 파라미터 조정 가능
# - 기존 Jump/Ready/카메라 선택/UDP 기능 유지

import argparse, time, json, socket, math, sys
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import os

# ===================== PyInstaller 리소스 경로 유틸 =====================
def resource_path(relative_path: str) -> str:
    """
    PyInstaller --onefile(exe)와 일반 파이썬 실행 양쪽에서
    tcn.onnx, tcn_meta.json 같은 리소스 파일 경로를 안전하게 찾기 위한 헬퍼.
    """
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS  # exe가 풀려 있는 임시 폴더
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ===================== 상수/기본값 =====================
DEF_T    = 33              # 학습/클립 길이(03과 일치)
DEF_TH   = 0.80            # 스윙 확신도 임계
DEF_CD   = 0.80            # 스윙 확정 쿨다운(초)
UDP_IP   = "127.0.0.1"
UDP_PORT = 5052
EPS      = 1e-6

# Ready 관련
TH_READY        = 0.60     # 피크가 아닐 때 Ready 확정 최소 확률
SEND_READY_UDP  = False    # True면 Ready도 UDP로 알림(swing=False)

# 실시간 dt 안정화(웹캠 권장)
TARGET_FPS = 30.0
TARGET_DT  = 1.0 / TARGET_FPS
EMA_ALPHA  = 0.20
DT_MIN, DT_MAX = 1/90.0, 1/20.0  # 90~20fps

# 피크 검출 파라미터(정규화 좌표계 기준)
PEAK_WIN        = 5          # 최근 창 길이(우측 끝=현재 프레임)
V_MIN_WRIST     = 1.10       # 손목 속도 최소(정규화/초)  ← 0.80 → 1.10
V_MIN_ELBOW     = 0.80       # 팔꿈치 속도 최소          ← 0.50 → 0.80
PROM_MIN        = 0.06       # 돌출도(피크 - 인접값)     ← 0.05 → 0.06
VIS_THR         = 0.60       # 가시도 임계
USE_ELBOW_RATIO = 0.5        # 창 내 손목 가시도 비율이 낮으면 팔꿈치 사용

# 적응형 임계(평균+표준편차)
ADAPT_R = 12      # 최근 12프레임(마지막 제외)로 기준선 계산
ADAPT_K = 1.0     # 기준선 = max(v_min_base, mean + K * std)

# ===================== 유틸 =====================
def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= (p.sum(axis=1, keepdims=True) + EPS)
    return p

def to_3d_btd(x: np.ndarray, T: int, D: int) -> np.ndarray:
    if x.ndim == 3:
        return x
    if x.ndim == 2:
        return x[None, ...]
    x2 = np.squeeze(x)
    if x2.ndim == 3:
        return x2
    return x.reshape(1, T, -1)

# ===================== 카메라 스캔/CLI 선택 =====================
BACKENDS = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]

def try_open_once(dev_id: int):
    """여러 백엔드로 열어보고 성공 시 (cap, backend_name) 반환. 실패 시 (None, None)."""
    for be in BACKENDS:
        cap = cv2.VideoCapture(dev_id, be)
        if not cap.isOpened():
            try:
                cap.release()
            except:
                pass
            continue

        # 포맷/해상도/FPS 기본 고정 (DroidCam 안정화 목적)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # 첫 프레임 워밍업
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            continue

        try:
            name = cap.getBackendName()
        except Exception:
            name = "UNKNOWN"
        return cap, name

    return None, None

def scan_cameras(max_dev: int = 10):
    """
    사용 가능한 카메라 인덱스와 백엔드명, 해상도 스캔.
    반환: [(dev_id, backend_name, (w,h)), ...]
    """
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
    """
    CMD에서 카메라 선택:
    - 현재 연결된 디바이스 스캔 → 번호. Device N (backend, WxH) 형식 출력
    - 번호 입력 → 해당 dev_id 반환
    - 'r' → 다시 스캔
    - 'q' 또는 빈 입력 → None 반환(종료)
    """
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
            print("[INFO] 사용자 종료 선택.")
            return None

        if sel.lower() == "r":
            print("[INFO] 다시 스캔합니다.\n")
            continue

        # 숫자 입력 시도
        try:
            idx = int(sel)
        except ValueError:
            print(f"[WARN] 잘못된 입력입니다: {sel}\n")
            continue

        if not found:
            print("[WARN] 현재 사용 가능한 카메라가 없습니다.\n")
            continue

        if 0 <= idx < len(found):
            dev_id = found[idx][0]
            print(f"[INFO] 선택된 카메라: 리스트[{idx}] → Device {dev_id}\n")
            return dev_id
        else:
            print(f"[WARN] 범위를 벗어난 번호입니다: {idx}\n")

def open_camera(dev_id: int):
    """선택된 dev_id로 안정적으로 카메라 열기."""
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
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 워밍업
        t0 = time.time()
        ok = False
        while time.time() - t0 < 2.0:
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.03)

        if ok:
            try:
                name = cap.getBackendName()
            except Exception:
                name = "UNKNOWN"
            print(f"[INFO] Camera opened: device={dev_id}, backend={name}")
            return cap

        cap.release()

    raise RuntimeError(f"Camera open failed for device {dev_id}")

# ===================== 특징 빌더 (03과 정렬) =====================
# 16D: [x,y, vx,vy, v, ax,ay, theta, front, d_ws, d_we, d_wl, dtheta, phi_shoulder(0), theta_rel, ang_elbow]
class FeatureBuilder:
    def __init__(self, T: int, D: int):
        self.T, self.D = T, D
        self.hist = {
            k: deque(maxlen=T) for k in [
                't',
                'x16', 'y16', 'x14', 'y14', 'x12', 'y12', 'x15', 'y15',
                'v_w', 'v_e', 'vis16', 'vis14'
            ]
        }
        self.t_accum = 0.0

    @staticmethod
    def _grad(a, t):
        n = len(a)
        if n < 2:
            return np.zeros(n, dtype=np.float32)
        a = np.asarray(a, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        out = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i == 0:
                dt = (t[i+1] - t[i]) + EPS
                out[i] = (a[i+1] - a[i]) / dt
            elif i == n-1:
                dt = (t[i] - t[i-1]) + EPS
                out[i] = (a[i] - a[i-1]) / dt
            else:
                dt = (t[i+1] - t[i-1]) + EPS
                out[i] = (a[i+1] - a[i-1]) / dt
        return out.astype(np.float32)

    def push_and_get(self, lm, dt: float):
        # 원 좌표 & 가시도
        def P(i):
            return (lm[i].x, lm[i].y)

        def V(i):
            return getattr(lm[i], 'visibility', 1.0)

        r_wrist = P(16)
        r_elbow = P(14)
        r_sh    = P(12)
        l_wrist = P(15)
        vis16 = float(V(16))
        vis14 = float(V(14))

        # 어깨 중심 정규화 + 상완 길이 스케일
        cx, cy = r_sh
        scale = max(math.hypot(r_sh[0] - r_elbow[0],
                               r_sh[1] - r_elbow[1]), 1e-6)

        def norm(p):
            return ((p[0] - cx) / scale, (p[1] - cy) / scale)

        nw  = norm(r_wrist)
        ne  = norm(r_elbow)
        ns  = norm(r_sh)
        nlw = norm(l_wrist)

        # 누적 시간/버퍼
        self.t_accum += dt
        H = self.hist
        H['t'].append(self.t_accum)
        H['x16'].append(nw[0]);  H['y16'].append(nw[1])
        H['x14'].append(ne[0]);  H['y14'].append(ne[1])
        H['x12'].append(ns[0]);  H['y12'].append(ns[1])
        H['x15'].append(nlw[0]); H['y15'].append(nlw[1])
        H['vis16'].append(vis16); H['vis14'].append(vis14)

        # 즉시 특징 생성 조건
        if len(H['t']) < self.T:
            return None, None

        # 배열화
        t  = np.asarray(H['t'], dtype=np.float32)
        xw = np.asarray(H['x16'], dtype=np.float32)
        yw = np.asarray(H['y16'], dtype=np.float32)
        xe = np.asarray(H['x14'], dtype=np.float32)
        ye = np.asarray(H['y14'], dtype=np.float32)
        xs = np.asarray(H['x12'], dtype=np.float32)
        ys = np.asarray(H['y12'], dtype=np.float32)
        xl = np.asarray(H['x15'], dtype=np.float32)
        yl = np.asarray(H['y15'], dtype=np.float32)

        # 속도/가속(손목)
        vx = self._grad(xw, t)
        vy = self._grad(yw, t)
        v  = np.hypot(vx, vy).astype(np.float32)
        ax = self._grad(vx, t)
        ay = self._grad(vy, t)

        # 팔꿈치 속도(피크 대체용)
        vxe = self._grad(xe, t)
        vye = self._grad(ye, t)
        v_e = np.hypot(vxe, vye).astype(np.float32)

        # 각도/거리/상대각
        dx = xw - xs
        dy = yw - ys
        theta  = np.arctan2(dy, dx).astype(np.float32)
        dtheta = self._grad(theta, t)

        # front: 어깨 중심 기준 → 손목 x가 0 이상이면 전방(오른쪽)
        front = (xw >= 0).astype(np.float32)

        # 거리들
        d_ws = np.hypot(xw - xs, yw - ys).astype(np.float32)   # wrist-shoulder
        d_we = np.hypot(xw - xe, yw - ye).astype(np.float32)   # wrist-elbow
        d_wl = np.hypot(xw - xl, yw - yl).astype(np.float32)   # wrist-left_wrist

        # phi_shoulder 사용하지 않음 → 0, 상대각 = theta
        phi_shoulder = np.zeros_like(theta, dtype=np.float32)
        theta_rel    = theta.copy()

        # 팔꿈치 내각(12-14-16)
        v_se_x, v_se_y = (xe - xs), (ye - ys)
        v_we_x, v_we_y = (xw - xe), (yw - ye)
        dot = (v_se_x * v_we_x + v_se_y * v_we_y)
        n1  = np.hypot(v_se_x, v_se_y) + 1e-6
        n2  = np.hypot(v_we_x, v_we_y) + 1e-6
        cosang    = np.clip(dot / (n1 * n2), -1.0, 1.0)
        ang_elbow = np.arccos(cosang).astype(np.float32)

        feats = np.stack([
            xw, yw, vx, vy, v, ax, ay, theta,
            front, d_ws, d_we, d_wl,
            dtheta, phi_shoulder, theta_rel, ang_elbow
        ], axis=1).astype(np.float32)

        # 속도/가시도 기록(피크 검사용)
        H['v_w'].append(float(v[-1]))
        H['v_e'].append(float(v_e[-1]))

        diag = {
            'v_w_hist':   list(H['v_w']),
            'v_e_hist':   list(H['v_e']),
            'vis16_hist': list(H['vis16']),
            'vis14_hist': list(H['vis14']),
        }

        # D 정합
        if feats.shape[1] < self.D:
            pad = np.zeros((self.T, self.D - feats.shape[1]),
                           dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=1)
        elif feats.shape[1] > self.D:
            feats = feats[:, :self.D]

        return feats, diag

# ===================== 피크 트리거 (적응형 포함) =====================
def is_recent_peak(seq, v_min_base, prom_min):
    """
    최근 값(마지막 프레임)을 피크로 볼지 판단.
    - 3프레임 국소최대 + 돌출도 + (적응형)최소속도(mean + K*std)
    """
    n = len(seq)
    if n < max(PEAK_WIN, 3):
        return False

    c  = float(seq[-1])
    p1 = float(seq[-2])
    p2 = float(seq[-3]) if n >= 3 else p1

    # 적응형 최소속도: 마지막 프레임 제외 구간
    if n >= ADAPT_R + 1:
        win = np.asarray(seq[-(ADAPT_R+1):-1], dtype=np.float32)
        mu  = float(np.mean(win))
        sd  = float(np.std(win))
        v_min_adapt = max(v_min_base, mu + ADAPT_K * sd)
    else:
        v_min_adapt = v_min_base

    cond_local_max = (c > p1) and (c > p2)
    cond_prom      = (c - max(p1, p2)) >= prom_min
    cond_speed     = (c >= v_min_adapt)

    return cond_local_max and cond_prom and cond_speed

def decide_peak(diag):
    vw   = diag['v_w_hist']
    ve   = diag['v_e_hist']
    visw = diag['vis16_hist']

    # 손목 가시도 낮으면 팔꿈치로 대체
    use_elbow = False
    if len(visw) >= PEAK_WIN:
        win_w = visw[-PEAK_WIN:]
        if sum(1 for v in win_w if v >= VIS_THR) / len(win_w) < USE_ELBOW_RATIO:
            use_elbow = True

    return is_recent_peak(
        ve if use_elbow else vw,
        V_MIN_ELBOW if use_elbow else V_MIN_WRIST,
        PROM_MIN
    )

# ===================== 시각화 =====================
def draw_body_ui(frame, lm, w, h, classes,
                 prob=None, last_detected="None", last_conf=0.0):
    key_map = {
        16: "R-Wrist",
        14: "R-Elbow",
        12: "R-Shoulder",
        15: "L-Wrist"
    }
    trail_joints = (16, 14, 12, 15)
    joint_colors = {
        16: (0, 200, 255),
        14: (255, 180, 0),
        12: (0, 140, 255),
        15: (180, 0, 200)
    }
    default_color = (200, 200, 200)

    # 관절 라벨링
    for idx, name in key_map.items():
        x = int(lm[idx].x * w)
        y = int(lm[idx].y * h)
        cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
        cv2.putText(frame, f"{name}({idx})", (x+6, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

    # 잔상 trail
    if not hasattr(draw_body_ui, "_trails"):
        draw_body_ui._trails = {}
    trails = draw_body_ui._trails
    trail_len = 25

    for j in trail_joints:
        if j not in trails or trails[j].maxlen != trail_len:
            trails[j] = deque(maxlen=trail_len)

    for j in list(trails.keys()):
        if j not in trail_joints:
            del trails[j]

    for j in trail_joints:
        xj = int(lm[j].x * w)
        yj = int(lm[j].y * h)
        trails[j].append((xj, yj))

    for j, q in trails.items():
        color = joint_colors.get(j, default_color)
        for i in range(1, len(q)):
            cv2.line(frame, q[i-1], q[i], color, 2)

    # 클래스별 확률 막대
    if prob is not None:
        base_x, base_y = 16, 140
        bar_w, bar_h = 220, 18
        gap = 8
        for i, cls in enumerate(classes):
            p = float(prob[0, i])
            x1 = base_x
            y1 = base_y + i * (bar_h + gap)
            x2 = x1 + int(bar_w * max(0.0, min(1.0, p)))

            cv2.rectangle(frame, (x1, y1), (x1+bar_w, y1+bar_h),
                          (50, 50, 50), 1)
            cv2.rectangle(frame, (x1, y1), (x2, y1+bar_h),
                          (60, 180, 75), -1)
            cv2.putText(frame, f"{cls}: {p:.2f}",
                (x1+6, y1+bar_h-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1)

# ===================== 메인 =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, default="tcn.onnx")
    ap.add_argument("--meta", type=str, default="tcn_meta.json")
    ap.add_argument("--device", type=int, default=-1)  # 필요시 직접 지정용
    ap.add_argument("--ip", type=str, default=UDP_IP)
    ap.add_argument("--port", type=int, default=UDP_PORT)
    ap.add_argument("--th", type=float, default=DEF_TH)
    ap.add_argument("--cooldown", type=float, default=DEF_CD)
    ap.add_argument("--show_landmarks", action="store_true", default=False)
    # 점프 감지 전용 옵션 (y 감소=위로 이동)
    ap.add_argument("--jump_thr", type=float, default=2.00,
                    help="점프 임계값( -dy/dt >= jump_thr 이면 점프, 단위 1/s )")
    ap.add_argument("--jump_hold", type=float, default=0.50,
                    help="점프 표시 유지 시간(초)")
    ap.add_argument("--jump_send_cooldown", type=float, default=0.30,
                    help="점프 UDP 연속 전송 쿨다운(초)")
    ap.add_argument("--no_picker", action="store_true",
                    help="CMD 카메라 선택 없이 --device 값으로 바로 열기")
    # ---- 피크 튜닝 CLI ----
    ap.add_argument("--peak_win", type=int, default=PEAK_WIN)
    ap.add_argument("--vmin_wrist", type=float, default=V_MIN_WRIST)
    ap.add_argument("--vmin_elbow", type=float, default=V_MIN_ELBOW)
    ap.add_argument("--prom_min", type=float, default=PROM_MIN)
    ap.add_argument("--vis_thr", type=float, default=VIS_THR)
    ap.add_argument("--use_elbow_ratio", type=float, default=USE_ELBOW_RATIO)
    ap.add_argument("--adapt_r", type=int, default=ADAPT_R)
    ap.add_argument("--adapt_k", type=float, default=ADAPT_K)
    args = ap.parse_args()

    # 전역 상수 값을 CLI로 주입 (global 없이 안전하게)
    for k, v in {
        "PEAK_WIN": args.peak_win,
        "V_MIN_WRIST": args.vmin_wrist,
        "V_MIN_ELBOW": args.vmin_elbow,
        "PROM_MIN": args.prom_min,
        "VIS_THR": args.vis_thr,
        "USE_ELBOW_RATIO": args.use_elbow_ratio,
        "ADAPT_R": args.adapt_r,
        "ADAPT_K": args.adapt_k,
    }.items():
        globals()[k] = v

    # ===== 카메라 선택 (CLI or 고정 device) =====
    if args.no_picker and args.device >= 0:
        chosen_dev = args.device
        print(f"[INFO] --no_picker 사용, device={chosen_dev} 로 직접 오픈 시도")
    else:
        chosen_dev = choose_camera_cli(max_dev=10)
        if chosen_dev is None:
            print("[INFO] 프로그램 종료.")
            return

    print(f"[INFO] 최종 선택 카메라: Device {chosen_dev}")

    # ===== 메타 로드 (PyInstaller 대응: resource_path 사용) =====
    meta_path = resource_path(args.meta)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    classes = meta["classes"]
    D       = int(meta["feat_dim"])
    T       = int(meta.get("target_T", DEF_T))
    in_name = meta.get("input_name", "clips")
    out_name= meta.get("output_name", "logits")
    mu  = np.asarray(meta["zscore_mu"],  dtype=np.float32)
    std = np.asarray(meta["zscore_std"], dtype=np.float32)
    std = np.where(np.abs(std) < 1e-8, 1.0, std)

    # ONNX & UDP (PyInstaller 대응: resource_path 사용)
    try:
        onnx_path = resource_path(args.onnx)
        sess = ort.InferenceSession(onnx_path,
                                    providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"[ERROR] ONNX 로드 실패: {e}", file=sys.stderr)
        sys.exit(1)

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

    # ===== 선택된 카메라 열기 =====
    cap = open_camera(chosen_dev)

    fb = FeatureBuilder(T=T, D=D)
    last_fire_ts   = 0.0
    last_detected  = "None"
    last_conf      = 0.0
    last_prob      = None
    last_wrist_speed = 0.0

    # 시간/표시
    t_prev   = time.perf_counter()
    dt_ema   = TARGET_DT
    printed_io_spec = False
    fps_vis  = 0.0
    fps_prev_wall = time.perf_counter()

    # Ready 인덱스(있을 때만)
    try:
        ready_idx = classes.index('Ready')
    except ValueError:
        ready_idx = None

    # 점프 감지 상태
    prev_nose_y       = None
    last_jump_ts      = -1e9
    last_jump_send_ts = -1e9

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        t_now  = time.perf_counter()
        dt_raw = t_now - t_prev
        t_prev = t_now

        # dt EMA
        dt_ema = EMA_ALPHA * dt_raw + (1 - EMA_ALPHA) * dt_ema
        dt_ema = max(DT_MIN, min(DT_MAX, dt_ema))

        # 모델 시간축은 항상 TARGET_DT로 간주
        dt_for_model = TARGET_DT

        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            # ---------- 점프 감지 (Nose y: 감소=위로 이동) ----------
            nose_y = float(lm[0].y)  # 0~1, 위로 갈수록 값이 감소
            if prev_nose_y is not None and dt_ema > 0:
                dy   = nose_y - prev_nose_y
                dydt = dy / dt_ema
                if (-dydt) >= args.jump_thr:
                    last_jump_ts = t_now
                    if (t_now - last_jump_send_ts) >= args.jump_send_cooldown:
                        jump_pkt = {
                            "jump": True,
                            "speed": round(-dydt, 4),
                            "ts": round(t_now, 3)
                        }
                        try:
                            sock.sendto(json.dumps(jump_pkt).encode("utf-8"),
                                        (args.ip, args.port))
                        except Exception as e:
                            print(f"[WARN] UDP send failed (jump): {e}",
                                  file=sys.stderr)
                        last_jump_send_ts = t_now
            prev_nose_y = nose_y
            # -----------------------------------------------

            feats, diag = fb.push_and_get(lm, dt=dt_for_model)

            if diag and diag.get('v_w_hist'):
                last_wrist_speed = float(diag['v_w_hist'][-1])

            draw_body_ui(frame, lm, w, h, classes,
                         prob=last_prob,
                         last_detected=last_detected,
                         last_conf=last_conf)

            if args.show_landmarks:
                mp_draw.draw_landmarks(frame,
                                       res.pose_landmarks,
                                       mp_pose.POSE_CONNECTIONS)

            if feats is not None and diag is not None:
                X  = feats.astype(np.float32)
                Xn = (X - mu) / (std + EPS)
                Xn = to_3d_btd(Xn, T, D)

                if not printed_io_spec:
                    ins = sess.get_inputs()[0]
                    print("[INFO] ONNX input spec:",
                          ins.name, ins.shape, ins.type)
                    print("[INFO] Feeding shape:",
                          Xn.shape, Xn.dtype)
                    printed_io_spec = True

                logits = sess.run([out_name], {in_name: Xn})[0]
                p      = softmax(logits)
                last_prob = p
                cls_idx = int(np.argmax(p, axis=1)[0])
                conf    = float(p[0, cls_idx])
                if 0 <= cls_idx < len(classes):
                    cls_name = classes[cls_idx]
                else:
                    cls_name = str(cls_idx)

                peak_now = decide_peak(diag)

                if peak_now:
                    if cls_name != 'Ready':
                        if (conf >= args.th and
                            (t_now - last_fire_ts) >= args.cooldown):
                            pkt = {
                                "swing": True,
                                "class": cls_name,
                                "conf": round(conf, 4),
                                "ts":   round(t_now, 3)
                            }
                            try:
                                sock.sendto(json.dumps(pkt).encode("utf-8"),
                                            (args.ip, args.port))
                            except Exception as e:
                                print(f"[WARN] UDP send failed: {e}",
                                      file=sys.stderr)
                            last_fire_ts  = t_now
                            last_detected = cls_name
                            last_conf     = conf
                else:
                    # 피크가 아닐 때 Ready 확률이 높으면 Ready 상태로 표시
                    if (ready_idx is not None and
                        cls_idx == ready_idx and
                        conf >= TH_READY):
                        last_detected = 'Ready'
                        last_conf     = conf
                        if SEND_READY_UDP:
                            pkt = {
                                "swing": False,
                                "class": "Ready",
                                "conf":  round(conf, 4),
                                "ts":    round(t_now, 3)
                            }
                            try:
                                sock.sendto(json.dumps(pkt).encode("utf-8"),
                                            (args.ip, args.port))
                            except Exception as e:
                                print(f"[WARN] UDP send failed: {e}",
                                      file=sys.stderr)

        # 상태 텍스트들
        cv2.putText(frame,
                    f"Last confirmed: {last_detected} ({last_conf:.2f})",
                    (16, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 220, 255), 2)

        now_wall = time.perf_counter()
        fps_dt   = now_wall - fps_prev_wall
        if fps_dt > 0:
            fps_vis = 1.0 / fps_dt
        fps_prev_wall = now_wall
        cv2.putText(frame,
                    f"FPS: {fps_vis:.1f}",
                    (16, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.putText(frame,
                    f"Wrist speed: {last_wrist_speed:.3f} (norm/s)",
                    (16, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2)

        # ===== 점프 표시 (하단 중앙, 빨간 글씨) =====
        if (time.perf_counter() - last_jump_ts) <= args.jump_hold:
            text = "JUMP!"
            (tw, th), _ = cv2.getTextSize(text,
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          1.2, 3)
            cx = w // 2 - tw // 2
            cy = h - 20
            cv2.putText(frame, text, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)

        cv2.imshow("Pose Inference (Webcam, Hybrid + Jump y↓)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
