import cv2, csv, sys, os, time
from tkinter import Tk, filedialog

# ====== 파일 탐색기로 mp4 선택 ======
Tk().withdraw()  # GUI 창 숨김
video_path = filedialog.askopenfilename(
    title="Select MP4 Video",
    filetypes=[("MP4 files", "*.mp4")]
)
if not video_path:
    print("[ERR] No video selected."); sys.exit(1)

# ====== 자동 이름 및 저장 경로 ======
base = os.path.splitext(os.path.basename(video_path))[0]   # 예: S001_side
out_dir = os.path.join("dataset", "events")
os.makedirs(out_dir, exist_ok=True)
out_csv = os.path.join(out_dir, f"{base}.csv")

print(f"[INFO] Selected video: {video_path}")
print(f"[INFO] Output CSV:     {out_csv}")

# ====== 기본 설정 ======
ORDER = ['Clear','Drop','Hairpin','Drive','Under']
REPS_PER_CLASS = 10

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

rows=[('start_sec','end_sec','class','note')]
i_cls, i_rep = 0, 0
playing=True
swing_active = False
t_start = None

def now_sec():
    return cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0

def jump(dt):
    t = now_sec() + dt
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t)*1000.0)

def save_csv():
    with open(out_csv,'w',newline='') as f: csv.writer(f).writerows(rows)
    print('saved:', out_csv)

# ====== 메인 루프 ======
while True:
    if playing or not swing_active:
        ret, frame = cap.read()
        if not ret: break
    t = now_sec()
    disp = frame.copy() if 'frame' in locals() else None
    if disp is not None:
        msg1=f"{t:7.3f}s | Class {i_cls+1}/{len(ORDER)}: {ORDER[i_cls]}  Rep {i_rep+1}/{REPS_PER_CLASS}"
        msg2="[Space]Start/End  [Enter]NextClass  [Backspace]Undo  [S]ave  [ESC]Quit"
        state=f"STATE: {'SWING-REC' if swing_active else 'READY'}"
        cv2.putText(disp, msg1, (12,32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30,220,255), 2)
        cv2.putText(disp, msg2, (12,60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230,230,230), 2)
        cv2.putText(disp, state,(12,88), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80,240,120) if swing_active else (180,180,180), 2)
        cv2.imshow('tag_interval', disp)

    k = cv2.waitKey(int(1000//fps)) & 0xFF
    if k == 32:  # Space: start/end toggle
        if not swing_active:
            t_start = now_sec()
            swing_active = True
        else:
            t_end = now_sec()
            if t_end > t_start + 1e-3:
                rows.append((round(t_start,3), round(t_end,3), ORDER[i_cls], 'interval'))
                i_rep += 1
                swing_active = False
                # jump(0.40)  # 건너뛰기 제거
                if i_rep >= REPS_PER_CLASS:
                    i_rep = 0; i_cls = min(i_cls+1, len(ORDER)-1)
    elif k == 13:  # Enter: next class
        if swing_active:
            t_end = now_sec()
            if t_end > t_start + 1e-3:
                rows.append((round(t_start,3), round(t_end,3), ORDER[i_cls], 'auto-end-by-enter'))
                i_rep += 1
        swing_active = False
        i_rep = 0; i_cls = min(i_cls+1, len(ORDER)-1)
    elif k == 8:   # Backspace: undo last finished interval
        if swing_active:
            swing_active = False
        else:
            if len(rows)>1:
                rows.pop()
                if i_rep>0: i_rep-=1
    elif k in (ord('s'), ord('S')):
        save_csv()
    elif k == 27:
        break

cap.release(); cv2.destroyAllWindows()
save_csv()
print('final saved:', out_csv)
