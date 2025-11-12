import cv2, mediapipe as mp, pandas as pd, sys, os

# Mediapipe Pose 초기화
mp_pose = mp.solutions.pose.Pose(
    model_complexity=0,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp4 = sys.argv[1]
out = sys.argv[2]
os.makedirs(os.path.dirname(out), exist_ok=True)

cap = cv2.VideoCapture(mp4)
rows = []

def L(lms, i):
    p = lms[i]
    return (p.x, p.y, getattr(p, 'visibility', 1.0))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mp_pose.process(rgb)

    if not res.pose_landmarks:
        continue

    lm = res.pose_landmarks.landmark

    # 추적 관절: 16(R-Wrist), 14(R-Elbow), 12(R-Shoulder), 15(L-Wrist)
    x16,y16,v16 = L(lm, 16)  # 오른손목
    x14,y14,v14 = L(lm, 14)  # 오른팔꿈치
    x12,y12,v12 = L(lm, 12)  # 오른어깨
    x15,y15,v15 = L(lm, 15)  # 왼손목

    rows.append([
        t,
        x16,y16,v16,
        x14,y14,v14,
        x12,y12,v12,
        x15,y15,v15
    ])

cap.release()

cols = [
    't',
    'x16','y16','v16',
    'x14','y14','v14',
    'x12','y12','v12',
    'x15','y15','v15'
]

pd.DataFrame(rows, columns=cols).to_parquet(out)
print('saved:', out)
