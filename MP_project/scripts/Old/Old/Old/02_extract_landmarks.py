# 02_extract_landmarks.py
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

    def L(i):
        p = lm[i]
        return (p.x, p.y, getattr(p, 'visibility', 1.0))

    # 필요한 관절만 추출 (순서 고정)
    # Right wrist(16), Right elbow(14), Right shoulder(12),
    # Left shoulder(11), Left elbow(13), Right hip(24)
    x16,y16,v16 = L(16)  # 오른손목
    x14,y14,v14 = L(14)  # 오른팔꿈치
    x12,y12,v12 = L(12)  # 오른어깨
    x11,y11,v11 = L(11)  # 왼어깨
    x13,y13,v13 = L(13)  # 왼팔꿈치
    x24,y24,v24 = L(24)  # 오른엉덩이

    rows.append([
        t,
        x16,y16,v16,
        x14,y14,v14,
        x12,y12,v12,
        x11,y11,v11,
        x13,y13,v13,
        x24,y24,v24
    ])

cap.release()

cols = [
    't',
    'x16','y16','v16',
    'x14','y14','v14',
    'x12','y12','v12',
    'x11','y11','v11',
    'x13','y13','v13',
    'x24','y24','v24'
]

pd.DataFrame(rows, columns=cols).to_parquet(out)
print('saved:', out)
