import cv2

# 여러 장치를 순차 탐색
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"[OK] Device {i} available")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Cam {i}", frame)
            cv2.waitKey(1000)
        cap.release()
cv2.destroyAllWindows()
