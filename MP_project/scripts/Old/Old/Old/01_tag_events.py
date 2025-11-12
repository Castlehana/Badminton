# 01_tag_events_block.py
import cv2, csv, sys, os

ORDER = ['Clear','Drop','Hairpin','Drive','Under']
REPS_PER_CLASS = 10

mp4 = sys.argv[1]; out = sys.argv[2]
os.makedirs(os.path.dirname(out), exist_ok=True)
cap = cv2.VideoCapture(mp4)
fps = cap.get(cv2.CAP_PROP_FPS) or 30

rows=[('timestamp_sec','class','note')]
i_cls, i_rep = 0, 0
playing=True

while True:
    if playing:
        ret, frame = cap.read()
        if not ret: break
    t = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
    disp = frame.copy()
    if disp is not None:
        msg=f"{t:7.3f}s | Class {i_cls+1}/{len(ORDER)}: {ORDER[i_cls]}  Rep {i_rep+1}/{REPS_PER_CLASS}"
        cv2.putText(disp, msg, (12,32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30,220,255), 2)
        cv2.putText(disp, "[Space]Mark  [Enter]NextClass  [Backspace]Undo  [S]ave  [ESC]Quit",
                    (12,60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230,230,230), 2)
        cv2.imshow('tag_block', disp)

    k = cv2.waitKey(int(1000//fps)) & 0xFF
    if k == 32:  # Space: mark
        rows.append((round(t,3), ORDER[i_cls], 'block'))
        i_rep += 1
        cap.set(cv2.CAP_PROP_POS_MSEC, (t+0.4)*1000)
        if i_rep >= REPS_PER_CLASS:
            i_rep = 0; i_cls = min(i_cls+1, len(ORDER)-1)
    elif k == 13:  # Enter: next class
        i_rep = 0; i_cls = min(i_cls+1, len(ORDER)-1)
    elif k == 8:   # Backspace
        if len(rows)>1:
            rows.pop()
            if i_rep>0: i_rep-=1
    elif k in (ord('s'), ord('S')):
        with open(out,'w',newline='') as f: csv.writer(f).writerows(rows)
        print('saved:', out)
    elif k == 27:
        break

cap.release(); cv2.destroyAllWindows()
with open(out,'w',newline='') as f: csv.writer(f).writerows(rows)
print('final saved:', out)
