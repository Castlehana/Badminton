# scripts/05_train_tcn.py
# -*- coding: utf-8 -*-
# - 입력: dataset/clips/raw/class=*/*.npz (X:(T,D), y, meta(json))
# - 표준화: z-score (train 통계)
# - 손실: FocalLoss(γ=1.5, label_smoothing=0.05)
# - 증강: 시간축 roll(±2) 학습시에만
# - 스케줄러: ReduceLROnPlateau(mode='max')
# - 결과 저장: best.pt (프로젝트 루트: {'model','mu','std','feat_dim','classes'})

import os, re, glob, json, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- 기본 설정 ----------------
random.seed(42); np.random.seed(42); torch.manual_seed(42)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH = 64
EPOCHS = 60
INIT_LR = 1e-3
AUG_SHIFT = 2
NUM_CLASSES = 5
CLASSES = ['Clear','Drive','Drop','Under','Hairpin']
TARGET_T = 32
TARGET_D = 16

# ---------------- 유틸 ----------------
def _resample_time(x, target_T=TARGET_T):
    T, D = x.shape
    if T == target_T: return x
    src = np.linspace(0, 1, T, dtype=np.float32)
    dst = np.linspace(0, 1, target_T, dtype=np.float32)
    out = np.empty((target_T, D), dtype=x.dtype)
    for d in range(D):
        out[:, d] = np.interp(dst, src, x[:, d])
    return out

def _fix_feat_dim(x, target_D=TARGET_D):
    T, D = x.shape
    if D == target_D: return x
    if D < target_D:
        return np.concatenate([x, np.zeros((T, target_D-D), dtype=x.dtype)], axis=1)
    return x[:, :target_D]

def _safe_load_npz(p):
    d = np.load(p, allow_pickle=True)
    X = d['X'].astype('float32')
    if X.shape[0] != TARGET_T: X = _resample_time(X, TARGET_T)
    if X.shape[1] != TARGET_D: X = _fix_feat_dim(X, TARGET_D)
    if not np.isfinite(X).all():
        raise ValueError("non-finite in features")
    y = int(d['y'])
    return X, y

def _list_files_for_videos(videos):
    files = []
    for v in videos:
        pat = os.path.join('dataset','clips','raw','class=*', f'{v}_*.npz')
        files += glob.glob(pat)
    return sorted(files)

def _discover_videos():
    paths = glob.glob(os.path.join('dataset','clips','raw','class=*','*.npz'))
    vids = set()
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        # 베이스네임에서 맨 앞의 비디오 키(예: S001_side_000123 → S001_side)
        m = re.match(r'([A-Za-z0-9]+_[A-Za-z0-9]+)', base)
        vids.add(m.group(1) if m else base)
    return sorted(vids)

def load_npzs(npz_files):
    Xs, ys, bad = [], [], []
    for p in npz_files:
        try:
            X, y = _safe_load_npz(p)
            Xs.append(X); ys.append(y)
        except Exception as e:
            bad.append((p, str(e)))
    if bad:
        print("[WARN] skipped {} files due to shape/NaN/etc:".format(len(bad)))
        for b in bad[:10]:
            print("   -", b[0], "->", b[1])
        if len(bad) > 10:
            print("   - ...")
    if len(Xs) == 0:
        raise RuntimeError("no valid npz after normalization")
    X = np.stack(Xs, axis=0)
    y = np.array(ys, dtype=np.int64)
    return X, y

def zscore_fit(X):
    # X: (N,T,D)
    mu = X.mean(axis=(0,1), keepdims=True)
    std = X.std(axis=(0,1), keepdims=True) + 1e-6
    return mu, std

def zscore_apply(X, mu, std):
    return (X - mu)/std

def roll_time(x, s):
    s = int(s)
    if s == 0: return x
    return np.concatenate([x[-s:], x[:-s]], axis=0)

# ---------------- 모델 ----------------
class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, dil=1, p=0.15):
        super().__init__()
        pad = (k-1)*dil
        self.pad1 = nn.ConstantPad2d((pad,0,0,0), 0.0)
        self.c1 = nn.Conv2d(c_in, c_out, (1,k), dilation=(1,dil))
        self.b1 = nn.BatchNorm2d(c_out)
        self.pad2 = nn.ConstantPad2d((pad,0,0,0), 0.0)
        self.c2 = nn.Conv2d(c_out, c_out, (1,k), dilation=(1,dil))
        self.b2 = nn.BatchNorm2d(c_out)
        self.drop = nn.Dropout(p)
        self.res = nn.Conv2d(c_in, c_out, 1) if c_in!=c_out else nn.Identity()
    def forward(self, x):
        y = self.pad1(x); y = F.relu(self.b1(self.c1(y)))
        y = self.pad2(y); y = F.relu(self.b2(self.c2(y)))
        y = self.drop(y)
        return F.relu(y + self.res(x))

class TCN(nn.Module):
    def __init__(self, feat_dim, num_classes=NUM_CLASSES):
        super().__init__()
        # 입력 (B,T,D) → (B,1,D,T)
        self.b1 = TCNBlock(1,   64, k=3, dil=1, p=0.20)
        self.b2 = TCNBlock(64, 128, k=3, dil=2, p=0.20)
        self.b3 = TCNBlock(128,128, k=3, dil=4, p=0.20)
        # ONNX 호환성: (feat_dim,1)로 고정 풀링 → Flatten → FC
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((feat_dim,1)),
            nn.Flatten(1),
            nn.Linear(128*feat_dim, 256),
            nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        # x: (B,T,D) → (B,1,D,T)
        x = x.transpose(1,2).unsqueeze(1)
        x = self.b1(x); x = self.b2(x); x = self.b3(x)
        return self.head(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction='mean', label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ls = label_smoothing
    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits, target,
            weight=self.alpha,
            reduction='none',
            label_smoothing=self.ls
        )
        with torch.no_grad():
            pt = torch.softmax(logits, dim=1).gather(1, target.view(-1,1)).clamp_min(1e-6).squeeze(1)
        focal = ((1-pt)**self.gamma) * ce
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal

# ---------------- 학습 1폴드 ----------------
def train_one_fold(name, train_ids, val_ids, test_ids):
    print(f'\n[Fold:{name}] train={train_ids} val={val_ids} test={test_ids}')
    tr_files = _list_files_for_videos(train_ids)
    va_files = _list_files_for_videos(val_ids)
    te_files = _list_files_for_videos(test_ids) if test_ids else []

    if len(tr_files)==0 or len(va_files)==0:
        raise RuntimeError(f'empty split: train={len(tr_files)} val={len(va_files)}')

    # 로드 & 표준화
    Xtr, ytr = load_npzs(tr_files)
    Xva, yva = load_npzs(va_files)
    mu, std = zscore_fit(Xtr)
    Xtr = zscore_apply(Xtr, mu, std)
    Xva = zscore_apply(Xva, mu, std)

    # 증강(시간 roll) - 학습시에만
    def augment_batch(x):
        if random.random() < 0.5:
            s = random.choice([-AUG_SHIFT, AUG_SHIFT])
            return np.stack([roll_time(xx, s) for xx in x], axis=0)
        return x

    # 샘플 가중치(클래스 역비율 기반)
    class_counts = np.bincount(ytr, minlength=NUM_CLASSES)
    inv = class_counts.max() / (class_counts + 1e-6)
    sample_w = inv[ytr]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.float32),
        num_samples=len(sample_w),
        replacement=True
    )

    tr_ld = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                       batch_size=BATCH, sampler=sampler, drop_last=True)
    va_ld = DataLoader(TensorDataset(torch.tensor(Xva), torch.tensor(yva)),
                       batch_size=BATCH, shuffle=False)

    model = TCN(feat_dim=Xtr.shape[2], num_classes=NUM_CLASSES).to(DEVICE)

    # FocalLoss 알파 = 정규화된 역비율
    inv_norm = (1.0/(class_counts+1e-6))
    inv_norm = inv_norm / np.mean(inv_norm)
    alpha = torch.tensor(inv_norm, dtype=torch.float32, device=DEVICE)

    loss_fn = FocalLoss(alpha=alpha, gamma=1.5, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=4, verbose=True)

    best_acc, best_state = 0.0, None

    for ep in range(1, EPOCHS+1):
        # train
        model.train()
        tr_loss, tr_n = 0.0, 0
        for xb, yb in tr_ld:
            xb = torch.tensor(augment_batch(xb.numpy()), dtype=torch.float32, device=DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += float(loss.item()) * len(yb)
            tr_n += len(yb)

        # validate
        model.eval()
        va_correct, va_n = 0, 0
        with torch.no_grad():
            for xb, yb in va_ld:
                xb = xb.to(DEVICE).float(); yb = yb.to(DEVICE)
                logits = model(xb)
                va_correct += int((logits.argmax(1) == yb).sum().item())
                va_n += len(yb)
        va_acc = va_correct / max(1, va_n)
        sched.step(va_acc)

        print(f'[EP {ep:03d}] loss={tr_loss/max(1,tr_n):.4f}  val_acc={va_acc:.4f}  lr={opt.param_groups[0]["lr"]:.2e}')

        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {
                'model': model.state_dict(),
                'mu': mu.astype(np.float32),
                'std': std.astype(np.float32),
                'feat_dim': Xtr.shape[2],
                'classes': CLASSES
            }

    # 저장
    if best_state is not None:
        save_path = os.path.join('best.pt')
        torch.save(best_state, save_path)
        print(f'\n[OK] best model saved -> {save_path}')

    # 테스트
    if best_state is not None and test_ids:
        Xte, yte = load_npzs(te_files)
        Xte = zscore_apply(Xte, best_state['mu'], best_state['std'])
        te_ld = DataLoader(TensorDataset(torch.tensor(Xte), torch.tensor(yte)),
                           batch_size=BATCH, shuffle=False)

        model.load_state_dict(best_state['model'])
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in te_ld:
                xb = xb.to(DEVICE).float()
                preds += model(xb).argmax(1).cpu().tolist()

        print('\n[Test Classification Report]')
        print(classification_report(yte, preds, target_names=CLASSES, digits=2))
        print('[Confusion Matrix]')
        print(confusion_matrix(yte, preds))

    return best_acc

# ---------------- 실행 ----------------
if __name__ == "__main__":
    # splits.json이 있으면 우선 사용, 없으면 자동 분할
    splits_path = os.path.join('dataset','meta','splits.json')
    folds = []
    if os.path.exists(splits_path):
        with open(splits_path,'r',encoding='utf-8') as f:
            sp = json.load(f)
        if isinstance(sp.get('video_fixed'), dict):
            folds = [{
                'name': 'video_fixed',
                'train': sp['video_fixed'].get('train', []),
                'val':   sp['video_fixed'].get('val',   []),
                'test':  sp['video_fixed'].get('test',  [])
            }]
        elif isinstance(sp.get('video_lovo'), list) and sp['video_lovo']:
            folds = [{'name': f"lovo_{i}", 'train': fd.get('train',[]), 'val': fd.get('val',[]), 'test': fd.get('test',[])}
                     for i,fd in enumerate(sp['video_lovo'])]
        else:
            vids = _discover_videos()
            if len(vids) < 3:
                raise RuntimeError("need >=3 videos for auto split; or provide dataset/meta/splits.json")
            folds = [{'name': 'auto', 'train': vids[:-2], 'val': [vids[-2]], 'test': [vids[-1]]}]
    else:
        vids = _discover_videos()
        if len(vids) < 3:
            raise RuntimeError("need >=3 videos for auto split; or provide dataset/meta/splits.json")
        folds = [{'name': 'auto', 'train': vids[:-2], 'val': [vids[-2]], 'test': [vids[-1]]}]

    for fd in folds:
        train_one_fold(fd['name'], fd['train'], fd['val'], fd['test'])
