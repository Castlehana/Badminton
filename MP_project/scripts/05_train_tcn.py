# scripts/05_train_tcn.py  (T=25, asymmetric 16|1|8 window assumed by data)
import os, re, glob, json, random, math, argparse
import numpy as np
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

CLASSES = ['Clear','Drive','Drop','Under','Hairpin','Idle']
NUM_CLASSES = len(CLASSES)
CLS2ID = {c:i for i,c in enumerate(CLASSES)}

TARGET_T = 25
FEAT_DIM = 16
PRE_FR   = 16
POST_FR  = 8
INPUT_NAME  = 'clips'
OUTPUT_NAME = 'logits'

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def list_npz(root: str):
    pats = [os.path.join(root, f'class={c}', '*.npz') for c in CLASSES]
    files = []
    for p in pats:
        files.extend(sorted(glob.glob(p)))
    return files

def read_npz(path: str):
    with np.load(path, allow_pickle=True) as z:
        X = z['X'].astype(np.float32)
        y = int(z['y'])
        meta = json.loads(str(z['meta']))
    return X, y, meta

def group_by_video(files):
    buckets = defaultdict(list)
    for f in files:
        try:
            with np.load(f, allow_pickle=True) as z:
                meta = json.loads(str(z['meta']))
            key = meta.get('video', os.path.basename(f).split('_')[0])
        except Exception:
            key = os.path.basename(f).split('_')[0]
        buckets[key].append(f)
    return buckets

def make_split(files, ratio=(0.8,0.1,0.1)):
    buckets = group_by_video(files)
    vids = sorted(buckets.keys())
    random.shuffle(vids)
    n = len(vids)
    n_train = int(n*ratio[0]); n_val = int(n*ratio[1])
    train_v = set(vids[:n_train])
    val_v   = set(vids[n_train:n_train+n_val])
    test_v  = set(vids[n_train+n_val:])
    train = []; val = []; test = []
    for v, flist in buckets.items():
        if v in train_v: train += flist
        elif v in val_v: val += flist
        else: test += flist
    return sorted(train), sorted(val), sorted(test)

def load_split_from_txt(txt_path: str):
    with open(txt_path, 'r', encoding='utf-8') as f:
        files = [ln.strip() for ln in f if ln.strip()]
    return files

class ClipsDataset(Dataset):
    def __init__(self, files, z_mu=None, z_std=None, train=False, aug_roll=1):
        self.files = files
        self.train = train
        self.aug_roll = aug_roll
        self.mu = z_mu
        self.std = z_std
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        f = self.files[idx]
        X, y, meta = read_npz(f)
        if X.shape[0] != TARGET_T or X.shape[1] != FEAT_DIM:
            T, D = X.shape
            if T < TARGET_T:
                pad = np.zeros((TARGET_T - T, D), dtype=np.float32)
                X = np.concatenate([X, pad], axis=0)
            elif T > TARGET_T:
                X = X[:TARGET_T, :]
            if X.shape[1] < FEAT_DIM:
                pad = np.zeros((TARGET_T, FEAT_DIM - X.shape[1]), dtype=np.float32)
                X = np.concatenate([X, pad], axis=1)
            elif X.shape[1] > FEAT_DIM:
                X = X[:, :FEAT_DIM]
        if self.train and self.aug_roll > 0:
            k = random.randint(-self.aug_roll, self.aug_roll)
            if k != 0: X = np.roll(X, shift=k, axis=0)
        if self.mu is not None and self.std is not None:
            X = (X - self.mu) / (self.std + 1e-6)
        X = torch.from_numpy(X).float()
        y = torch.tensor(y).long()
        return X, y

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, d, p_drop=0.2):
        super().__init__()
        pad = (k-1)*d
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=d, padding=pad)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, dilation=d, padding=pad)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(p_drop)
        self.down  = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.relu  = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out); out = self.drop(out)
        out = self.conv2(out); out = self.bn2(out); out = self.drop(out)
        Tlen = x.shape[-1]; out = out[..., -Tlen:]
        res = self.down(x);  res = res[..., -Tlen:]
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, in_feat=FEAT_DIM, num_classes=NUM_CLASSES, p_drop=0.2):
        super().__init__()
        chs = [64, 128, 128]; ks = 3
        self.in_proj = nn.Conv1d(in_feat, chs[0], kernel_size=1)
        blocks = []; dils = [1,2,4,8]; c_in = chs[0]
        for d in dils:
            blocks.append(TemporalBlock(c_in, chs[1], ks, d, p_drop)); c_in = chs[1]
        blocks.append(TemporalBlock(c_in, chs[2], ks, 1, p_drop))
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(chs[2], num_classes)
    def forward(self, x):
        x = x.permute(0,2,1); x = self.in_proj(x); x = self.tcn(x)
        x = self.pool(x).squeeze(-1); return self.head(x)

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, label_smoothing=0.05, reduction='mean'):
        super().__init__(); self.gamma=gamma; self.ls=label_smoothing; self.reduction=reduction
    def forward(self, logits, target):
        B,K = logits.shape; logp = F.log_softmax(logits, dim=1); p = logp.exp()
        with torch.no_grad():
            true = torch.zeros_like(logp).scatter_(1, target.view(-1,1), 1.0)
            y = (1 - self.ls) * true + self.ls / K
        ce = -(y * logp).sum(dim=1)
        pt = (y * p).sum(dim=1).clamp_min(1e-6)
        focal = (1 - pt).pow(self.gamma) * ce
        return focal.mean()

def fit(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    root = args.data_root
    files = list_npz(root)
    if len(files) == 0: raise RuntimeError(f'No npz files under {root}')
    if args.split_dir and os.path.isdir(args.split_dir):
        def load_or_none(p): 
            return load_split_from_txt(p) if os.path.isfile(p) else None
        tr = load_or_none(os.path.join(args.split_dir,'train.txt')) or []
        va = load_or_none(os.path.join(args.split_dir,'val.txt')) or []
        te = load_or_none(os.path.join(args.split_dir,'test.txt')) or []
        def fix(lst): return [f if os.path.isabs(f) else os.path.join(root, os.path.normpath(f)) for f in lst]
        train_files, val_files, test_files = fix(tr), fix(va), fix(te)
        if not train_files or not val_files:
            train_files, val_files, test_files = make_split(files)
    else:
        train_files, val_files, test_files = make_split(files)

    Xs=[]; ys=[]
    for f in train_files:
        X, y, _ = read_npz(f)
        if X.shape != (TARGET_T, FEAT_DIM):
            T,D = X.shape
            if T < TARGET_T:
                pad = np.zeros((TARGET_T - T, D), dtype=np.float32); X = np.concatenate([X, pad], axis=0)
            elif T > TARGET_T:
                X = X[:TARGET_T,:]
            if X.shape[1] < FEAT_DIM:
                pad = np.zeros((TARGET_T, FEAT_DIM - X.shape[1]), dtype=np.float32); X = np.concatenate([X, pad], axis=1)
            elif X.shape[1] > FEAT_DIM:
                X = X[:, :FEAT_DIM]
        Xs.append(X); ys.append(y)
    Xs = np.stack(Xs, axis=0); mu = Xs.mean(axis=(0,1)); std = Xs.std(axis=(0,1)); std[std<1e-6]=1.0

    ds_tr = ClipsDataset(train_files, mu, std, train=True,  aug_roll=1)
    ds_va = ClipsDataset(val_files,   mu, std, train=False, aug_roll=0)
    ds_te = ClipsDataset(test_files,  mu, std, train=False, aug_roll=0)

    from collections import Counter
    y_counts = Counter(ys); class_count = np.array([y_counts.get(i,1) for i in range(NUM_CLASSES)], dtype=np.float32)
    class_weight = 1.0 / class_count
    sample_weights = []
    for f in train_files:
        _, y, _ = read_npz(f); sample_weights.append(class_weight[y])
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(ds_tr, batch_size=args.batch, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader  = DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=0)

    model = TCN(in_feat=FEAT_DIM, num_classes=NUM_CLASSES, p_drop=0.2).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=5, verbose=True)
    criterion = FocalLoss(gamma=1.5, label_smoothing=0.05, reduction='mean')

    best_f1 = -1.0
    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, 'best.pt')

    for ep in range(1, args.epochs+1):
        model.train(); tr_loss = 0.0
        for X, y in train_loader:
            X = X.to(device); y = y.to(device)
            logits = model(X); loss = criterion(logits, y)
            optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); optim.step()
            tr_loss += loss.item() * X.size(0)
        tr_loss /= len(ds_tr)

        model.eval(); preds=[]; gts=[]; va_loss=0.0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device); y = y.to(device)
                logits = model(X); loss = criterion(logits, y)
                va_loss += loss.item() * X.size(0)
                prob = F.softmax(logits, dim=1); pred = prob.argmax(dim=1)
                preds.append(pred.cpu().numpy()); gts.append(y.cpu().numpy())
        va_loss /= max(1, len(ds_va))
        if preds:
            preds = np.concatenate(preds); gts = np.concatenate(gts)
            f1 = f1_score(gts, preds, average='macro', zero_division=0)
        else:
            f1 = 0.0
        sched.step(f1)
        print(f"[EP {ep:03d}] train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_macroF1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            ckpt = {
                'model': model.state_dict(),
                'classes': CLASSES,
                'zscore_mu': mu.tolist(),
                'zscore_std': std.tolist(),
                'feat_dim': FEAT_DIM,
                'target_T': TARGET_T,
                'input_name': INPUT_NAME,
                'output_name': OUTPUT_NAME,
                'seed': SEED,
                'peak_pos': 'asymmetric(16|1|8)',
                'pre': PRE_FR,
                'post': POST_FR,
                'causal': False,
                'include_peak': True
            }
            torch.save(ckpt, best_path)
            print(f"  -> best updated: {best_path} (macroF1={best_f1:.4f})")

    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model'])
    model.eval(); preds=[]; gts=[]
    with torch.no_grad():
        for X, y in DataLoader(ds_te, batch_size=args.batch, shuffle=False):
            X = X.to(device); y = y.to(device)
            logits = model(X); pred = logits.argmax(dim=1)
            preds.append(pred.cpu().numpy()); gts.append(y.cpu().numpy())
    if preds:
        preds = np.concatenate(preds); gts = np.concatenate(gts)
        print("\n=== Classification Report (TEST) ===")
        print(classification_report(gts, preds, target_names=CLASSES, digits=3, zero_division=0))
        print("=== Confusion Matrix (TEST) ===")
        print(confusion_matrix(gts, preds, labels=list(range(NUM_CLASSES))))
    else:
        print("\n[WARN] No test samples — skipping report.")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default='dataset/clips/raw')
    ap.add_argument('--split_dir', type=str, default='dataset/splits/fixed')
    ap.add_argument('--out_dir', type=str, default='.')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--cpu', action='store_true')
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    fit(args)
