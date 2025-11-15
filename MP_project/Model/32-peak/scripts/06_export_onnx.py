# scripts/06_export_onnx.py  (peak-included regime; follows ckpt meta target_T=33)
# - best.pt → tcn.onnx + tcn_meta.json
# - Reads target_T/feat_dim/classes/z-score from checkpoint meta (saved in 05).
# - If meta missing, uses safe fallbacks (target_T=33 to match “32 before + peak(=1)”).
# - ONNX I/O tensor names come from meta (clips/logits by default).

import os, json, argparse, torch
import torch.nn as nn

# ===== Fallbacks (used only if ckpt meta is missing) =====
FEAT_DIM_DEFAULT = 16
NUM_CLASSES_FALLBACK = 6
TARGET_T_DEFAULT = 33         # 32 before + the peak frame
INPUT_NAME_DEFAULT = 'clips'
OUTPUT_NAME_DEFAULT = 'logits'

# ===== TCN (must mirror 05_train_tcn.py) =====
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, d, p_drop=0.2):
        super().__init__()
        pad = (k - 1) * d
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=d, padding=pad)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, dilation=d, padding=pad)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(p_drop)
        self.down  = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C, T)
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out); out = self.drop(out)
        out = self.conv2(out); out = self.bn2(out); out = self.drop(out)
        # causal crop: cut right padding so output length == input length
        Tlen = x.shape[-1]
        out = out[..., -Tlen:]
        res = self.down(x); res = res[..., -Tlen:]
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, in_feat=FEAT_DIM_DEFAULT, num_classes=NUM_CLASSES_FALLBACK, p_drop=0.0):
        super().__init__()
        chs = [64, 128, 128]
        ks  = 3
        self.in_proj = nn.Conv1d(in_feat, chs[0], kernel_size=1)
        blocks = []
        dils = [1, 2, 4, 8]
        c_in = chs[0]
        for d in dils:
            blocks.append(TemporalBlock(c_in, chs[1], ks, d, p_drop))
            c_in = chs[1]
        blocks.append(TemporalBlock(c_in, chs[2], ks, 1, p_drop))
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(chs[2], num_classes)

    def forward(self, x):
        # x: (B, T, D) → (B, D, T)
        x = x.permute(0, 2, 1)
        x = self.in_proj(x)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)

# ===== Checkpoint I/O =====
def load_checkpoint(ckpt_path: str):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu")  # weights_only=False (meta 포함)
    if isinstance(obj, dict) and 'model' in obj:
        return obj
    elif isinstance(obj, dict):
        return {'model': obj}
    raise TypeError("unsupported checkpoint format")

def extract_meta(ckpt: dict):
    classes  = ckpt.get('classes', ['Clear','Drive','Drop','Under','Hairpin','Idle'])
    if not isinstance(classes, (list, tuple)) or len(classes) == 0:
        classes = ['Clear','Drive','Drop','Under','Hairpin','Idle']

    feat_dim = int(ckpt.get('feat_dim', FEAT_DIM_DEFAULT))
    target_T = int(ckpt.get('target_T', TARGET_T_DEFAULT))

    in_name  = ckpt.get('input_name',  INPUT_NAME_DEFAULT)
    out_name = ckpt.get('output_name', OUTPUT_NAME_DEFAULT)

    # z-score stats (prefer zscore_mu/std; fallback mu/std; final fallback unit)
    z_mu  = ckpt.get('zscore_mu', ckpt.get('mu',  [0.0]*feat_dim))
    z_std = ckpt.get('zscore_std', ckpt.get('std', [1.0]*feat_dim))
    if z_mu is None or z_std is None:
        z_mu, z_std = [0.0]*feat_dim, [1.0]*feat_dim

    # pass-through extra flags if present
    extra = {}
    for k in ['peak_pos','causal','include_peak','seed']:
        if k in ckpt:
            extra[k] = ckpt[k]

    meta = {
        'classes': classes,
        'feat_dim': feat_dim,
        'target_T': target_T,
        'input_name': in_name,
        'output_name': out_name,
        'zscore_mu': z_mu,
        'zscore_std': z_std,
    }
    meta.update(extra)
    return meta

# ===== Main =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, default='best.pt')
    ap.add_argument('--onnx', type=str, default='tcn.onnx')
    ap.add_argument('--meta', type=str, default='tcn_meta.json')
    ap.add_argument('--opset', type=int, default=13)
    # Optional overrides (use only if your ckpt meta is incorrect)
    ap.add_argument('--force_target_T', type=int, default=None)
    ap.add_argument('--force_feat_dim', type=int, default=None)
    args = ap.parse_args()

    ckpt = load_checkpoint(args.ckpt)
    meta = extract_meta(ckpt)

    if args.force_target_T is not None:
        meta['target_T'] = int(args.force_target_T)
    if args.force_feat_dim is not None:
        meta['feat_dim'] = int(args.force_feat_dim)

    classes  = meta['classes']
    feat_dim = int(meta['feat_dim'])
    target_T = int(meta['target_T'])
    in_name  = meta['input_name']
    out_name = meta['output_name']

    # Build model and load weights
    model = TCN(in_feat=feat_dim, num_classes=len(classes), p_drop=0.0)
    state = ckpt['model']
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:   print(f"[WARN] missing keys: {missing}")
    if unexpected:print(f"[WARN] unexpected keys: {unexpected}")
    model.eval()

    # Static-length export (B=1, T=meta.target_T, D=meta.feat_dim)
    dummy = torch.zeros(1, target_T, feat_dim, dtype=torch.float32)

    torch.onnx.export(
        model, dummy, args.onnx,
        opset_version=args.opset,
        input_names=[in_name],
        output_names=[out_name],
        dynamic_axes=None,              # fixed T for runtime simplicity
        do_constant_folding=True
    )
    print(f"[INFO] ONNX saved: {args.onnx}")

    with open(args.meta, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[INFO] META saved: {args.meta}")
    print(f"[INFO] classes={classes}")
    print(f"[INFO] feat_dim={feat_dim}, target_T={target_T}, input='{in_name}', output='{out_name}'")

if __name__ == '__main__':
    main()
