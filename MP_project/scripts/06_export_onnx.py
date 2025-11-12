# scripts/06_export_onnx.py
# - best.pt → tcn.onnx + tcn_meta.json 내보내기
# - 체크포인트 호환 처리:
#     · zscore_mu/std 또는 mu/std 키 모두 지원
#     · classes/feat_dim/target_T 누락 시 안전한 기본값으로 보완
# - ONNX I/O 텐서명: INPUT_NAME('clips'), OUTPUT_NAME('logits')

import os, json, argparse, torch
import torch.nn as nn

# ===== TCN 구조 (05_train_tcn.py와 동일) =====
FEAT_DIM_DEFAULT = 16
NUM_CLASSES_FALLBACK = 6
TARGET_T_DEFAULT = 33
INPUT_NAME_DEFAULT = 'clips'
OUTPUT_NAME_DEFAULT = 'logits'

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
        T = x.shape[-1]
        out = out[..., -T:]
        res = self.down(x); res = res[..., -T:]
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
        # x: (B, T, D)
        x = x.permute(0, 2, 1)  # (B, D, T)
        x = self.in_proj(x)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        logits = self.head(x)
        return logits

# ===== 체크포인트 로더(호환) =====
def load_checkpoint(ckpt_path: str):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    # NOTE: PyTorch 2.4+는 weights_only=True 권장. 여기선 메타 포함이므로 False 유지.
    obj = torch.load(ckpt_path, map_location="cpu")  # weights_only=False (기본)
    if isinstance(obj, dict) and 'model' in obj:
        ckpt = obj
    elif isinstance(obj, dict):
        # 순수 state_dict만 저장된 경우 감싸기
        ckpt = {'model': obj}
    else:
        raise TypeError("unsupported checkpoint format")
    return ckpt

def extract_meta(ckpt: dict):
    # classes
    classes = ckpt.get('classes', ['Clear','Drive','Drop','Under','Hairpin','Idle'])
    if not isinstance(classes, (list, tuple)) or len(classes) == 0:
        classes = ['Clear','Drive','Drop','Under','Hairpin','Idle']

    # feat_dim / target_T
    feat_dim = int(ckpt.get('feat_dim', FEAT_DIM_DEFAULT))
    target_T = int(ckpt.get('target_T', TARGET_T_DEFAULT))

    # I/O names
    in_name  = ckpt.get('input_name',  INPUT_NAME_DEFAULT)
    out_name = ckpt.get('output_name', OUTPUT_NAME_DEFAULT)

    # z-score 통계: zscore_mu/std 우선, 없으면 mu/std, 최후에는 단위(zeros/ones)
    z_mu = ckpt.get('zscore_mu', ckpt.get('mu', None))
    z_std= ckpt.get('zscore_std', ckpt.get('std', None))
    if z_mu is None or z_std is None:
        # 안전 폴백
        z_mu  = [0.0]*feat_dim
        z_std = [1.0]*feat_dim

    return {
        'classes': classes,
        'feat_dim': feat_dim,
        'target_T': target_T,
        'input_name': in_name,
        'output_name': out_name,
        'zscore_mu': z_mu,
        'zscore_std': z_std,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, default='best.pt')
    ap.add_argument('--onnx', type=str, default='tcn.onnx')
    ap.add_argument('--meta', type=str, default='tcn_meta.json')
    ap.add_argument('--opset', type=int, default=13)
    args = ap.parse_args()

    ckpt = load_checkpoint(args.ckpt)
    meta = extract_meta(ckpt)

    feat_dim  = int(meta['feat_dim'])
    target_T  = int(meta['target_T'])
    classes   = meta['classes']
    in_name   = meta['input_name']
    out_name  = meta['output_name']

    # 모델 구성 & 가중치 로드
    num_classes = len(classes)
    model = TCN(in_feat=feat_dim, num_classes=num_classes, p_drop=0.0)
    state = ckpt['model']
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] missing keys in state_dict: {missing}")
    if unexpected:
        print(f"[WARN] unexpected keys in state_dict: {unexpected}")
    model.eval()

    # 더미 입력 (B=1, T, D)
    dummy = torch.zeros(1, target_T, feat_dim, dtype=torch.float32)

    # ONNX 내보내기
    torch.onnx.export(
        model, dummy, args.onnx,
        opset_version=args.opset,
        input_names=[in_name],
        output_names=[out_name],
        dynamic_axes=None,  # 고정 길이(T=33, D=feat_dim) 가정
        do_constant_folding=True
    )
    print(f"[INFO] ONNX saved: {args.onnx}")

    # 메타 JSON 저장
    with open(args.meta, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[INFO] META saved: {args.meta}")
    print(f"[INFO] classes: {classes}")
    print(f"[INFO] feat_dim={feat_dim}, target_T={target_T}, input='{in_name}', output='{out_name}'")

if __name__ == '__main__':
    main()
