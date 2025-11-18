# scripts/06_export_onnx.py
# - best.pt → tcn.onnx + tcn_meta.json 내보내기
# - 체크포인트 호환 처리:
#     · zscore_mu/std 또는 mu/std 키 모두 지원
#     · classes/feat_dim/target_T 누락 시 안전한 기본값으로 보완
# - ONNX I/O 텐서명: INPUT_NAME('clips'), OUTPUT_NAME('logits')
# - ★ TEST 성능지표(classification_report + confusion_matrix)도 meta.json에 저장

import os, json, argparse, torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix  # ★ 추가

# ===== 기본값 =====
FEAT_DIM_DEFAULT = 16
NUM_CLASSES_FALLBACK = 6
TARGET_T_DEFAULT = 33
INPUT_NAME_DEFAULT = 'clips'
OUTPUT_NAME_DEFAULT = 'logits'


# ===== TCN 구조 (05_train_tcn.py와 동일) =====
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


# ===== 체크포인트 로더 =====
def load_checkpoint(ckpt_path: str):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and 'model' in obj:
        ckpt = obj
    elif isinstance(obj, dict):
        ckpt = {'model': obj}
    else:
        raise TypeError("unsupported checkpoint format")
    return ckpt


# ===== 메타 추출 (여기서는 학습 메타만 읽어옴) =====
def extract_meta(ckpt: dict):
    classes = ckpt.get('classes', ['Clear','Drive','Drop','Under','Hairpin','Idle'])
    feat_dim = int(ckpt.get('feat_dim', FEAT_DIM_DEFAULT))
    target_T = int(ckpt.get('target_T', TARGET_T_DEFAULT))
    in_name  = ckpt.get('input_name', INPUT_NAME_DEFAULT)
    out_name = ckpt.get('output_name', OUTPUT_NAME_DEFAULT)

    z_mu  = ckpt.get('zscore_mu', ckpt.get('mu', [0.0] * feat_dim))
    z_std = ckpt.get('zscore_std', ckpt.get('std', [1.0] * feat_dim))

    frames_before = ckpt.get('frames_before_peak', None)
    frames_after  = ckpt.get('frames_after_peak', None)
    peak_centered = ckpt.get('peak_centered', None)

    best_f1       = ckpt.get('best_val_macroF1', None)
    best_epoch    = ckpt.get('best_epoch', None)
    best_val_loss = ckpt.get('best_val_loss', None)
    train_loss_at_best = ckpt.get('train_loss_at_best', None)

    num_train = ckpt.get('num_train_samples', None)
    num_val   = ckpt.get('num_val_samples', None)
    num_test  = ckpt.get('num_test_samples', None)

    return {
        'classes': classes,
        'feat_dim': feat_dim,
        'target_T': target_T,
        'input_name': in_name,
        'output_name': out_name,
        'zscore_mu': z_mu,
        'zscore_std': z_std,

        'frames_before_peak': frames_before,
        'frames_after_peak': frames_after,
        'peak_centered': peak_centered,

        'best_val_macroF1': best_f1,
        'best_val_loss': best_val_loss,
        'train_loss_at_best': train_loss_at_best,
        'best_epoch': best_epoch,

        'num_train_samples': num_train,
        'num_val_samples': num_val,
        'num_test_samples': num_test,
    }


# ===== 메인 =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, default='best.pt')
    ap.add_argument('--onnx', type=str, default='tcn.onnx')
    ap.add_argument('--meta', type=str, default='tcn_meta.json')
    ap.add_argument('--opset', type=int, default=13)
    ap.add_argument('--data_root', type=str, default='./dataset')   # ★ test.npz 찾기 위해 추가
    args = ap.parse_args()

    ckpt = load_checkpoint(args.ckpt)
    meta = extract_meta(ckpt)

    classes   = meta['classes']
    feat_dim  = int(meta['feat_dim'])
    target_T  = int(meta['target_T'])
    in_name   = meta['input_name']
    out_name  = meta['output_name']

    # === 모델 구성 ===
    model = TCN(in_feat=feat_dim, num_classes=len(classes), p_drop=0.0)
    state = ckpt['model']
    model.load_state_dict(state, strict=False)
    model.eval()

    # === TEST 지표 계산 (meta.json용) === ★★ 핵심 추가 부분 ★★
    test_npz = os.path.join(args.data_root, 'test.npz')
    if os.path.isfile(test_npz):
        te = np.load(test_npz, allow_pickle=True)
        x_te, y_te = te['x'], te['y']

        # zscore 적용
        z_mu, z_std = np.array(meta['zscore_mu']), np.array(meta['zscore_std'])
        x_te = (x_te - z_mu) / (z_std + 1e-9)

        with torch.no_grad():
            logits = model(torch.from_numpy(x_te).float())
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        report = classification_report(y_te, preds, digits=3, output_dict=True)
        cm = confusion_matrix(y_te, preds).tolist()

        meta['test_classification_report'] = report
        meta['test_confusion_matrix'] = cm
        meta['test_accuracy'] = report['accuracy']
        meta['test_macro_f1'] = report['macro avg']['f1-score']
    else:
        print("[WARN] test.npz not found, skipping test metrics")

    # === ONNX 내보내기 ===
    dummy = torch.zeros(1, target_T, feat_dim, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, args.onnx,
        opset_version=args.opset,
        input_names=[in_name],
        output_names=[out_name],
        dynamic_axes=None,
        do_constant_folding=True
    )
    print(f"[INFO] ONNX saved: {args.onnx}")

    # === META 저장 ===
    with open(args.meta, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[INFO] META saved: {args.meta}")


if __name__ == '__main__':
    main()
