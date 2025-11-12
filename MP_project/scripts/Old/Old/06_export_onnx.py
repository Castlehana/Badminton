# scripts/06_export_onnx.py
# - best.pt 체크포인트를 로드해 TCN을 재구성하고 ONNX로 내보냄
# - z-score 통계(mu/std)와 클래스 목록을 meta JSON으로 함께 저장
#
# 예)
#   python scripts/06_export_onnx.py
#   python scripts/06_export_onnx.py --ckpt best.pt --out exports\tcn.onnx --meta exports\tcn_meta.json --opset 17
#
# I/O:
#   입력  : clips  (float32, [B, T, D], 동적 B/T, 고정 D)
#   출력  : logits (float32, [B, C])

import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------- 모델 정의 -----------------------
class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, dil=1, p=0.2):
        super().__init__()
        pad = (k - 1) * dil
        self.pad1 = nn.ConstantPad2d((pad, 0, 0, 0), 0.0)
        self.c1 = nn.Conv2d(c_in, c_out, (1, k), dilation=(1, dil))
        self.b1 = nn.BatchNorm2d(c_out)
        self.pad2 = nn.ConstantPad2d((pad, 0, 0, 0), 0.0)
        self.c2 = nn.Conv2d(c_out, c_out, (1, k), dilation=(1, dil))
        self.b2 = nn.BatchNorm2d(c_out)
        self.drop = nn.Dropout(p)
        # ★ 학습 스크립트와 동일: 채널 동일이면 Identity, 다르면 1x1 Conv
        self.res = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        y = self.pad1(x); y = F.relu(self.b1(self.c1(y)))
        y = self.pad2(y); y = F.relu(self.b2(self.c2(y)))
        y = self.drop(y)
        return F.relu(y + self.res(x))

class TimeAvgPoolKeepD(nn.Module):
    """[B, C, D, T] -> 시간축 평균 -> [B, C, D, 1] (학습 head 구조와 호환되는 무파라미터 연산)"""
    def forward(self, x):
        return x.mean(dim=3, keepdim=True)

class TCN(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.b1 = TCNBlock(1,   64, k=3, dil=1, p=0.2)
        self.b2 = TCNBlock(64, 128, k=3, dil=2, p=0.2)
        self.b3 = TCNBlock(128,128, k=3, dil=4, p=0.2)
        # 학습 시 head.* 키 구조와 동일 인덱스 유지(0~5)
        self.head = nn.Sequential(
            TimeAvgPoolKeepD(),                 # head.0 (AdaptiveAvgPool2d((feat_dim,1))와 동일한 역할: T만 평균)
            nn.Flatten(1),                      # head.1
            nn.Linear(128 * feat_dim, 256),     # head.2
            nn.ReLU(),                          # head.3
            nn.Dropout(0.3),                    # head.4
            nn.Linear(256, num_classes)         # head.5
        )

    def forward(self, x):  # x: [B, T, D]
        x = x.transpose(1, 2).unsqueeze(1)   # [B, 1, D, T]
        x = self.b1(x); x = self.b2(x); x = self.b3(x)  # [B,128,D,T]
        return self.head(x)                  # [B,C]

# ------------------------------- 유틸 -------------------------------
def load_checkpoint(ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    # 참고: 경고 메시지 회피를 원하면 torch>=2.4에서 weights_only=True를 사용할 수 있음.
    obj = torch.load(ckpt_path, map_location="cpu")
    needed = ['model', 'mu', 'std', 'feat_dim', 'classes']
    missing = [k for k in needed if k not in obj]
    if missing:
        raise KeyError(f"checkpoint missing keys: {missing}")
    return obj

def save_meta(meta_path, mu, std, classes, target_T, feat_dim):
    meta = {
        "zscore_mu": np.asarray(mu, dtype=np.float32).tolist(),
        "zscore_std": np.asarray(std, dtype=np.float32).tolist(),
        "classes": list(classes),
        "target_T": int(target_T),
        "feat_dim": int(feat_dim),
        "input_name": "clips",
        "output_name": "logits",
        "note": "z-score는 (X - mu)/std 형태로 적용"
    }
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# ------------------------------- 메인 -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="best.pt", help="학습 가중치 체크포인트(.pt)")
    ap.add_argument("--out",  type=str, default="tcn.onnx", help="내보낼 ONNX 경로")
    ap.add_argument("--meta", type=str, default="tcn_meta.json", help="전처리/클래스 메타 JSON 경로")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset 버전(권장 17 이상)")
    ap.add_argument("--verify", dest="verify", action="store_true", help="onnxruntime 간이 검증 실행")
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    ap.set_defaults(verify=True)
    args = ap.parse_args()

    ckpt = load_checkpoint(args.ckpt)
    feat_dim = int(ckpt["feat_dim"])
    classes  = ckpt["classes"]
    num_classes = len(classes)
    target_T = 32  # 메타 기록용 (학습 파이프라인의 T)

    # 모델 로드 (학습과 동일 구조)
    model = TCN(feat_dim=feat_dim, num_classes=num_classes)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # 더미 입력(B=1, 동적 T 허용)
    B, T, D = 1, target_T, feat_dim
    example = torch.zeros((B, T, D), dtype=torch.float32)

    # 폴더 생성
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ONNX 내보내기
    input_names  = ["clips"]
    output_names = ["logits"]
    dynamic_axes = {
        "clips":  {0: "batch", 1: "time"},
        "logits": {0: "batch"}
    }
    torch.onnx.export(
        model, example, args.out,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=args.opset
    )
    print(f"[OK] ONNX saved -> {args.out}")

    # 메타 저장
    save_meta(args.meta, ckpt["mu"], ckpt["std"], classes, target_T=target_T, feat_dim=feat_dim)
    print(f"[OK] META saved -> {args.meta}")

    # 간단 검증
    if args.verify:
        try:
            import onnxruntime as ort
            with torch.no_grad():
                pt_logits = model(example).cpu().numpy()
            sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
            ort_logits = sess.run([output_names[0]], {input_names[0]: example.numpy()})[0]
            pt_pred  = int(np.argmax(pt_logits, axis=1)[0])
            ort_pred = int(np.argmax(ort_logits, axis=1)[0])
            print(f"[VERIFY] pytorch={pt_pred}  onnx={ort_pred}  -> {'OK' if pt_pred==ort_pred else 'MISMATCH'}")
        except Exception as e:
            print(f"[WARN] onnxruntime verify failed: {e}")

if __name__ == "__main__":
    main()
