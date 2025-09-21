# fix_split_for_barracuda.py
import onnx
from onnx import helper, checker, shape_inference

m = onnx.load("model.onnx")

# 가능하면 shape 추론 시도(분할 길이를 정확히 계산하는 데 도움)
try:
    m = shape_inference.infer_shapes(m)
except Exception:
    pass

def find_dim_size(value_info, name, axis):
    for vi in value_info:
        if vi.name == name:
            dims = vi.type.tensor_type.shape.dim
            if 0 <= axis < len(dims):
                d = dims[axis]
                if d.HasField("dim_value"):
                    return d.dim_value
    return None

changed = 0
for n in m.graph.node:
    if n.op_type != "Split":
        continue
    if any(a.name == "split" for a in n.attribute):
        continue

    # axis 가져오기(없으면 0)
    axis = 0
    for a in n.attribute:
        if a.name == "axis":
            axis = a.i

    k = len(n.output)  # 출력 개수
    # 입력 텐서 해당 축 길이를 알면 정확 분할, 모르면 균등 1로 분할
    dim = find_dim_size(m.graph.value_info, n.input[0], axis)
    if dim is None:
        dim = find_dim_size(m.graph.input, n.input[0], axis)

    split_list = [dim // k] * k if (dim and dim % k == 0) else [1] * k
    n.attribute.extend([helper.make_attribute("split", split_list)])
    changed += 1

checker.check_model(m)
onnx.save(m, "model_barracuda.onnx")
print(f"Patched {changed} Split node(s).")
