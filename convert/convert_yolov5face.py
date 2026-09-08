"""
Reproducible YOLOv5s-face -> ONNX export for Photos for Proton (face detector).

Source model: deepcam-cn/yolov5-face (GPL-3.0), yolov5s-face weights.
Output: a single decoded tensor [1, N, 16] per detection row
        [cx, cy, w, h, obj, lm1x,lm1y, lm2x,lm2y, lm3x,lm3y, lm4x,lm4y, lm5x,lm5y, cls]
        in 640x640 letterbox pixel space, exactly what FaceDetector.kt decodes.

Run from inside the cloned yolov5-face repo dir:
    python convert_yolov5face.py weights/yolov5s-face.pt yolov5face.onnx
"""
import sys
import torch
import onnx
import onnxruntime
import numpy as np

weights, out = sys.argv[1], sys.argv[2]

ckpt = torch.load(weights, map_location="cpu", weights_only=False)
model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
model = model.float().eval()

# Turn off inplace ops for a clean trace, and switch the Detect head to the
# concatenated-decoded ONNX output path (grids/anchors rebuilt for 640).
for m in model.modules():
    if hasattr(m, "inplace"):
        m.inplace = False
det = model.model[-1]
det.export_cat = True
# As the repo's own export.py does: make anchor_grid a plain list so the export path's
# _make_grid_new can reassign correctly-shaped per-layer grids (it recomputes them from the
# intact `anchors` buffer; the fixed-shape registered buffer would otherwise reject the write).
delattr(det, "anchor_grid")
det.anchor_grid = [torch.zeros(1)] * det.nl

img = torch.zeros(1, 3, 640, 640)
with torch.no_grad():
    y = model(img)
    y0 = y[0] if isinstance(y, (list, tuple)) else y
print("torch output shape:", tuple(y0.shape))

torch.onnx.export(
    model, img, out,
    opset_version=12,
    input_names=["input"],
    output_names=["output"],
    do_constant_folding=True,
    dynamo=False,
)

onnx.checker.check_model(onnx.load(out))
sess = onnxruntime.InferenceSession(out, providers=["CPUExecutionProvider"])
o = sess.run(None, {sess.get_inputs()[0].name: img.numpy().astype(np.float32)})[0]
print("onnx input name:", sess.get_inputs()[0].name, "shape", sess.get_inputs()[0].shape)
print("onnx output name:", sess.get_outputs()[0].name, "shape", o.shape)
print("row stride == 16:", o.shape[-1] == 16)
print("num outputs:", len(sess.get_outputs()))
