"""
Reproducible GhostFaceNet .h5 -> ONNX export for Photos for Proton (face embedder).

Source model: HamadYA/GhostFaceNets (MIT), GhostFaceNetV1 W1.3 basic_model .h5.
Output: [1, 512] embedding for a 112x112x3 NHWC input normalized (x-127.5)/128,
        exactly what FaceEmbedder.kt feeds (the app L2-normalizes the 512-d output itself).

Run from inside the cloned GhostFaceNets repo dir (so any custom layers import):
    python convert_ghostfacenet.py <path.h5> ghostfacenet.onnx
"""
import os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import tf2onnx
import onnxruntime

h5, out = sys.argv[1], sys.argv[2]

model = tf.keras.models.load_model(h5, compile=False)
print("keras input_shape:", model.input_shape, "output_shape:", model.output_shape)

# Fixed batch-1 112x112x3 NHWC input, matching FaceEmbedder.kt.
spec = (tf.TensorSpec((1, 112, 112, 3), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(
    model, input_signature=spec, opset=13, output_path=out
)

sess = onnxruntime.InferenceSession(out, providers=["CPUExecutionProvider"])
i, o = sess.get_inputs()[0], sess.get_outputs()[0]
r = sess.run(None, {i.name: np.zeros((1, 112, 112, 3), np.float32)})[0]
print("onnx input:", i.name, i.shape)
print("onnx output:", o.name, r.shape)
print("512-d:", r.shape[-1] == 512)
