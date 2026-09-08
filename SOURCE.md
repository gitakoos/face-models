# Source and verification

Where every model comes from and how to reproduce or confirm it, so nothing has to be taken on trust.

## v2 models (current)

The two v2 files are ONNX exports produced from the upstream published weights. The export scripts (`convert_yolov5face.py`, `convert_ghostfacenet.py`) are in `convert/`.

### `yolov5face.onnx`

- Upstream: [deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face) (GPL-3.0), the `yolov5s-face` weights (`.pt`), from that repository's README download links.
- Export: clone the repo, then `python convert_yolov5face.py weights/yolov5s-face.pt yolov5face.onnx`. The script switches the detection head to the concatenated-decoded output and exports at 640x640, opset 12.
- I/O: input `input` `[1,3,640,640]` (RGB, NCHW, `/255`, letterboxed); one output `output` `[1,25200,16]`, each row `cx,cy,w,h, obj, 5x(x,y) landmarks, class`, in 640-letterbox pixels.

### `ghostfacenet.onnx`

- Upstream: [HamadYA/GhostFaceNets](https://github.com/HamadYA/GhostFaceNets) (MIT), the GhostFaceNetV1 W1.3 basic-model weights `GN_W1.3_S2_ArcFace_epoch48.h5`, from that repository's releases.
- Export: clone the repo, then `python convert_ghostfacenet.py GN_W1.3_S2_ArcFace_epoch48.h5 ghostfacenet.onnx` (Keras load plus tf2onnx, opset 13).
- I/O: input `input` `[1,112,112,3]` (RGB, NHWC, `(x-127.5)/128`); output `[1,512]` embedding (the app L2-normalizes it).

### Verify

```bash
sha256sum yolov5face.onnx ghostfacenet.onnx
```

Expected:

```
8ece145c7a956ed276250778bdb89e40c8ac9521c8669b14d79718cc83ccab32  yolov5face.onnx
ffd8203a0c9e93d90a4957e24d173a883a5a04d64d23c771742c66273518a0db  ghostfacenet.onnx
```

Sizes: `yolov5face.onnx` 32370314 bytes, `ghostfacenet.onnx` 16190333 bytes.

## v1 models (previous)

The two v1 files are the original OpenCV Zoo releases, unmodified.

Upstream: OpenCV Zoo, https://github.com/opencv/opencv_zoo (each model carries its own license, noted in `NOTICE`).

| File here | Upstream file | Upstream directory |
|---|---|---|
| `yunet.onnx` | `face_detection_yunet_2023mar.onnx` | `models/face_detection_yunet` |
| `sface.onnx` | `face_recognition_sface_2021dec.onnx` | `models/face_recognition_sface` |

The blobs are stored with Git LFS upstream, so use the media host (a plain raw URL returns the LFS pointer, not the file):

```bash
curl -L -o yunet.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

curl -L -o sface.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

Expected:

```
8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  yunet.onnx
0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79  sface.onnx
```

Sizes: `yunet.onnx` 232589 bytes, `sface.onnx` 38696353 bytes.

- `yunet.onnx`: input `input` `[1,3,640,640]`; outputs `cls_{8,16,32}`, `obj_{8,16,32}`, `bbox_{8,16,32}`, `kps_{8,16,32}` per stride.
- `sface.onnx`: input `data` `[1,3,112,112]`; output `fc1` `[1,128]`.
