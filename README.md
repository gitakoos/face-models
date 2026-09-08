# face-models

On-device face-grouping models for [Photos for Proton](https://github.com/gitakoos/proton-photos), an unofficial open-source Proton Drive Photos client for Android.

The app groups faces entirely on the device, opt-in and off by default. Nothing about a face leaves the phone: detection, embedding and clustering all run locally, and only the model files below are downloaded (once, over the network the app already uses).

## Current models (v2)

Face recognition runs a two-stage pipeline: a detector finds where the faces are, and an embedder turns each aligned face into a vector the app compares locally.

| File | Purpose | Upstream | License |
|---|---|---|---|
| `yolov5face.onnx` | Face detection (boxes + 5 landmarks) | YOLOv5-face, [deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face) | GPL-3.0 |
| `ghostfacenet.onnx` | Face recognition (512-d embedding) | GhostFaceNetV1 W1.3, [HamadYA/GhostFaceNets](https://github.com/HamadYA/GhostFaceNets) | MIT |

Unlike the previous set, these two files are ONNX exports produced from the upstream published weights, not upstream ONNX files. `SOURCE.md` records the exact weights and the reproducible export steps, and the export scripts are in `convert/`.

### Integrity

The app verifies each download against these before use.

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `yolov5face.onnx` | 32370314 | `8ece145c7a956ed276250778bdb89e40c8ac9521c8669b14d79718cc83ccab32` |
| `ghostfacenet.onnx` | 16190333 | `ffd8203a0c9e93d90a4957e24d173a883a5a04d64d23c771742c66273518a0db` |

Published as assets on the [`v2` release](https://github.com/gitakoos/face-models/releases/tag/v2), not committed to the tree.

## Previous models (v1)

Earlier app versions use two unmodified OpenCV Zoo models, still published on the [`v1` release](https://github.com/gitakoos/face-models/releases/tag/v1).

| File | Purpose | Upstream | License |
|---|---|---|---|
| `yunet.onnx` | Face detection (boxes + 5 landmarks) | OpenCV Zoo, `face_detection_yunet_2023mar` | MIT |
| `sface.onnx` | Face recognition (128-d embedding) | OpenCV Zoo, `face_recognition_sface_2021dec` | Apache-2.0 |

`yunet.onnx` 232589 bytes, `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`; `sface.onnx` 38696353 bytes, `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`.

## Provenance and reproduction

`SOURCE.md` records, for every model, the exact upstream source and how to reproduce or verify it. The v2 files are exported from the upstream weights with the scripts in `convert/`; the v1 files are unmodified OpenCV Zoo blobs, fetched as-is.

## Reuse

Each model keeps its upstream license, with full texts in `licenses/` and attribution in `NOTICE`:

- `yolov5face.onnx`: GPL-3.0 (`licenses/yolov5face-LICENSE.txt`)
- `ghostfacenet.onnx`: MIT (`licenses/ghostfacenet-LICENSE.txt`)
- `yunet.onnx`: MIT (`licenses/yunet-LICENSE.txt`)
- `sface.onnx`: Apache-2.0 (`licenses/sface-LICENSE.txt`)

The packaging and documentation in this repository are Apache-2.0 (`LICENSE`). If you reuse any of this, keep `LICENSE`, `NOTICE` and `licenses/` intact.
