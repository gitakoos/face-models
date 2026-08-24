# face-models

On-device face-grouping models for [Photos for Proton](https://github.com/gitakoos/proton-photos), an unofficial open-source Proton Drive Photos client for Android.

The app groups faces entirely on the device, opt-in and off by default. Nothing about a face leaves the phone: detection, embedding and clustering all run locally, and only the two model files below are downloaded (once, over the network the app already uses).

## Models

Both are the original OpenCV Zoo releases, redistributed here so the app can fetch them; they are not modified.

| File | Purpose | Source | License |
|---|---|---|---|
| `yunet.onnx` | Face detection (boxes + 5 landmarks) | OpenCV Zoo, `face_detection_yunet_2023mar` | MIT |
| `sface.onnx` | Face recognition (128-d embedding) | OpenCV Zoo, `face_recognition_sface_2021dec` | Apache-2.0 |

### Integrity

The app verifies each download against these before use.

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `yunet.onnx` | 232589 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `sface.onnx` | 38696353 | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |

## Download

The model files are published as assets on the [`v1` release](https://github.com/gitakoos/face-models/releases/tag/v1), not committed to the tree. Verify a file after downloading:

```bash
sha256sum yunet.onnx sface.onnx
```

## Provenance and reproduction

These files come straight from OpenCV Zoo. No conversion is done here; they are already ONNX and are downloaded as-is. `SOURCE.md` records the exact upstream paths and the SHA-256 values, so anyone can fetch the originals and confirm they match byte for byte.

## Reuse

The models keep their upstream licenses, whose full texts are in `licenses/` (`yunet-LICENSE.txt`, MIT; `sface-LICENSE.txt`, Apache-2.0) and whose attribution is in `NOTICE`. The packaging and documentation in this repository are Apache-2.0 (`LICENSE`). If you reuse any of this, keep `LICENSE`, `NOTICE` and `licenses/` intact.
