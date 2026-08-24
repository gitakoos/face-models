# Source and verification

The two model files are the original OpenCV Zoo releases, unmodified. This file
records where they come from and how to confirm the bytes match, so nothing has
to be taken on trust.

## Upstream

OpenCV Zoo: https://github.com/opencv/opencv_zoo (Apache-2.0 repository; each
model carries its own license, noted in `../NOTICE`).

| File here | Upstream file | Upstream directory |
|---|---|---|
| `yunet.onnx` | `face_detection_yunet_2023mar.onnx` | `models/face_detection_yunet` |
| `sface.onnx` | `face_recognition_sface_2021dec.onnx` | `models/face_recognition_sface` |

## Fetch the originals

The model blobs are stored with Git LFS, so use the media host (a plain raw URL
returns the LFS pointer, not the file):

```bash
curl -L -o yunet.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

curl -L -o sface.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

## Verify

```bash
sha256sum yunet.onnx sface.onnx
```

Expected:

```
8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  yunet.onnx
0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79  sface.onnx
```

Sizes: `yunet.onnx` 232589 bytes, `sface.onnx` 38696353 bytes.

## Model I/O (for reference)

- `yunet.onnx`: input `input` `[1,3,640,640]`; outputs `cls_{8,16,32}`,
  `obj_{8,16,32}`, `bbox_{8,16,32}`, `kps_{8,16,32}` per stride.
- `sface.onnx`: input `data` `[1,3,112,112]`; output `fc1` `[1,128]`.
