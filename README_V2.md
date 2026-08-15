# UAV Detection — RetinaNet V2

A modernized UAV/drone detection pipeline built around PyTorch RetinaNet.

## V2 improvements
- Maintained torchvision RetinaNet ResNet-50 FPN V2 backbone
- Pretrained weights for transfer learning
- Pascal VOC/XML dataset loader
- UAV-oriented augmentation: flip, brightness/contrast and blur
- Automatic image scaling
- AdamW + cosine learning-rate schedule
- CUDA mixed precision when available
- Gradient clipping
- Best-checkpoint saving
- Clean Python package structure instead of a notebook-only workflow

## Dataset
Place Pascal VOC-style image/XML pairs in `dataset/`. The loader also tries common image extensions when an XML filename does not resolve directly.

## Train
```bash
pip install -r requirements-v2.txt
python train_v2.py --data dataset --epochs 30 --batch-size 4
```

The original notebook remains untouched. V2 lives alongside it so results can be compared safely.

## Next engineering loop
1. Establish baseline mAP/precision/recall.
2. Audit annotation quality and class distribution.
3. Tune anchors/resolution for small UAVs.
4. Add hard-example mining and stronger geometric augmentation.
5. Benchmark against a modern real-time detector.
6. Keep only changes that improve the held-out test set.
