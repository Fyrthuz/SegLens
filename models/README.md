# Trained checkpoints

The `*.pth` files here are gitignored. To reproduce:

```bash
make train-unet
```

The latest run on RTX 5080 (60 epochs, init_features=64, early stop at epoch 23) reached:

- best_val_dice: 0.6626
- test_dice: 0.6559
- test_iou:  0.5321

Lower than the TFM baseline (Dice 0.894) due to differences in
augmentation/scheduler/early-stop. Sufficient for the explainability
pipeline; the framework is what we are validating, not the SOTA
segmentation metric.
