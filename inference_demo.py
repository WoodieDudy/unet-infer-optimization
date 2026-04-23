import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch

from train import SegmentationModule, get_loaders


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="experiments/inference_demo.png")
    p.add_argument("--num-images", type=int, default=4)
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda"

    module = SegmentationModule.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        model=smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=3,
        ),
    )
    module = module.to(device, dtype=torch.float16).eval()

    _, test_loader = get_loaders(
        input_size=args.input_size,
        batch_size=args.num_images,
        num_workers=0,
    )

    g = torch.Generator().manual_seed(args.seed)
    skip = int(torch.randint(0, max(1, len(test_loader) - 1), (1,), generator=g).item())
    it = iter(test_loader)
    for _ in range(skip):
        next(it)
    images, masks = next(it)
    images_dev = images.to(device, dtype=torch.float16)

    with torch.inference_mode():
        logits = module.model(images_dev)
        preds = logits.argmax(dim=1).cpu().numpy()

    images_np = images.numpy()
    masks_np = masks.numpy()
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0]

    cmap = plt.get_cmap("tab10", 3)

    fig, axes = plt.subplots(args.num_images, 3, figsize=(9, 3 * args.num_images))
    if args.num_images == 1:
        axes = np.array([axes])

    titles = ["Image", "Ground truth", "Prediction"]
    for i in range(args.num_images):
        axes[i, 0].imshow(images_np[i].transpose(1, 2, 0))
        axes[i, 1].imshow(masks_np[i], cmap=cmap, vmin=0, vmax=2)
        axes[i, 2].imshow(preds[i], cmap=cmap, vmin=0, vmax=2)
        for j, t in enumerate(titles):
            if i == 0:
                axes[i, j].set_title(t)
            axes[i, j].axis("off")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
