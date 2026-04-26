import argparse
from pathlib import Path

import torch
import segmentation_models_pytorch as smp

from train import SegmentationModule


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--onnx", required=True)
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--encoder-name", default="resnet34")
    p.add_argument("--classes", type=int, default=3)
    p.add_argument("--in-channels", type=int, default=3)
    args = p.parse_args()

    module = SegmentationModule.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        model=smp.Unet(
            encoder_name=args.encoder_name,
            encoder_weights=None,
            in_channels=args.in_channels,
            classes=args.classes,
        ),
    )
    module.eval().cpu()

    dummy = torch.randn(1, args.in_channels, args.input_size, args.input_size)

    Path(args.onnx).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module.model,
        dummy,
        args.onnx,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"ONNX saved: {args.onnx}")


if __name__ == "__main__":
    main()
