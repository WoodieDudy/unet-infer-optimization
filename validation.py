import argparse
import yaml
import json
import torch
import lightning as L
import segmentation_models_pytorch as smp
from train import SegmentationModule, get_loaders
from pathlib import Path

DEVICE = torch.device("cuda")


def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    _, test_loader = get_loaders(**config["data"])

    all_results = {}
    for model_config in config["models"]:
        if model_config["model_path"].endswith("ckpt"):
            model = SegmentationModule.load_from_checkpoint(
                checkpoint_path=model_config["model_path"],
                model=smp.Unet(**model_config["model"])
            )
            model = model.to(DEVICE, dtype=torch.float16)
            if model_config["compile"]:
                model.model = torch.compile(model.model)
        elif model_config["model_path"].endswith(".engine"):
            from quantize import TRTModelWrapper
            trt = TRTModelWrapper(model_config["model_path"])
            model = SegmentationModule(trt)
        else:
            raise NotImplementedError("Неизвестный формат")
        
        model.eval()

        trainer = L.Trainer(
            default_root_dir=config["exp_dir"],
            accelerator="gpu",
            devices=1,
            logger=False,
            precision=16
        )
        
        results = trainer.validate(model, dataloaders=test_loader)
        all_results[model_config["model_name"]] = results
    
        print(f'\nMetrics {model_config["model_name"]}:')
        for key, value in results[0].items():
            print(f"{key}: {value:.4f}")

    exp_dir = Path(config["exp_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "validation_results.json", "w") as f:
        f.write(json.dumps(all_results))


if __name__ == "__main__":
    run_evaluation()
