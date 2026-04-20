import argparse
import yaml
import torch
import lightning as L
import segmentation_models_pytorch as smp
from train import SegmentationModule, get_loaders 

def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best-unet.ckpt")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    _, test_loader = get_loaders(
        config["data"]["input_size"], 
        config["data"]["batch_size"]
    )

    model = SegmentationModule.load_from_checkpoint(
        checkpoint_path=args.checkpoint,
        model=smp.Unet(
            config["model"]["encoder_name"], 
            classes=config["model"]["classes"],
            encoder_weights=None # Веса все равно заменятся из чекпоинта
        )
    )
    model.eval()

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=False
    )

    print("\n" + "="*30)
    print("RUNNING EVALUATION ON TEST SET")
    print("="*30)
    
    results = trainer.validate(model, dataloaders=test_loader)
    
    print("\nFinal Metrics:")
    for key, value in results[0].items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    run_evaluation()
