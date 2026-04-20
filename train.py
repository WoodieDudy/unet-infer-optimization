import argparse
import yaml

import torch
import torch.nn as nn
import lightning as L
import segmentation_models_pytorch as smp
import torchvision.transforms as T
from torchvision.datasets import OxfordIIITPet
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np

DEVICE = "cuda"


class SegmentationModule(L.LightningModule):
    def __init__(self, model, lr=1e-4, num_classes=3):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, masks)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        
        outputs = self.model(images) # (B, Classes, H, W)
        val_loss = self.loss_fn(outputs, masks)
        
        pred_masks = outputs.argmax(dim=1) # (B, H, W)
        
        # (B, 1, H, W) -> (B, H, W)
        masks = masks.squeeze(1).long() 
        
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_masks, 
            masks, 
            mode='multiclass', 
            num_classes=self.num_classes
        )
        
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        self.log("val_iou", iou, prog_bar=True, on_epoch=True)
        self.log("val_loss", val_loss, on_epoch=True)
        
        return val_loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def to_indices(x):
    # Превращаем PIL Image в тензор и сдвигаем 1,2,3 -> 0,1,2
    mask = torch.from_numpy(np.array(x)).long()
    return mask - 1


def get_loaders(input_size, batch_size):
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
    ])
    target_transform = T.Compose([
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.NEAREST),
        to_indices,
    ])

    train_set = OxfordIIITPet(
        root="./data", 
        split="trainval", 
        target_types="segmentation", 
        download=True, 
        transform=transform,
        target_transform=target_transform
    )

    test_set = OxfordIIITPet(
        root="./data", 
        split="test", 
        target_types="segmentation", 
        download=True, 
        transform=transform,
        target_transform=target_transform
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def print_env_info(input_shape):
    print("=" * 60)
    print("Environment")
    print("=" * 60)
    print(f"  PyTorch        : {torch.__version__}")
    print(f"  CUDA (runtime) : {torch.version.cuda}")
    print(f"  cuDNN          : {torch.backends.cudnn.version()}")
    print(f"  GPU            : {torch.cuda.get_device_name(0)}")
    print(f"  Input shape    : {tuple(input_shape)}")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    
    input_shape = (config["data"]["batch_size"], 3, config["data"]["input_size"], config["data"]["input_size"])
    print_env_info(input_shape)

    train_loader, test_loader = get_loaders(config["data"]["input_size"], config["data"]["batch_size"])

    model = smp.Unet(**config["model"])
    model = model.to(DEVICE)
    model = SegmentationModule(model, lr=config["lr"], num_classes=config["model"]["classes"])
    
    checkpoint_callback = ModelCheckpoint(**config["model_checkpoint"])
    trainer = L.Trainer(
        default_root_dir=config["exp_dir"],
        callbacks=[checkpoint_callback],
        **config["trainer"]
    )
    trainer.fit(model, train_loader, val_dataloaders=test_loader)


if __name__ == "__main__":
    main()
