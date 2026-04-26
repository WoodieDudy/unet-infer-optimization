"""Sparse-aware fine-tune через torch.nn.utils.prune (reparametrization).

На вход — обычный обученный dense чекпоинт. Скрипт сам:
1. Считает 2:4 mask: в каждой группе из 4 подряд весов (по input-channel оси
   для Conv2d/Linear) оставляет 2 с наибольшим |w|, остальные 2 → 0.
2. Применяет mask через `torch.nn.utils.prune.custom_from_mask` —
   reparametrization, маска применяется внутри forward-графа на каждом шаге,
   поэтому не зависит от GradScaler/AMP.
3. Запускает 3 эпохи fine-tune (lr=1e-5) — оставшиеся 50% весов
   компенсируют потерю от обнулённых.
4. После training сливает `weight_orig * weight_mask` обратно в `weight`
   (`prune.remove`) и сохраняет финальный Lightning-ckpt с физическими
   нулями внутри.
"""

import argparse

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

from train import SegmentationModule, get_loaders


def _mask_2to4_last_dim(t: torch.Tensor) -> torch.Tensor:
    *prefix, n = t.shape
    aligned = (n // 4) * 4
    mask = torch.ones_like(t)
    if aligned == 0:
        return mask
    g = t[..., :aligned].reshape(*prefix, aligned // 4, 4)
    _, idx = torch.topk(g.abs(), k=2, dim=-1)
    keep = torch.zeros_like(g).scatter_(-1, idx, 1.0)
    mask[..., :aligned] = keep.reshape(*prefix, aligned)
    return mask


def make_2to4_mask(weight: torch.Tensor) -> torch.Tensor:
    """2:4 mask вдоль input-channel оси.
    - Conv2d (out_c, in_c, kH, kW): группируем по in_c
    - Linear (out, in): группируем по in
    """
    if weight.ndim == 4:
        oc, ic, kh, kw = weight.shape
        flat = weight.permute(0, 2, 3, 1).contiguous().reshape(oc * kh * kw, ic)
        m = _mask_2to4_last_dim(flat).reshape(oc, kh, kw, ic).permute(0, 3, 1, 2).contiguous()
        return m
    if weight.ndim == 2:
        return _mask_2to4_last_dim(weight)
    return torch.ones_like(weight)


def apply_2to4_prune(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            mask = make_2to4_mask(m.weight.data).to(m.weight.dtype)
            prune.custom_from_mask(m, name="weight", mask=mask)


def remove_prune_reparam(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, "weight_orig"):
            prune.remove(m, "weight")


def count_zeros(model: nn.Module):
    z = t = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            z += int((m.weight == 0).sum())
            t += m.weight.numel()
    return z, t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--dense-ckpt", required=True,
                        help="Обученный dense Lightning-ckpt, из которого считаем 2:4 mask")
    args = parser.parse_args()

    with open(args.config_path) as f:
        cfg = yaml.safe_load(f)

    train_loader, test_loader = get_loaders(**cfg["data"])

    base = smp.Unet(**cfg["model"])
    module = SegmentationModule.load_from_checkpoint(
        checkpoint_path=args.dense_ckpt,
        model=base,
        lr=cfg["lr"],
        num_classes=cfg["model"]["classes"],
    )

    apply_2to4_prune(module.model)
    z, t = count_zeros(module.model)
    print(f"Init mask: {z}/{t} = {100*z/t:.2f}% zeros")

    ckpt_cb = ModelCheckpoint(**cfg["model_checkpoint"])
    trainer = L.Trainer(
        default_root_dir=cfg["exp_dir"],
        callbacks=[ckpt_cb],
        **cfg["trainer"],
    )
    trainer.validate(module, dataloaders=test_loader)
    trainer.fit(module, train_loader, val_dataloaders=test_loader)

    if ckpt_cb.best_model_path:
        print(f"Loading best: {ckpt_cb.best_model_path}")
        best = torch.load(ckpt_cb.best_model_path, map_location="cpu", weights_only=False)
        module.load_state_dict(best["state_dict"])

    remove_prune_reparam(module.model)
    z, t = count_zeros(module.model)
    print(f"After prune.remove: {z}/{t} = {100*z/t:.2f}% physical zeros")

    out_path = f"{cfg['exp_dir']}/trained_sparse_merged.ckpt"
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in module.state_dict().items()},
            "hyper_parameters": {"lr": cfg["lr"], "num_classes": cfg["model"]["classes"]},
            "pytorch-lightning_version": L.__version__,
            "epoch": 0,
            "global_step": 0,
        },
        out_path,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
