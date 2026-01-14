import os
import time
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from dist_utils import setup_distributed, cleanup_distributed, is_main, broadcast_model, barrier
from ema import EMA
from unet import TinyUNet
from ddpm import DDPM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/base.yaml")
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open(args.cfg, "r"))
    rank, world_size = setup_distributed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = cfg["out_dir"]
    if is_main():
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "ckpt"), exist_ok=True)
    barrier()

    # --- Data ---
    tfm = transforms.Compose([
        transforms.Resize(cfg["img_size"]),
        transforms.RandomResizedCrop(cfg["img_size"], scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = datasets.ImageFolder(cfg["data_dir"], transform=tfm)
    num_classes = len(ds.classes)

    sampler = torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=world_size, rank=rank, shuffle=True
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=cfg["batch_size"], sampler=sampler,
        num_workers=cfg["num_workers"], drop_last=True, pin_memory=True
    )

    # --- Model ---
    model = TinyUNet(num_classes=num_classes, base_ch=cfg["base_ch"]).to(device)
    ddpm = DDPM(timesteps=cfg["timesteps"], device=device)

    # Wrap in DDP
    model = DDP(model, device_ids=[torch.cuda.current_device()] if device == "cuda" else None)

    # EMA on rank0 canonical; everyone keeps a copy but we periodically broadcast from rank0
    ema = EMA(model.module, decay=cfg["ema_decay"])

    opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and cfg.get("amp", False)))

    global_step = 0
    for epoch in range(1, cfg["epochs"] + 1):
        sampler.set_epoch(epoch)
        model.train()

        pbar = tqdm(dl, disable=not is_main(), desc=f"Epoch {epoch}/{cfg['epochs']}")
        start = time.time()

        for x0, y in pbar:
            x0, y = x0.to(device), y.to(device)
            b = x0.size(0)
            t = torch.randint(0, cfg["timesteps"], (b,), device=device).long()

            with torch.cuda.amp.autocast(enabled=(device == "cuda" and cfg.get("amp", False))):
                xt, noise = ddpm.q_sample(x0, t)
                pred = model(xt, t, y)
                loss = loss_fn(pred, noise)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            # Update EMA locally from the current model.module
            ema.update(model.module)

            global_step += 1

            if is_main():
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Periodic EMA sync: broadcast EMA weights from rank0 to others
            if cfg.get("ema_sync_every", 0) > 0 and (global_step % cfg["ema_sync_every"] == 0):
                broadcast_model(ema.ema_model, src=0)

        # Save + sample on rank0
        if is_main() and (epoch % cfg.get("save_every", 5) == 0 or epoch == 1):
            # quick class-conditioned sampling (small)
            n_per_class = cfg.get("sample_n_per_class", 8)
            grids = []
            for cls in range(min(num_classes, cfg.get("sample_classes_cap", 8))):
                y_cls = torch.full((n_per_class,), cls, device=device, dtype=torch.long)
                samples = ddpm.sample(
                    ema.ema_model, n=n_per_class, img_size=cfg["img_size"], 
                    y=y_cls, seed=cls+123
                ).cpu()
                grids.append(samples)
            grid = utils.make_grid(
                torch.cat(grids, 0), nrow=n_per_class, 
                normalize=True, value_range=(-1, 1)
            )
            utils.save_image(
                grid, 
                os.path.join(out_dir, "samples", f"samples_e{epoch:03d}.png")
            )

            ckpt = {
                "model": model.module.state_dict(),
                "ema_model": ema.ema_model.state_dict(),
                "classes": ds.classes,
                "cfg": cfg,
                "epoch": epoch,
                "global_step": global_step,
            }
            torch.save(
                ckpt, 
                os.path.join(out_dir, "ckpt", f"ddp_e{epoch:03d}.pt")
            )
            print(f"[rank0] saved epoch {epoch}")

    cleanup_distributed()

if __name__ == "__main__":
    main()
