import os
import yaml
import argparse
import torch
import torch.distributed as dist
from torchvision import utils
from dist_utils import setup_distributed, cleanup_distributed, is_main, all_gather_tensor, barrier
from unet import TinyUNet
from ddpm import DDPM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/base.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="results/samples_ddp.png")
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open(args.cfg, "r"))
    rank, world_size = setup_distributed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    classes = ckpt["classes"]
    num_classes = len(classes)

    model = TinyUNet(num_classes=num_classes, base_ch=cfg["base_ch"]).to(device)
    model.load_state_dict(ckpt.get("ema_model", ckpt["model"]))
    model.eval()

    ddpm = DDPM(timesteps=cfg["timesteps"], device=device)

    # Each rank generates a shard of samples per class
    n_total = cfg.get("eval_n_per_class", 16)
    n_local = max(1, n_total // world_size)

    all_rows = []
    for cls in range(min(num_classes, cfg.get("sample_classes_cap", 8))):
        y = torch.full((n_local,), cls, device=device, dtype=torch.long)
        seed = cfg.get("seed", 123) + cls + 1000 * rank
        samples = ddpm.sample(
            model, n=n_local, img_size=cfg["img_size"], 
            y=y, seed=seed
        )

        # Gather from all ranks
        gathered = all_gather_tensor(samples)  # (n_total-ish, 3, H, W)
        if is_main():
            all_rows.append(gathered.cpu())

    if is_main():
        grid = utils.make_grid(
            torch.cat(all_rows, 0), nrow=n_total, 
            normalize=True, value_range=(-1, 1)
        )
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        utils.save_image(grid, args.out)
        print("Saved:", args.out)

    barrier()
    cleanup_distributed()

if __name__ == "__main__":
    main()
