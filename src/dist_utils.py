import os
import torch
import torch.distributed as dist

def setup_distributed():
    """Initialize torch.distributed from env vars set by torchrun."""
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return rank, world_size

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())

    return rank, world_size

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

@torch.no_grad()
def broadcast_model(model, src=0):
    """Broadcast model parameters from src rank to all ranks."""
    for p in model.parameters():
        dist.broadcast(p.data, src=src)

@torch.no_grad()
def all_gather_tensor(t: torch.Tensor):
    """Gather same-shaped tensors from all ranks."""
    world_size = dist.get_world_size()
    gather_list = [torch.empty_like(t) for _ in range(world_size)]
    dist.all_gather(gather_list, t)
    return torch.cat(gather_list, dim=0)
