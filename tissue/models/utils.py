import torch


def move_to_numpy(x: torch.Tensor):
    if isinstance(x, torch.Tensor):
        if "cuda" in x.device.type:  # if tensor is on gpu
            x = x.cpu()
        if x.requires_grad:
            x = x.detach()
        x = x.numpy()
    return x
