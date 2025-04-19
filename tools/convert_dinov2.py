import torch
import os.path as osp
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import argparse


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("pretrained", type=str)
    args.add_argument("converted", type=str)
    args.add_argument("--kernel", default=16, type=int)
    args.add_argument("--height", default=512, type=int)
    args.add_argument("--width", default=512, type=int)
    args.add_argument("--patch_key", default="backbone.patch_embed.proj.weight", type=str)
    args.add_argument("--pos_key", default="backbone.pos_embed", type=str)
    return args.parse_args()


def load_weight(pretrained_path):
    if not osp.isfile(pretrained_path):
        raise FileNotFoundError(
            f"{pretrained_path} doesn't exist (absolute path: {osp.abspath(pretrained_path)})"
        )
    weight = torch.load(pretrained_path, map_location="cpu")
    if not isinstance(weight, dict):
        raise TypeError("Loaded object is not a dict.")
    # Some checkpoints might wrap weights in another dict (e.g., {'model': ...})
    if "model" in weight:
        weight = weight["model"]
    return weight


def interpolate_patch_embed_(weight, key="backbone.patch_embed.proj.weight", kernel_conv=16):
    if key not in weight:
        raise KeyError(f"{key} not found in weight keys.")
    ori_shape = weight[key].shape
    weight[key] = F.interpolate(
        weight[key].float(),
        size=(kernel_conv, kernel_conv),
        mode="bicubic",
        align_corners=False,
    )
    dst_shape = weight[key].shape
    print(f"Converted patch_embed: {ori_shape} -> {dst_shape}")


def interpolate_pos_embed_(weight, key="backbone.pos_embed", crop_size=(512, 512), kernel_conv=16):
    if key not in weight:
        raise KeyError(f"{key} not found in weight keys.")
    pos_embed = weight[key]
    pos_cls, pos_tokens = pos_embed[:, :1, :], pos_embed[:, 1:, :]
    embed_dim = pos_tokens.shape[-1]
    orig_size = int(pos_tokens.shape[1] ** 0.5)
    assert orig_size * orig_size == pos_tokens.shape[1], "Position tokens must form a square grid."
    crop_size = tuple(L // kernel_conv for L in crop_size)
    pos_tokens = pos_tokens.reshape(1, orig_size, orig_size, embed_dim).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(
        pos_tokens, size=crop_size, mode="bicubic", align_corners=False
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    weight[key] = torch.cat((pos_cls, pos_tokens), dim=1)
    print(f"Converted pos_embed to {crop_size[0]}x{crop_size[1]} patches")


def main():
    args = parse_args()
    weight = load_weight(args.pretrained)
    print("Loaded pretrained weights from", args.pretrained)

    interpolate_patch_embed_(weight, key=args.patch_key, kernel_conv=args.kernel)
    interpolate_pos_embed_(
        weight,
        key=args.pos_key,
        crop_size=(args.height, args.width),
        kernel_conv=args.kernel,
    )

    torch.save({"model": weight}, args.converted)
    print("Saved converted weights to", args.converted)


if __name__ == "__main__":
    main()
