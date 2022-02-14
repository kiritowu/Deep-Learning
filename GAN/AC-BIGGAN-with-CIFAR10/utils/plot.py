import os
from typing import Dict, Sequence, Union

import wandb
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np


def inverseNormalize(imgs: torch.Tensor, mean, std):
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    std_inv = 1 / (std + 1e-7)
    mean_inv = -mean * std_inv

    return F.normalize(imgs, mean=mean_inv, std=std_inv)

def classnames_from_tensor(labels: torch.Tensor, classname_mapping: Sequence[str])->Sequence[str]:
    """
    Return list of classnames based on label tensor
    """
    return list(map(lambda l: classname_mapping[l], labels))


def plot_grid(
    epoch:int,
    imgs: torch.Tensor,
    grid_row: int = 8,
    labels=None,
    save_path=None,
    figsize=None,
    inv_preprocessing=None,
    sort_by_class=None,
    save_wandb=False,
    disable_visualise=False,
) -> None:
    """
    Generate matplotlib figure based on input tensor in dimension (N, C, W, H).
    """
    if sort_by_class:
        raise NotImplementedError()

    if not figsize:
        figsize = (9, 9)

    if inv_preprocessing:
        for inv in inv_preprocessing:
            imgs = inv(imgs)

    fig = plt.figure(figsize=figsize, tight_layout=True)
    for i in range(grid_row * grid_row):
        ax = fig.add_subplot(grid_row, grid_row, i + 1)
        ax.imshow(F.to_pil_image(imgs[i]))
        ax.set_title("{}".format(labels[i] if labels else ""))
        ax.axis("off")

    if save_path:
        plt.savefig(f"{save_path}/generated_img_{epoch}.png")

    if save_wandb:
        wandb.log({"generated_images":wandb.Image(imgs), "epoch":epoch})

    if not disable_visualise:
        plt.show()
    else:
        plt.close()


def save_all_generated_img(
    epoch:int,
    base_folder:str,
    generator:nn.Module,
    image_num:int,
    n_classes:int,
    latent_dim:int,
    classname_mapping:Sequence[str],
    device:Union[str,torch.device],
    inv_preprocessing=None,
    )->None:
    batch_size = image_num//n_classes
    n_batch = image_num//batch_size
    latent_space = torch.normal(
                    0, 1, (n_batch, batch_size, latent_dim), device=device, requires_grad=False)
    gen_grids = torch.arange(0, n_classes, device=device)\
                    .tile(100)\
                    .reshape(batch_size, n_batch)\
                    .T
    # Randomly shuffle the labels or else the generated result will be bad 
    # due different distribution than training
    # Generate Random Indices
    perm_indices = torch.randperm(batch_size*n_batch)
    # Generate Indices Mapping
    random_map = {idx: int(random_idx) for idx, random_idx in enumerate(perm_indices)}
    reverse_map = {v:k for k,v in random_map.items()}
    # Generate Indices List for Mapping
    to_random = list(random_map.values())
    to_ordered = list({reverse_key: reverse_map[reverse_key] for reverse_key in sorted(reverse_map)}.values())
    # Randomized Label
    gen_labels = gen_grids.reshape(-1)[to_random].reshape(n_batch, batch_size)

    gen_imgs = []
    with torch.no_grad():
        for latent_batch, gen_batch in zip(latent_space, gen_labels):
            gen_img = generator(latent_batch, gen_batch)
            gen_imgs.append(gen_img.cpu())
            
    gen_imgs_torch = torch.stack(gen_imgs).reshape(-1, 3, 32, 32).cpu()
    gen_labels = np.array(classnames_from_tensor(gen_labels.reshape(-1).cpu(), classname_mapping))
    # Revert Randomized Label and Image to Ordered Form
    gen_imgs_torch_sorted = gen_imgs_torch[to_ordered]
    gen_labels_sorted = gen_labels[to_ordered]

    for folder_dir in [base_folder, f"{base_folder}/{epoch}", 
                        *[f"{base_folder}/{epoch}/{classname}" for classname in classname_mapping]]:
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)

    if inv_preprocessing is not None:
        gen_imgs_torch_sorted = inv_preprocessing(gen_imgs_torch_sorted)

    for idx, (img, label) in enumerate(zip(gen_imgs_torch_sorted, gen_labels_sorted)):
        F.to_pil_image(img).save(f"{base_folder}/{epoch}/{label}/{idx}.png")