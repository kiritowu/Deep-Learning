import imp
import torch
import torch.nn as nn

from models import acgan
from models.acgan import Generator


class Discriminator(acgan.Discriminator):
    def __init__(self, n_classes: int, image_size: int, channels: int = 3, **kwargs):
        super().__init__(n_classes, image_size, channels, **kwargs)
        self.adv_layer[-1] = nn.Identity()

def calculate_gradient_penalty(discriminator, real_images, fake_images, device):
    """Calculates the gradient penalty loss for WGAN GP
    Reference from:
    https://github.com/Lornatang/WassersteinGAN_GP-PyTorch/blob/master/wgangp_pytorch/utils.py
    """
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real_images.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)
    model_interpolates, _ = discriminator(interpolates)
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty