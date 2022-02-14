from typing import Optional
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms

import PIL.Image as Image

try:
    from torchmetrics.image import FID, IS
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "torchmetrics is not found. Please install ignite by running `pip install torchmetrics[image]`"
    )

class FID10k(FID):
    def __init__(self, device=None, **kwargs) -> None:
        super().__init__(**kwargs)
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def interpolate229x229(self, batch):
        """
        Resize images to 299 x 299
        """
        arr = []
        for img in batch:
            pil_img = transforms.ToPILImage()(img)
            resized_img = pil_img.resize((299, 299), Image.BILINEAR)
            img_tensor = transforms.ToTensor()(resized_img)
            arr.append(img_tensor)
        return torch.stack(arr)

    def evaluate10k(
        self,
        generator: nn.Module,
        real_data: data.Dataset,
        latent_dim: int,
        n_classes: int,
        batch_size: int = 100,
        sample_size: int = 10_000,
        inv_preprocessing=None,
        )->float:

        n_batch = (sample_size + batch_size - 1) // batch_size
        data_loader = data.DataLoader(real_data, batch_size=batch_size)
        data_iter = iter(data_loader)

        with torch.no_grad():
            for index in range(n_batch):
                latent_space = torch.normal(
                    0, 1, (batch_size, latent_dim), device=self._device, requires_grad=False)
                gen_labels = torch.randint(
                    0, n_classes, (batch_size,), device=self._device, requires_grad=False)
                
                real_img, _ = next(data_iter)
                fake_img = generator(latent_space, gen_labels)

                if inv_preprocessing:
                    real_img = inv_preprocessing(real_img)
                    fake_img = inv_preprocessing(fake_img)
                
                uint_real_img = (self.interpolate229x229(real_img)*255).type(torch.uint8)
                uint_fake_img = (self.interpolate229x229(fake_img)*255).type(torch.uint8)

                uint_real_img = uint_real_img.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                uint_fake_img = uint_fake_img.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

                self.update(uint_real_img, real=True)
                self.update(uint_fake_img, real=False)

        return self.compute().cpu().item()

class IS10k(IS):
    def __init__(self, device=None, **kwargs) -> None:
        super().__init__(**kwargs)
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def interpolate229x229(self, batch):
        """
        Resize images to 299 x 299
        """
        arr = []
        for img in batch:
            pil_img = transforms.ToPILImage()(img)
            resized_img = pil_img.resize((299, 299), Image.BILINEAR)
            img_tensor = transforms.ToTensor()(resized_img)
            arr.append(img_tensor)
        return torch.stack(arr)

    def evaluate10k(
        self,
        generator: nn.Module,
        latent_dim: int,
        n_classes: int,
        batch_size: int = 100,
        sample_size: int = 10_000,
        inv_preprocessing = None,
        )->float:

        n_batch = (sample_size + batch_size - 1) // batch_size

        with torch.no_grad():
            for index in range(n_batch):
                latent_space = torch.normal(
                    0, 1, (batch_size, latent_dim), device=self._device, requires_grad=False)
                gen_labels = torch.randint(
                    0, n_classes, (batch_size,), device=self._device, requires_grad=False)
                
                fake_img = generator(latent_space, gen_labels)
                
                if inv_preprocessing:
                    fake_img = inv_preprocessing(fake_img)

                uint_fake_img = (self.interpolate229x229(fake_img)*255).type(torch.uint8)

                uint_fake_img = uint_fake_img.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

                self.update(uint_fake_img)

        return self.compute()[0].cpu().item()