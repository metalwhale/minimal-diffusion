import csv
import math
import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn, optim

from .data import MixtureGaussian, plot_histogram


_DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


# Doc: https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
class MinimalDdpm(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x):
        logits = self.sequential(x)
        return logits


# Ref: https://arxiv.org/abs/2406.08929
def train(
    result_dir: os.PathLike,
    target_distribution: MixtureGaussian,  # Target distribution
    model: MinimalDdpm,
    s: float = 2.0,  # Terminal standard deviation
    dt: float = 0.01,  # Step size
    batch_size: int = 10000,
    epoch_num: int = 10000,
) -> MinimalDdpm:
    model = model.to(_DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters())
    with open(result_dir / "log.csv", "w") as log_file:
        log_writer = csv.DictWriter(log_file, ["epoch", "loss"])
        log_writer.writeheader()
        for epoch in range(epoch_num):
            x_batch = []
            y_batch = []
            for _ in range(batch_size):
                x0 = target_distribution.sample(1)[0]
                t = np.random.uniform(low=0.0, high=1.0)
                xt = x0 + np.random.normal(scale=s * math.sqrt(t))
                xt_next = xt + np.random.normal(scale=s * math.sqrt(dt))
                x_batch.append([xt_next, t + dt])
                y_batch.append(x0)
            x_batch = torch.tensor(x_batch).to(_DEVICE)
            y_batch = torch.tensor(y_batch).to(_DEVICE)
            pred = model(x_batch).squeeze(1)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch % 500 == 0:
                log_writer.writerow({"epoch": epoch, "loss": loss.item()})
                log_file.flush()
                _eval(result_dir, target_distribution, model, s, dt, f"{epoch:0{len(str(epoch_num))}d}", epoch)
    return model


def _eval(
    result_dir: os.PathLike,
    target_distribution: MixtureGaussian,
    model: MinimalDdpm,
    s: float,
    dt: float,
    image_name: str,
    epoch: int,
    sample_num: int = 100000,
    bin_num: int = 1000,
):
    # Sample the distributions
    target_samples = target_distribution.sample(sample_num)
    diffusion_samples = _diffuse(target_distribution, s, dt, sample_num)
    reverse_samples = _reverse(model, target_distribution.mean(), s, dt, sample_num)
    # Draw the histograms
    domain = (
        min(min(target_samples), min(diffusion_samples), min(reverse_samples)),
        max(max(target_samples), max(diffusion_samples), max(reverse_samples)),
    )
    target_hist, _ = np.histogram(target_samples, bins=bin_num, range=domain)
    diffusion_hist, _ = np.histogram(diffusion_samples, bins=bin_num, range=domain)
    reverse_hist, _ = np.histogram(reverse_samples, bins=bin_num, range=domain)
    max_hist = max(max(target_hist), max(diffusion_hist), max(reverse_hist))
    target_image = plot_histogram(target_samples, bin_num, domain=domain, top=max_hist, title="Target")
    diffusion_image = plot_histogram(diffusion_samples, bin_num, domain=domain, top=max_hist, title="Diffusion")
    reverse_image = plot_histogram(reverse_samples, bin_num, domain=domain, top=max_hist, title="Reverse")
    image = Image.fromarray(np.hstack([target_image, diffusion_image, reverse_image]))
    image_draw = ImageDraw.Draw(image)
    image_draw.text((10, 10), f"Epoch: {epoch}", fill="black", font_size=20)
    image.save(result_dir / f"{image_name}.png")


def _diffuse(
    target_distribution: MixtureGaussian,
    s: float,
    dt: float,
    batch_size: int,
) -> list[float]:
    xt_batch = np.array(target_distribution.sample(batch_size))
    for _ in np.arange(0.0, 1.0, dt):
        n_batch = np.random.normal(scale=s * math.sqrt(dt), size=batch_size)  # Gaussian noise
        xt_batch += n_batch
    samples = xt_batch.tolist()
    return samples


def _reverse(
    model: MinimalDdpm,
    m: float,
    s: float,
    dt: float,
    batch_size: int,
) -> list[float]:
    xt_batch = np.random.normal(loc=m, scale=s, size=batch_size)
    with torch.no_grad():
        xt_batch = torch.tensor(xt_batch, dtype=torch.float32).to(_DEVICE).unsqueeze(1)
        for t in np.arange(1.0, 0.0, -dt):
            t_batch = torch.tensor([t for _ in range(batch_size)], dtype=torch.float32).to(_DEVICE).unsqueeze(1)
            x0_mean_batch = model(torch.cat((xt_batch, t_batch), dim=1))
            xt_prev_mean_batch = dt / t * x0_mean_batch + (1 - dt / t) * xt_batch
            n_batch = np.random.normal(scale=s * math.sqrt(dt), size=batch_size)  # Gaussian noise
            n_batch = torch.tensor(n_batch, dtype=torch.float32).to(_DEVICE).unsqueeze(1)
            xt_batch = xt_prev_mean_batch + n_batch
        xt_batch = xt_batch.squeeze(1)
    samples = xt_batch.tolist()
    return samples
