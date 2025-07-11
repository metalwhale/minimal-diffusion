import os
import pathlib

from minimal_ddpm.data import MixtureGaussian
from minimal_ddpm.model import MinimalDdpm, train


STORAGE_DIR = pathlib.Path(__file__).parent.parent / "storage"


def main():
    target_distribution = MixtureGaussian([(-2.0, 1.0, 0.4), (2.0, 0.5, 0.6)])
    model = MinimalDdpm()
    result_dir = STORAGE_DIR / "minimal_ddpm"
    os.makedirs(result_dir, exist_ok=True)
    train(result_dir, target_distribution, model)


if __name__ == "__main__":
    main()
