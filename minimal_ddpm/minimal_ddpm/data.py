import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class MixtureGaussian:
    _parameters: list[tuple[float, float]]
    _weights: list[float]

    def __init__(self, parameters: list[tuple[float, float, float]]):
        self._parameters = []
        self._weights = []
        assert isinstance(parameters, list)
        for parameter in parameters:
            assert isinstance(parameter, tuple)
            assert len(parameter) == 3
            m, s, w = parameter  # Mean, standard deviation and weight
            assert isinstance(m, float) and isinstance(s, float)
            self._parameters.append((m, s))
            assert isinstance(w, float)
            self._weights.append(w)
        weights_sum = sum(self._weights)
        for i, weight in enumerate(self._weights):
            self._weights[i] = weight / weights_sum

    def sample(self, size: int) -> list[float]:
        samples = []
        for _ in range(size):
            parameter = self._parameters[np.random.choice(len(self._parameters), p=self._weights)]
            m, s = parameter
            samples.append(np.random.normal(loc=m, scale=s))
        return samples

    def mean(self) -> float:
        return sum([m * w for (m, *_), w in zip(self._parameters, self._weights)])


def plot_histogram(
    samples: list[float],
    bin_num: int,
    domain: tuple[float, float] | None = None,
    top: float | None = None,
    title: str | None = None,
) -> np.ndarray:
    if domain is None:
        domain = (min(samples), max(samples))
    plt.hist(samples, bins=bin_num, range=domain)
    if top is not None:
        plt.ylim(top=top)
    if title is not None:
        plt.title(title, loc="center")
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.clf()
    buffer.seek(0)
    image = np.array(Image.open(buffer))
    return image
