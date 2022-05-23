import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
from typing import Tuple


class SOM(nn.Module):
    def __init__(self, input_size: int, out_size=(10, 10), lr=0.3) -> None:

        super(SOM, self).__init__()
        self.input_size = input_size
        self.out_size = out_size

        self.lr = lr
        self.sigma = max(out_size) / 2

        self.weight = nn.Parameter(torch.randn(input_size, out_size[0] * out_size[1]))
        self.locations = nn.Parameter(torch.Tensor(list(self.get_map_index())))
        self.pdist_fn = nn.PairwiseDistance(p=2)

    def get_map_index(self):
        for x in range(self.out_size[0]):
            for y in range(self.out_size[0]):
                yield (x, y)

    def _neighborhood_fn(
        self, input: torch.Tensor, current_sigma: float
    ) -> torch.Tensor:
        ''' pow(e, -(input / sigma ** 2))'''
        input = input / current_sigma ** 2
        input = -input
        input = torch.exp(input)

        return input

    def forward(self, input: torch.Tensor) -> Tuple[list, float]:

        batch_size = input.size()[0]
        input = input.view(batch_size, -1, 1)
        batch_weight = self.weight.expand(batch_size, -1, -1)

        dists = self.pdist_fn(input, batch_weight).min(dim=1, keepdim=True)

        losses = dists[0]
        bmu_indexes = dists[1] 
        bmu_locations = self.locations[bmu_indexes]

        return bmu_locations, losses.sum().div_(batch_size).item()

    def self_organizing(
        self, input: torch.Tensor, current_iter: int, max_iter: int
    ) -> float:

        iter_correction = 1.0 - current_iter / max_iter
        lr = self.lr * iter_correction
        sigma = self.sigma * iter_correction

        bmu_locations, loss = self.forward(input)
        # print(f"bmu_loc {type(bmu_locations)} loss {type(loss)}")

        distance_squares = self.locations.float() - bmu_locations.float()
        distance_squares.pow_(2)
        distance_squares = torch.sum(distance_squares, dim=2)

        lr_locations = self._neighborhood_fn(distance_squares, sigma)
        lr_locations.mul_(lr).unsqueeze_(1)

        delta = lr_locations * (input.unsqueeze(2) - self.weight)
        delta = delta.sum(dim=0)
        self.weight.data.add_(delta)

        return loss

    def save_result(self, dir, im_size=(0, 0, 0, 0)) -> None:
        images = self.weight.view(
            im_size[0], im_size[1], im_size[2], self.out_size[0] * self.out_size[1]
        )
        images = images.permute(3, 0, 1, 2)
        save_image(images, dir, normalize=True, padding=1, nrow=self.out_size[0])
