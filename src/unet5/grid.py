import numpy as np
import torch


def create_grid(nc: int, offset=0.5) -> torch.Tensor:
    """
    Create xy values of nc x nc grid
    offset (float): offset in units of original 256 x 256 image
                    offset 0 and nc 256 give identity mapping
                    Use offset 0.5 for shifted contrail label
    Returns: grid (Tensor)
      grid points in [-1, 1] for torch grid_sample()
    """
    grid = np.zeros((nc, nc, 2), dtype=np.float32)
    for ix in range(nc):
        for iy in range(nc):
            grid[ix, iy, 1] = -1 + 2 * (ix + 0.5) / nc + offset / 128
            grid[ix, iy, 0] = -1 + 2 * (iy + 0.5) / nc + offset / 128
    grid = torch.from_numpy(grid).unsqueeze(0)
    return grid
