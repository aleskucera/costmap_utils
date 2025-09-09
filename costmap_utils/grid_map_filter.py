import numpy as np
import warp as wp
from geometry_msgs.msg import TransformStamped

from .filter_kernels import filter_grid
from .filter_kernels import inflate_obstacles_kernel
from .grid_utils import meters_to_cells


class GridMapFilter:
    def __init__(
        self,
        device: wp.context.Device,
        verbose: bool,
        grid_resolution: float,
        grid_height: int,
        grid_width: int,
        support_radius_m: float,
        support_ratio: float,
        inflation_radius_m: float,
        obstacle_threshold: float,
    ):

        self.device = device
        self.verbose = verbose

        self.grid_resolution = grid_resolution
        self.height = grid_height
        self.width = grid_width
        self.shape = (self.height, self.width)

        self.support_radius_cells = meters_to_cells(support_radius_m, grid_resolution)
        self.support_ratio = support_ratio

        self.inflation_radius_cells = meters_to_cells(inflation_radius_m, grid_resolution)
        self.obstacle_threshold = obstacle_threshold

        # Create GPU arrays for inputs and outputs
        with wp.ScopedDevice(self.device):
            self._elevation_map = wp.zeros(self.shape, dtype=wp.float32)
            self._cost_map = wp.zeros(self.shape, dtype=wp.float32)
            self._filtered_cost = wp.zeros(self.shape, dtype=wp.float32)
            self._inflated_cost = wp.zeros(self.shape, dtype=wp.float32)
            self._box_filtered_cost = wp.zeros(self.shape, dtype=wp.float32)

    def apply_filters(
        self,
        raw_elevation: np.ndarray,
        cost_map: np.ndarray,
    ) -> np.ndarray:
        assert raw_elevation.shape == self.shape, "Invalid shape of the raw elevation map."
        assert cost_map.shape == self.shape, "Invalid shape of the cost map."

        # Upload data to GPU
        self._elevation_map.assign(wp.from_numpy(raw_elevation, device=self.device))
        self._cost_map.assign(wp.from_numpy(cost_map, device=self.device))

        with wp.ScopedTimer("Grid Map Filtering Pipeline", active=self.verbose):
            # 1. Apply support-based filtering (reliability based on NaN density)
            wp.launch(
                kernel=filter_grid,
                dim=(self.height, self.width),
                inputs=[
                    self._elevation_map,
                    self._cost_map,
                    self.height,
                    self.width,
                    self.support_radius_cells,
                    self.support_ratio,
                ],
                outputs=[self._filtered_cost],
                device=self.device,
            )

            wp.launch(
                kernel=inflate_obstacles_kernel,
                dim=(self.height, self.width),
                inputs=[
                    self._filtered_cost,
                    self.height,
                    self.width,
                    self.inflation_radius_cells,
                    self.obstacle_threshold,
                ],
                outputs=[
                    self._inflated_cost,
                ],
            )

        wp.synchronize()

        return self._inflated_cost.numpy()
