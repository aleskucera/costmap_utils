import math

import numpy as np
import warp as wp

from .geometric_traversability_kernels import apply_gaussian_blur_kernel
from .geometric_traversability_kernels import combine_costs_kernel
from .geometric_traversability_kernels import compute_roughness_kernel
from .geometric_traversability_kernels import compute_slope_sobel_kernel
from .geometric_traversability_kernels import compute_step_height_cost_kernel
from .geometric_traversability_kernels import morph_op_kernel
from .grid_utils import meters_to_cells


def gaussian_kernel(sigma_m: float, grid_resolution: float, device):
    """Creates a 2D Gaussian kernel on the GPU."""
    sigma_cells = meters_to_cells(sigma_m, grid_resolution)
    radius = math.ceil(3.0 * sigma_cells)
    kernel_size = 2 * radius + 1

    kernel_np = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for r_k in range(kernel_size):
        for c_k in range(kernel_size):
            dr, dc = r_k - radius, c_k - radius
            kernel_np[r_k, c_k] = np.exp(-(dr**2 + dc**2) / (2.0 * sigma_cells**2))

    return wp.from_numpy(kernel_np, dtype=wp.float32, device=device), radius


class GeometricTraversabilityAnalyzer:
    """Manages the full GPU-accelerated traversability analysis pipeline."""

    def __init__(
        self,
        device: str,
        verbose: bool,
        # Grid parameters
        grid_resolution: float,
        grid_height: int,
        grid_width: int,
        # Kernel parameters
        smoothing_sigma_m: float,
        roughness_window_radius_m: int,
        # Normalization factors
        max_slope_rad: float,
        max_step_height_m: float,
        max_roughness_m: float,
        # Weights
        slope_cost_weight: float,
        step_height_cost_weight: float,
        surf_roughness_cost_weight: float,
    ):
        # --- Store all arguments as attributes ---
        self.device = device
        self.verbose = verbose

        self.grid_resolution = grid_resolution
        self.height = grid_height
        self.width = grid_width
        self.shape = (self.height, self.width)

        self.roughness_window_radius_cells = meters_to_cells(
            roughness_window_radius_m, grid_resolution
        )

        self.max_slope_rad = max_slope_rad
        self.max_step_height_m = max_step_height_m
        self.max_roughness_m = max_roughness_m

        self.slope_cost_weight = slope_cost_weight
        self.step_height_cost_weight = step_height_cost_weight
        self.surf_roughness_cost_weight = surf_roughness_cost_weight

        # Grid dimensions are initialized later

        with wp.ScopedDevice(self.device):
            self._heightmap = wp.zeros(self.shape, dtype=wp.float32)
            self._heightmap_smoothed = wp.zeros(self.shape, dtype=wp.float32)
            self._normals = wp.zeros(self.shape, dtype=wp.vec3)
            self._slope_cost = wp.zeros(self.shape, dtype=wp.float32)
            self._step_height_cost = wp.zeros(self.shape, dtype=wp.float32)
            self._surf_roughness_cost = wp.zeros(self.shape, dtype=wp.float32)
            self._traversability_cost = wp.zeros(self.shape, dtype=wp.float32)
            self._dilated_map = wp.zeros(self.shape, dtype=wp.float32)
            self._eroded_map = wp.zeros(self.shape, dtype=wp.float32)

        self.gaussian_kernel, self.gaussian_kernel_radius = self.gaussian_kernel(
            smoothing_sigma_m,
            grid_resolution,
            device,
        )

    def compute_traversability(self, heightmap: np.ndarray) -> np.ndarray:
        assert heightmap.shape != self.shape, "Invalid shape of the heightmap."

        self._heightmap.assign(wp.from_numpy(heightmap, device=self.device))

        with wp.ScopedTimer("Full Traversability Pipeline", active=self.verbose):
            # 1. Smooth the input heightmap
            wp.launch(
                kernel=apply_gaussian_blur_kernel,
                dim=(self.height, self.width),
                inputs=[
                    self._heightmap,
                    self.gaussian_kernel,
                    self.gaussian_kernel_radius,
                    self.height,
                    self.width,
                ],
                outputs=[self._heightmap_smoothed],
                device=self.device,
            )

            # 2. Compute slope and surface normals
            wp.launch(
                kernel=compute_slope_sobel_kernel,
                dim=(self.height, self.width),
                inputs=[
                    self._heightmap_smoothed,
                    self.grid_resolution,
                    self.height,
                    self.width,
                    self.max_slope_rad,
                ],
                outputs=[self._normals, self._slope_cost],
                device=self.device,
            )

            # 3. Morphological operations for step height
            wp.launch(
                kernel=morph_op_kernel,
                dim=(self.height, self.width),
                inputs=[
                    self._heightmap_smoothed,
                    self.height,
                    self.width,
                    1,
                ],
                outputs=[self._dilated_map],
                device=self.device,
            )
            wp.launch(
                kernel=morph_op_kernel,
                dim=(self.height, self.width),
                inputs=[
                    self._heightmap_smoothed,
                    self.height,
                    self.width,
                    0,
                ],
                outputs=[self._eroded_map],
                device=self.device,
            )

            # 4. Compute step height cost
            wp.launch(
                kernel=compute_step_height_cost_kernel,
                dim=(self.height, self.width),
                inputs=[
                    self._dilated_map,
                    self._eroded_map,
                    self.max_step_height_m,
                ],
                outputs=[self._step_height_cost],
                device=self.device,
            )

            # 5. Compute surface roughness
            wp.launch(
                kernel=compute_roughness_kernel,
                dim=(self.height, self.width),
                inputs=[
                    self._heightmap_smoothed,
                    self.height,
                    self.width,
                    self.roughness_window_radius_cells,
                    self.max_roughness_m,
                ],
                outputs=[self._surf_roughness_cost],
                device=self.device,
            )

            # 6. Combine all costs
            wp.launch(
                kernel=combine_costs_kernel,
                dim=(self.height, self.width),
                inputs=[
                    self._slope_cost,
                    self._step_height_cost,
                    self._surf_roughness_cost,
                    self.slope_cost_weight,
                    self.step_height_cost_weight,
                    self.surf_roughness_cost_weight,
                ],
                outputs=[self._traversability_cost],
                device=self.device,
            )

        wp.synchronize()

        return self._traversability_cost.numpy()
