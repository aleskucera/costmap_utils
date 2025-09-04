import math

import numpy as np
import warp as wp

from .geometric_traversability_kernels import apply_gaussian_blur_kernel
from .geometric_traversability_kernels import combine_costs_kernel
from .geometric_traversability_kernels import compute_roughness_kernel
from .geometric_traversability_kernels import compute_slope_sobel_kernel
from .geometric_traversability_kernels import compute_step_height_cost_kernel
from .geometric_traversability_kernels import morph_op_kernel


class GeometricTraversabilityAnalyzer:
    """Manages the full GPU-accelerated traversability analysis pipeline."""

    def __init__(self, **params):
        """
        Initializes the analyzer with a dictionary of parameters.

        Args:
            params (dict): A dictionary of configuration parameters. Expected keys include:
                - device (str): 'cuda' or 'cpu'.
                - verbose (bool): If True, prints kernel timings.
                - grid_resolution (float): The resolution of the grid in meters/cell.
                - smoothing_sigma (float): Gaussian blur sigma in meters.
                - slope_normalization_factor (float): Max slope in radians for normalization.
                - step_height_normalization_factor (float): Max step height in meters for normalization.
                - surf_roughness_normalization_factor (float): Max roughness in meters for normalization.
                - slope_cost_weight (float): Weight for the slope cost.
                - step_height_cost_weight (float): Weight for the step height cost.
                - surf_roughness_cost_weight (float): Weight for the surface roughness cost.
                - roughness_window_radius (int): Radius for the roughness calculation window in cells.
        """
        self.params = params
        self.device = params.get("device", "cuda")
        self.verbose = params.get("verbose", False)

        # Grid parameters are taken from the input map, not during initialization
        self.height = 0
        self.width = 0
        self.resolution = params["grid_resolution"]

    def _initialize_arrays(self, height: int, width: int):
        """
        Allocates or resizes GPU arrays if the input map dimensions change.
        """
        self.height, self.width = height, width
        shape = (self.height, self.width)

        # Re-allocate only if shape has changed to avoid unnecessary overhead
        if hasattr(self, "_heightmap") and self._heightmap.shape == shape:
            return

        # Create GPU arrays for all intermediate and final maps
        dtype_f32 = wp.float32
        self._heightmap = wp.zeros(shape, dtype=dtype_f32, device=self.device)
        self._heightmap_smoothed = wp.zeros(shape, dtype=dtype_f32, device=self.device)
        self._normals = wp.zeros(shape, dtype=wp.vec3, device=self.device)
        self._slope_cost = wp.zeros(shape, dtype=dtype_f32, device=self.device)
        self._step_height_cost = wp.zeros(shape, dtype=dtype_f32, device=self.device)
        self._surf_roughness_cost = wp.zeros(shape, dtype=dtype_f32, device=self.device)
        self._traversability_cost = wp.zeros(shape, dtype=dtype_f32, device=self.device)
        self._dilated_map = wp.zeros(shape, dtype=dtype_f32, device=self.device)
        self._eroded_map = wp.zeros(shape, dtype=dtype_f32, device=self.device)

    def _create_gaussian_kernel(self) -> tuple[wp.array, int]:
        """Creates a 2D Gaussian kernel on the GPU."""
        sigma_m = self.params["smoothing_sigma"]
        sigma_cells = sigma_m / self.resolution
        radius = math.ceil(3.0 * sigma_cells)
        kernel_size = 2 * radius + 1

        kernel_np = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        for r_k in range(kernel_size):
            for c_k in range(kernel_size):
                dr, dc = r_k - radius, c_k - radius
                kernel_np[r_k, c_k] = np.exp(-(dr**2 + dc**2) / (2.0 * sigma_cells**2))

        return wp.from_numpy(kernel_np, dtype=wp.float32, device=self.device), radius

    def compute_traversability(self, heightmap_np: np.ndarray) -> dict[str, np.ndarray]:
        """
        Executes the full traversability pipeline on a given heightmap.

        Args:
            heightmap_np: A 2D NumPy array representing the input heightmap.

        Returns:
            A dictionary containing various computed maps as NumPy arrays.
        """
        height, width = heightmap_np.shape
        self._initialize_arrays(height, width)

        # Warp's `assign` is an efficient way to copy data from a CPU (NumPy) array
        # to a GPU (Warp) array. Based on the documentation, this is similar to
        # creating a new array from numpy, but can reuse existing memory.
        # [nvidia.github.io](https://nvidia.github.io/warp/basics.html)
        self._heightmap.assign(wp.from_numpy(heightmap_np, device=self.device))

        p = self.params
        gaussian_kernel, kernel_radius = self._create_gaussian_kernel()

        with wp.ScopedTimer("Full Traversability Pipeline", active=self.verbose):
            # 1. Smooth the input heightmap
            wp.launch(
                kernel=apply_gaussian_blur_kernel,
                dim=(self.height, self.width),
                inputs=[self._heightmap, gaussian_kernel, kernel_radius, self.height, self.width],
                outputs=[self._heightmap_smoothed],
                device=self.device,
            )

            # 2. Compute slope and surface normals
            wp.launch(
                kernel=compute_slope_sobel_kernel,
                dim=(self.height, self.width),
                inputs=[
                    self._heightmap_smoothed,
                    self.resolution,
                    self.height,
                    self.width,
                    p["slope_normalization_factor"],
                ],
                outputs=[self._normals, self._slope_cost],
                device=self.device,
            )

            # 3. Morphological operations for step height
            wp.launch(
                kernel=morph_op_kernel,
                dim=(self.height, self.width),
                inputs=[self._heightmap_smoothed, self.height, self.width, 1],
                outputs=[self._dilated_map],
                device=self.device,
            )  # 1=DILATE
            wp.launch(
                kernel=morph_op_kernel,
                dim=(self.height, self.width),
                inputs=[self._heightmap_smoothed, self.height, self.width, 0],
                outputs=[self._eroded_map],
                device=self.device,
            )  # 0=ERODE

            # 4. Compute step height cost
            wp.launch(
                kernel=compute_step_height_cost_kernel,
                dim=(self.height, self.width),
                inputs=[self._dilated_map, self._eroded_map, p["step_height_normalization_factor"]],
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
                    p["roughness_window_radius"],
                    p["surf_roughness_normalization_factor"],
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
                    p["slope_cost_weight"],
                    p["step_height_cost_weight"],
                    p["surf_roughness_cost_weight"],
                ],
                outputs=[self._traversability_cost],
                device=self.device,
            )

        wp.synchronize()

        # Return results by copying from GPU to CPU (NumPy)
        return {
            "smoothed_map": self._heightmap_smoothed.numpy(),
            "normals": self._normals.numpy(),
            "slope_cost": self._slope_cost.numpy(),
            "step_height_cost": self._step_height_cost.numpy(),
            "surf_roughness_cost": self._surf_roughness_cost.numpy(),
            "traversability_cost": self._traversability_cost.numpy(),
        }
