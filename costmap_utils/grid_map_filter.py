import numpy as np
import warp as wp

from .filter_kernels import filter_grid


class GridMapFilter:
    """Manages GPU-accelerated filtering of grid maps for reliability and other criteria."""

    def __init__(self, **params):
        """
        Initializes the filter with a dictionary of parameters.

        Args:
            params (dict): A dictionary of configuration parameters. Expected keys include:
                - device (str): 'cuda' or 'cpu'.
                - verbose (bool): If True, prints kernel timings.
                - grid_resolution (float): The resolution of the grid in meters/cell.
                - support_radius (int): Radius in cells for the support neighborhood window.
                - support_ratio (float): Threshold ratio of measured (non-NaN) points in the neighborhood.
                - box_filter_enabled (bool): If True, enables box filtering (e.g., to mask robot self-scans).
                - box_size_x (float): Size of the filter box in x-direction (meters).
                - box_size_y (float): Size of the filter box in y-direction (meters).
                - box_size_z (float): Size of the filter box in z-direction (meters); if 0, no z-filtering.
                - box_transform (dict or np.array): Transform defining the box position and orientation.
                  Expected as a 4x4 homogeneous transform matrix or dict with 'position' (x,y,z) and 'rotation' (quaternion).
                - Additional params can be added for other filters (e.g., edge detection, outlier removal).
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
        if hasattr(self, "_elevation_map") and self._elevation_map.shape == shape:
            return

        # Create GPU arrays for inputs and outputs
        with wp.ScopedDevice(self.device):
            self._elevation_map = wp.zeros(shape, dtype=wp.float32)
            self._cost_map = wp.zeros(shape, dtype=wp.float32)
            self._filtered_cost = wp.zeros(shape, dtype=wp.float32)

    def apply_filters(self, raw_elevation_np: np.ndarray, cost_map_np: np.ndarray) -> np.ndarray:
        height, width = raw_elevation_np.shape
        self._initialize_arrays(height, width)

        # Upload data to GPU
        self._elevation_map.assign(wp.from_numpy(raw_elevation_np, device=self.device))
        self._cost_map.assign(wp.from_numpy(cost_map_np, device=self.device))

        p = self.params

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
                    p["support_radius"],
                    p["support_ratio"],
                ],
                outputs=[self._filtered_cost],
                device=self.device,
            )

            # 2. Placeholder for box filtering (to be implemented with a new kernel)
            # If box_filter_enabled, launch a kernel to mask out the box region
            # This would require converting box_transform to grid coordinates and applying a mask
            if p.get("box_filter_enabled", False):
                # TODO: Implement and launch box_filter_kernel
                pass

            # Additional filters can be chained here

        wp.synchronize()

        # Return the filtered cost by copying from GPU to CPU (NumPy)
        return self._filtered_cost.numpy()
