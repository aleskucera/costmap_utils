import numpy as np
import warp as wp

from .filter_kernels import count_obstacles_kernel
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
        obstacle_growth_threshold: float,
        rejection_limit_frames: int,
        min_obstacle_baseline: int,
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

        # State for tracking obstacle growth with hysteresis
        self.num_obstacles_last_frame = 0
        self.consecutive_rejection_count = 0
        self.obstacle_growth_threshold = obstacle_growth_threshold
        self.rejection_limit_frames = rejection_limit_frames
        self.min_obstacle_baseline = min_obstacle_baseline

        with wp.ScopedDevice(self.device):
            self._elevation_map = wp.zeros(self.shape, dtype=wp.float32)
            self._cost_map = wp.zeros(self.shape, dtype=wp.float32)
            self._filtered_cost = wp.zeros(self.shape, dtype=wp.float32)
            self._inflated_cost = wp.zeros(self.shape, dtype=wp.float32)
            self._num_obstacles = wp.zeros(1, dtype=wp.int32)

    def _is_obstacle_growth_stable(self, num_obstacles_current_frame: int) -> bool:
        """
        Checks for rapid obstacle growth and manages state for hysteresis.

        This function contains the core logic for accepting or rejecting a map
        based on the change in obstacle count. It updates the internal state
        for consecutive rejections and the baseline obstacle count.

        Returns:
            True if the frame should be processed (i.e., it's stable or force-accepted).
            False if the frame should be rejected.
        """
        # A low baseline isn't reliable for ratio checks.
        if self.num_obstacles_last_frame < self.min_obstacle_baseline:
            self.num_obstacles_last_frame = num_obstacles_current_frame
            self.consecutive_rejection_count = 0
            return True

        growth_ratio = num_obstacles_current_frame / self.num_obstacles_last_frame

        if growth_ratio > self.obstacle_growth_threshold:
            # Potential bad frame, increment counter
            self.consecutive_rejection_count += 1

            # Check if we are over the rejection limit
            if self.consecutive_rejection_count <= self.rejection_limit_frames:
                # We are within the limit, so reject this frame.
                # Do NOT update the baseline.
                return False
            else:
                # We've exceeded the limit. Force-accept the frame by treating it
                # as stable to establish a new baseline.
                pass

        # --- Stable Frame or Force-Accept Case ---
        # If we reach here, the frame is considered stable.
        # Reset the rejection counter and update the baseline for the next frame.
        self.consecutive_rejection_count = 0
        self.num_obstacles_last_frame = num_obstacles_current_frame
        return True

    def apply_filters(
        self,
        raw_elevation: np.ndarray,
        cost_map: np.ndarray,
    ) -> np.ndarray:
        assert raw_elevation.shape == self.shape, "Invalid shape"
        assert cost_map.shape == self.shape, "Invalid shape"

        self._elevation_map.assign(wp.from_numpy(raw_elevation, device=self.device))
        self._cost_map.assign(wp.from_numpy(cost_map, device=self.device))

        with wp.ScopedTimer("Grid Map Filtering Pipeline", active=self.verbose):
            # 1. Apply support-based filtering
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

            # 2. Count obstacles in the current filtered map
            self._num_obstacles.zero_()
            wp.launch(
                kernel=count_obstacles_kernel,
                dim=(self.height, self.width),
                inputs=[self._filtered_cost, self.obstacle_threshold],
                outputs=[self._num_obstacles],
                device=self.device,
            )

            wp.synchronize()
            num_obstacles_current_frame = self._num_obstacles.numpy()[0]

            # 3. Check for rapid growth with hysteresis
            if not self._is_obstacle_growth_stable(num_obstacles_current_frame):
                # The frame is unstable and should be rejected.
                return np.full_like(cost_map, float("nan"))

            # 4. Inflate obstacles
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
                outputs=[self._inflated_cost],
                device=self.device,
            )

        wp.synchronize()
        return self._inflated_cost.numpy()
