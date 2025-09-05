import math

import numpy as np
import rclpy
import tf2_ros
import warp as wp
from geometry_msgs.msg import TransformStamped

from .filter_kernels import filter_box_kernel
from .filter_kernels import filter_grid
from .filter_kernels import filter_rotated_box_kernel


class GridMapFilter:
    """Manages GPU-accelerated filtering of grid maps for reliability and other criteria."""

    def __init__(self, tf_buffer=None, **params):
        """
        Initializes the filter with a dictionary of parameters.

        Args:
            tf_buffer: TF buffer for transform lookups (optional)
            params (dict): A dictionary of configuration parameters.
        """
        self.params = params
        self.device = params.get("device", "cuda")
        self.verbose = params.get("verbose", False)
        self.tf_buffer = tf_buffer

        # Store the last valid transform
        self.last_valid_transform = None
        self.last_transform_time = None

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
            self._box_filtered_cost = wp.zeros(shape, dtype=wp.float32)

    def apply_filters(
        self,
        raw_elevation_np: np.ndarray,
        cost_map_np: np.ndarray,
        map_frame: str,
        map_origin_x: float = 0.0,
        map_origin_y: float = 0.0,
        map_length_x: float = None,
        map_length_y: float = None,
    ) -> np.ndarray:
        """
        Apply filters to the cost map.

        Args:
            raw_elevation_np: Raw elevation map data
            cost_map_np: Cost map data to filter
            map_frame: The frame ID of the grid map
            map_origin_x: X coordinate of map origin
            map_origin_y: Y coordinate of map origin
            map_length_x: Map length in x direction (meters)
            map_length_y: Map length in y direction (meters)

        Returns:
            Filtered cost map as numpy array
        """
        height, width = raw_elevation_np.shape
        self._initialize_arrays(height, width)

        # Set map dimensions if not provided
        if map_length_x is None:
            map_length_x = width * self.resolution
        if map_length_y is None:
            map_length_y = height * self.resolution

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

            # 2. Apply box filtering if enabled
            filtered_result = self._filtered_cost

            if p.get("box_filter_enabled", False):
                # Convert box dimensions from meters to grid cells
                box_size_x_cells = int(p.get("box_size_x", 1.0) / self.resolution)
                box_size_y_cells = int(p.get("box_size_y", 1.0) / self.resolution)

                # Check if we should use TF
                if self.tf_buffer is not None and p.get("use_tf", True):
                    # Try to get the transform from map frame to base_link
                    base_link_frame = p.get("base_link_frame", "base_link")
                    transform = None

                    try:
                        # Get the transform from map to base_link
                        transform = self.tf_buffer.lookup_transform(
                            map_frame,  # target frame
                            base_link_frame,  # source frame
                            rclpy.time.Time(0),  # get the latest available transform
                            rclpy.duration.Duration(seconds=0.1),  # timeout
                        )

                        # Store as last valid transform
                        self.last_valid_transform = transform
                        self.last_transform_time = rclpy.time.Time.now()

                        if self.verbose:
                            print(
                                f"Using fresh TF transform: robot at ({transform.transform.translation.x}, {transform.transform.translation.y})"
                            )

                    except Exception as e:
                        # Log the error but try to use last valid transform
                        print(f"Error getting fresh transform: {e}")
                        if self.verbose:
                            import traceback

                            traceback.print_exc()

                        # Use last valid transform if it exists
                        transform = self.last_valid_transform
                        if transform is not None and self.verbose:
                            print(f"Using last valid transform from {self.last_transform_time}")

                    # If we have a valid transform (either fresh or stored), use it
                    if transform is not None:
                        # Extract robot position in map frame
                        robot_x = transform.transform.translation.x
                        robot_y = transform.transform.translation.y

                        # Extract rotation quaternion
                        q = transform.transform.rotation
                        # Convert quaternion to yaw angle
                        yaw = math.atan2(
                            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                        )

                        # GridMap's coordinate system has origin at the geometric center of the grid
                        # First, convert from world to grid coordinates

                        # Calculate cell coordinates
                        # Note: In GridMap, the origin is at the center of the grid
                        # We need to translate from map origin to the bottom-left corner (0,0) of the grid
                        grid_origin_x = map_origin_x - map_length_x / 2.0
                        grid_origin_y = map_origin_y - map_length_y / 2.0

                        # Convert robot position to cell coordinates
                        cell_x = (robot_x - grid_origin_x) / self.resolution
                        cell_y = (robot_y - grid_origin_y) / self.resolution

                        # Convert to row, col format (row increases from top to bottom)
                        center_c = int(cell_x)
                        center_r = int(height - cell_y - 1)  # Flip y-axis

                        # Ensure values are within grid bounds
                        center_c = max(0, min(center_c, width - 1))
                        center_r = max(0, min(center_r, height - 1))

                        # Calculate sine and cosine of the yaw angle
                        # Note: we negate the yaw because of the row/column conversion
                        cos_theta = math.cos(-yaw)
                        sin_theta = math.sin(-yaw)

                        # Launch the rotated box filter kernel
                        wp.launch(
                            kernel=filter_rotated_box_kernel,
                            dim=(self.height, self.width),
                            inputs=[
                                self._filtered_cost,
                                self.height,
                                self.width,
                                center_r,
                                center_c,
                                box_size_x_cells // 2,
                                box_size_y_cells // 2,
                                cos_theta,
                                sin_theta,
                            ],
                            outputs=[self._box_filtered_cost],
                            device=self.device,
                        )

                        filtered_result = self._box_filtered_cost
                    else:
                        # No transform available (neither fresh nor stored)
                        # Just return the filtered cost without box filtering
                        if self.verbose:
                            print("No valid transform available, skipping box filter")
                else:
                    # TF not enabled, no box filtering
                    if self.verbose:
                        print("TF not enabled, skipping box filter")

            # Additional filters can be chained here

        wp.synchronize()

        # Return the filtered cost by copying from GPU to CPU (NumPy)
        return filtered_result.numpy()
