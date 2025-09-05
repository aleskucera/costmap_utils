#!/usr/bin/env python3
import math
import traceback

import numpy as np
import rclpy
import warp as wp
from grid_map_msgs.msg import GridMap
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField

from .geometric_traversability_analyzer import GeometricTraversabilityAnalyzer
from .grid_map_filter import GridMapFilter


class GeometricTraversabilityNode(Node):
    """A ROS 2 Node for GPU-accelerated geometric traversability analysis from GridMaps."""

    def __init__(self):
        super().__init__("geometric_traversability_node")

        # --- Parameters for Configuration ---
        self.declare_parameter("input_topic", "/elevation_mapping_node/elevation_map_filter")
        self.declare_parameter("output_topic", "/traversability_cloud")
        self.declare_parameter("traversability_input_layer", "inpaint")
        self.declare_parameter("use_cpu", False)
        self.declare_parameter("verbose", False)

        # Cost weights
        self.declare_parameter("weights.slope", 0.4)
        self.declare_parameter("weights.step_height", 0.4)
        self.declare_parameter("weights.surface_roughness", 0.2)

        # Pre-processing
        self.declare_parameter("preprocessing.smoothing_sigma_m", 0.08)

        # Normalization thresholds
        self.declare_parameter("normalization.max_slope_deg", 70.0)
        self.declare_parameter("normalization.max_step_height_m", 0.4)
        self.declare_parameter("normalization.max_roughness_m", 0.2)

        # Neighborhood parameters (in grid cells)
        self.declare_parameter("neighborhood.roughness_window_radius_cells", 2)

        # --- Added parameters for the new GridMapFilter ---
        self.declare_parameter("filter.enabled", True)
        self.declare_parameter("filter.raw_elevation_layer", "elevation")
        self.declare_parameter("filter.support_radius_cells", 2)
        self.declare_parameter("filter.support_ratio", 0.75)

        # --- Subscribers and Publishers ---
        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value

        self.subscription = self.create_subscription(
            GridMap, self.input_topic, self.map_callback, 10
        )
        self.publisher = self.create_publisher(
            PointCloud2, self.output_topic, qos_profile_system_default
        )

        self.analyzer = None  # Will be initialized on first message
        self.filter = None  # Will be initialized on first message

        self.get_logger().info(
            f"Node '{self.get_name()}' initialized. Waiting for GridMap on '{self.input_topic}'..."
        )

    def map_callback(self, msg: GridMap):
        """Callback to process an incoming GridMap message."""
        if self.analyzer is None:
            self.initialize_analyzer(msg)

        filter_enabled = self.get_parameter("filter.enabled").value
        if filter_enabled and self.filter is None:
            self.initialize_filter(msg)

        try:
            resolution = msg.info.resolution
            rows = int(round(msg.info.length_y / resolution))
            cols = int(round(msg.info.length_x / resolution))

            # --- 1. Extract and Process Layers ---
            traversability_layer_name = self.get_parameter("traversability_input_layer").value
            if traversability_layer_name not in msg.layers:
                self.get_logger().error(
                    f"Traversability input layer '{traversability_layer_name}' not found. Available: {msg.layers}",
                    throttle_duration_sec=5,
                )
                return

            layer_idx = msg.layers.index(traversability_layer_name)

            # --- FIXED: Reshape based on row-major ('C') order used by grid_map_msgs ---
            # Removing order='F' makes it default to 'C' (row-major).
            inpainted_elevation_np = np.array(msg.data[layer_idx].data, dtype=np.float32).reshape(
                (rows, cols), order="C"
            )

            # Prepare inpainted map for traversability analyzer (fill NaNs)
            nan_mask = np.isnan(inpainted_elevation_np)
            traversability_input_map = np.copy(inpainted_elevation_np)
            fill_value = np.nanmin(traversability_input_map) if not np.all(nan_mask) else 0.0
            traversability_input_map[nan_mask] = fill_value

            # --- 2. Compute Traversability ---
            results = self.analyzer.compute_traversability(traversability_input_map)
            traversability_cost_map = results["traversability_cost"]
            # Restore NaNs where the original inpainted map had them
            traversability_cost_map[nan_mask] = np.nan

            final_cost_map = traversability_cost_map

            # --- 3. Apply Reliability Filter (Optional) ---
            if filter_enabled:
                raw_elevation_layer_name = self.get_parameter("filter.raw_elevation_layer").value
                if raw_elevation_layer_name not in msg.layers:
                    self.get_logger().error(
                        f"Raw elevation layer '{raw_elevation_layer_name}' for filtering not found. Available: {msg.layers}",
                        throttle_duration_sec=5,
                    )
                    return

                raw_layer_idx = msg.layers.index(raw_elevation_layer_name)
                # --- FIXED: Use correct row-major ('C') order for this layer as well ---
                raw_elevation_np = np.array(msg.data[raw_layer_idx].data, dtype=np.float32).reshape(
                    (rows, cols), order="C"
                )

                # Apply the filter
                self.get_logger().debug("Applying reliability filter to the cost map.")
                final_cost_map = self.filter.apply_filters(
                    raw_elevation_np, traversability_cost_map
                )

            # --- 4. Create point cloud from traversability and elevation data ---
            points = self.create_point_cloud_data(
                final_cost_map,
                inpainted_elevation_np,
                msg.info.pose.position.x,
                msg.info.pose.position.y,
                resolution,
            )

            # --- 5. Publish PointCloud2 message ---
            cloud_msg = self.create_point_cloud_msg(points, msg.header.frame_id)
            self.publisher.publish(cloud_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing GridMap: {e}")
            traceback.print_exc()

    def create_point_cloud_data(
        self, traversability_map, elevation_map, origin_x, origin_y, resolution
    ):
        """Convert traversability and elevation data to point cloud points."""
        rows, cols = traversability_map.shape
        points = []

        half_length_x = (cols * resolution) / 2.0
        half_length_y = (rows * resolution) / 2.0

        for i in range(rows):
            for j in range(cols):
                # FIXED: Reversed the subtraction to correct the 180-degree rotation
                x = origin_x + half_length_x - (j + 0.5) * resolution
                y = origin_y + half_length_y - (i + 0.5) * resolution
                z = elevation_map[i, j]
                traversability = traversability_map[i, j]

                if not np.isnan(z) and not np.isnan(traversability):
                    points.append([x, y, z, traversability])

        return points

    def create_point_cloud_msg(self, points, frame_id):
        """Create a PointCloud2 message from point data."""
        cloud_msg = PointCloud2()
        cloud_msg.header.stamp = self.get_clock().now().to_msg()
        cloud_msg.header.frame_id = frame_id
        cloud_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_msg.point_step = 16
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.is_dense = False
        cloud_msg.data = np.array(points, dtype=np.float32).tobytes()

        return cloud_msg

    def _get_analyzer_params(self, msg: GridMap) -> dict:
        """Gathers parameters from the ROS parameter server."""
        return {
            "device": "cpu" if self.get_parameter("use_cpu").value else wp.get_device(),
            "verbose": self.get_parameter("verbose").value,
            "grid_resolution": msg.info.resolution,
            "smoothing_sigma": self.get_parameter("preprocessing.smoothing_sigma_m").value,
            "slope_normalization_factor": math.radians(
                self.get_parameter("normalization.max_slope_deg").value
            ),
            "step_height_normalization_factor": self.get_parameter(
                "normalization.max_step_height_m"
            ).value,
            "surf_roughness_normalization_factor": self.get_parameter(
                "normalization.max_roughness_m"
            ).value,
            "slope_cost_weight": self.get_parameter("weights.slope").value,
            "step_height_cost_weight": self.get_parameter("weights.step_height").value,
            "surf_roughness_cost_weight": self.get_parameter("weights.surface_roughness").value,
            "roughness_window_radius": self.get_parameter(
                "neighborhood.roughness_window_radius_cells"
            ).value,
        }

    def _get_filter_params(self, msg: GridMap) -> dict:
        """Gathers parameters for the GridMapFilter from the ROS parameter server."""
        return {
            "device": "cpu" if self.get_parameter("use_cpu").value else wp.get_device(),
            "verbose": self.get_parameter("verbose").value,
            "grid_resolution": msg.info.resolution,
            "support_radius": self.get_parameter("filter.support_radius_cells").value,
            "support_ratio": self.get_parameter("filter.support_ratio").value,
        }

    def initialize_filter(self, msg: GridMap):
        """Initializes the GridMapFilter upon receiving the first message."""
        self.get_logger().info("First map received. Initializing GridMapFilter...")
        filter_params = self._get_filter_params(msg)
        self.filter = GridMapFilter(**filter_params)
        self.get_logger().info(f"GridMapFilter initialized on device '{filter_params['device']}'.")

    def initialize_analyzer(self, msg: GridMap):
        """Initializes the TraversabilityAnalyzer upon receiving the first message."""
        self.get_logger().info("First map received. Initializing TraversabilityAnalyzer...")
        analyzer_params = self._get_analyzer_params(msg)
        self.analyzer = GeometricTraversabilityAnalyzer(**analyzer_params)

        resolution = msg.info.resolution
        rows = int(round(msg.info.length_y / resolution))
        cols = int(round(msg.info.length_x / resolution))
        self.get_logger().info(
            f"Analyzer initialized on device '{analyzer_params['device']}' for a {cols}x{rows} grid."
        )


def main(args=None):
    rclpy.init(args=args)
    node = GeometricTraversabilityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
