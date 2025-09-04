#!/usr/bin/env python3
import math
import traceback

import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
import warp as wp
from grid_map_msgs.msg import GridMap
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import MultiArrayLayout

from .geometric_traversability_analyzer import GeometricTraversabilityAnalyzer


class GeometricTraversabilityNode(Node):
    """A ROS 2 Node for GPU-accelerated geometric traversability analysis from GridMaps."""

    def __init__(self):
        super().__init__("geometric_traversability_node")

        # --- Parameters for Configuration ---
        self.declare_parameter("input_topic", "/elevation_mapping_node/elevation_map_filter")
        self.declare_parameter("output_topic", "/traversability_cloud")
        self.declare_parameter("input_layer", "inpaint")
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

        # --- Subscribers and Publishers ---
        self.input_topic = self.get_parameter("input_topic").value
        self.input_layer = self.get_parameter("input_layer").value
        self.output_topic = self.get_parameter("output_topic").value

        self.subscription = self.create_subscription(
            GridMap, self.input_topic, self.map_callback, 10
        )
        # Change publisher to PointCloud2
        self.publisher = self.create_publisher(
            PointCloud2, self.output_topic, qos_profile_system_default
        )

        self.analyzer = None  # Will be initialized on first message
        self.get_logger().info(
            f"Node '{self.get_name()}' initialized. Waiting for GridMap on '{self.input_topic}'..."
        )

    def map_callback(self, msg: GridMap):
        """Callback to process an incoming GridMap message."""
        if self.analyzer is None:
            self.initialize_analyzer(msg)

        try:
            resolution = msg.info.resolution
            rows = int(round(msg.info.length_y / resolution))
            cols = int(round(msg.info.length_x / resolution))

            # --- 1. Extract and Process the Input Layer ---
            if self.input_layer not in msg.layers:
                self.get_logger().error(
                    f"Input layer '{self.input_layer}' not found in GridMap. Available layers: {msg.layers}",
                    throttle_duration_sec=5,
                )
                return

            layer_index = msg.layers.index(self.input_layer)
            layer_data = msg.data[layer_index]

            # Reshape based on column-major ('F'ortran) order used by grid_map
            elevation_np = np.array(layer_data.data, dtype=np.float32).reshape(
                (rows, cols), order="F"
            )

            # --- Handle NaNs for computation ---
            nan_mask = np.isnan(elevation_np)
            heightmap_np = np.copy(elevation_np)

            # Replace NaNs with the minimum valid elevation value for stable computation.
            # If the entire map is NaN, replace with 0.0.
            if np.all(nan_mask):
                fill_value = 0.0
            else:
                fill_value = np.nanmin(elevation_np)
            heightmap_np[nan_mask] = fill_value

            # --- 2. Compute Traversability ---
            results = self.analyzer.compute_traversability(heightmap_np)
            traversability_cost_map = results["traversability_cost"]

            # --- 3. Restore NaNs in the final cost map ---
            traversability_cost_map[nan_mask] = np.nan

            # --- 4. Create point cloud from traversability and elevation data ---
            points = self.create_point_cloud_data(
                traversability_cost_map,
                elevation_np,
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

        for i in range(rows):
            for j in range(cols):
                # Calculate world coordinates
                x = origin_x + (j * resolution)
                y = origin_y + (i * resolution)
                z = elevation_map[i, j]
                traversability = traversability_map[i, j]

                # Skip NaN points
                if not np.isnan(z) and not np.isnan(traversability):
                    # Each point has x, y, z coordinates and traversability as intensity
                    points.append([x, y, z, traversability])

        return points

    def create_point_cloud_msg(self, points, frame_id):
        """Create a PointCloud2 message from point data."""
        # Define fields for the point cloud
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        # Create PointCloud2 message
        cloud_msg = pc2.create_cloud(
            header=rclpy.time.Time().to_msg(), fields=fields, points=points
        )

        # Set the frame ID and timestamp
        cloud_msg.header.frame_id = frame_id
        cloud_msg.header.stamp = self.get_clock().now().to_msg()

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
