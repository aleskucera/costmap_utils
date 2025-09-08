#!/usr/bin/env python3
import math
import traceback

import numpy as np
import rclpy
import tf2_ros
import warp as wp
from geometry_msgs.msg import TransformStamped
from grid_map_msgs.msg import GridMap
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField

from .geometric_traversability_analyzer import GeometricTraversabilityAnalyzer
from .grid_map_filter import GridMapFilter
from .grid_utils import extract_layer
from .grid_utils import meters_to_cells


def create_traversability_cloud_data(
    traversability_map: np.ndarray,
    elevation_map: np.ndarray,
    origin_x,
    origin_y,
    resolution: float,
):
    rows, cols = traversability_map.shape
    points = []

    half_length_x = (cols * resolution) / 2.0
    half_length_y = (rows * resolution) / 2.0

    for r in range(rows):
        for c in range(cols):
            x = origin_x + half_length_x - (c + 0.5) * resolution
            y = origin_y + half_length_y - (r + 0.5) * resolution
            z = elevation_map[r, c]
            traversability = traversability_map[r, c]

            if not np.isnan(z) and not np.isnan(traversability):
                points.append([x, y, z, traversability])

    return points


class GeometricTraversabilityNode(Node):
    def __init__(self):
        super().__init__("geometric_traversability_node")

        # --- Parameters for Configuration ---
        self.declare_parameter("input_topic", "/elevation_mapping_node/elevation_map_filter")
        self.declare_parameter("output_topic", "/geometric_traversability_cloud")
        self.declare_parameter("traversability_input_layer", "inpaint")
        self.declare_parameter("use_cpu", False)
        self.declare_parameter("verbose", False)
        self.declare_parameter("base_frame", "base_link")

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

        # Neighborhood parameters (in meters)
        self.declare_parameter("neighborhood.step_window_radius_m", 0.1)
        self.declare_parameter("neighborhood.roughness_window_radius_m", 0.1)

        # --- Added parameters for the new GridMapFilter ---
        self.declare_parameter("filter.enabled", True)
        self.declare_parameter("filter.raw_elevation_layer", "elevation")
        self.declare_parameter("filter.support_radius_m", 0.1)
        self.declare_parameter("filter.support_ratio", 0.75)
        self.declare_parameter("filter.inflation_radius_m", 0.3)
        self.declare_parameter("filter.obstacle_threshold", 0.8)

        # --- TF2 Listener ---
        self.get_logger().info("Initializing TF2 buffer and listener...")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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

        if self.get_parameter("use_cpu").value:
            wp.set_device("cpu")

    def map_callback(self, msg: GridMap):
        """Callback to process an incoming GridMap message."""
        if self.analyzer is None:
            self.initialize_analyzer(msg)

        filter_enabled = self.get_parameter("filter.enabled").value
        if filter_enabled and self.filter is None:
            self.initialize_filter(msg)

        try:
            # --- Get Transform ---
            base_frame = self.get_parameter("base_frame").value
            map_frame = msg.header.frame_id
            transform = None
            try:
                transform = self.tf_buffer.lookup_transform(
                    base_frame,  # Target frame
                    map_frame,  # Source frame
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5),
                )
            except tf2_ros.TransformException as ex:
                self.get_logger().error(
                    f"Could not transform '{map_frame}' to '{base_frame}': {ex}",
                    throttle_duration_sec=5,
                )
                return

            # --- 1. Extract and Process Layers ---
            traversability_layer_name = self.get_parameter("traversability_input_layer").value
            if traversability_layer_name not in msg.layers:
                self.get_logger().error(
                    f"Traversability input layer '{traversability_layer_name}' "
                    f"not found. Available: {msg.layers}",
                    throttle_duration_sec=5,
                )
                return

            # --- 2. Compute Traversability ---
            inpainted_elevation = extract_layer(msg, traversability_layer_name)
            cost_map = self.analyzer.compute_traversability(inpainted_elevation)

            # --- 3. Apply Reliability Filter (Optional) ---
            if filter_enabled:
                raw_elevation_layer_name = self.get_parameter("filter.raw_elevation_layer").value
                if raw_elevation_layer_name not in msg.layers:
                    self.get_logger().error(
                        f"Raw elevation layer '{raw_elevation_layer_name}' for "
                        f"filtering not found. Available: {msg.layers}",
                        throttle_duration_sec=5,
                    )
                    return

                raw_elevation = extract_layer(msg, raw_elevation_layer_name)

                # Apply the filter
                cost_map = self.filter.apply_filters(
                    raw_elevation,
                    cost_map,
                    transform,
                )

            # --- 4. Create point cloud from traversability and elevation data ---
            points = create_traversability_cloud_data(
                cost_map,
                inpainted_elevation,
                msg.info.pose.position.x,
                msg.info.pose.position.y,
                msg.info.resolution,
            )

            # --- 5. Publish PointCloud2 message ---
            cloud_msg = self.create_point_cloud_msg(points, msg.header.frame_id)
            self.publisher.publish(cloud_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing GridMap: {e}")
            traceback.print_exc()

    def create_point_cloud_msg(self, points, frame_id):
        """Create a PointCloud2 message from point data."""
        cloud_msg = PointCloud2()
        cloud_msg.header.stamp = self.get_clock().now().to_msg()
        cloud_msg.header.frame_id = frame_id
        cloud_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="geometric_cost", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_msg.point_step = 16
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.is_dense = False
        cloud_msg.data = np.array(points, dtype=np.float32).tobytes()

        return cloud_msg

    def initialize_filter(self, msg: GridMap):
        self.get_logger().info("First map received. Initializing GridMapFilter...")

        verbose = self.get_parameter("verbose").value

        grid_resolution = msg.info.resolution
        grid_height = meters_to_cells(msg.info.length_y, grid_resolution)
        grid_width = meters_to_cells(msg.info.length_x, grid_resolution)

        support_radius_m = self.get_parameter("filter.support_radius_m").value
        support_ratio = self.get_parameter("filter.support_ratio").value

        inflation_radius_m = self.get_parameter("filter.inflation_radius_m").value
        obstacle_threshold = self.get_parameter("filter.obstacle_threshold").value

        self.filter = GridMapFilter(
            device=wp.get_device(),
            verbose=verbose,
            grid_resolution=grid_resolution,
            grid_height=grid_height,
            grid_width=grid_width,
            support_radius_m=support_radius_m,
            support_ratio=support_ratio,
            inflation_radius_m=inflation_radius_m,
            obstacle_threshold=obstacle_threshold,
        )

        self.get_logger().info(f"GridMapFilter initialized on device '{wp.get_device()}'.")

    def initialize_analyzer(self, msg: GridMap):
        self.get_logger().info("First map received. Initializing TraversabilityAnalyzer...")

        verbose = self.get_parameter("verbose").value

        grid_resolution = msg.info.resolution
        grid_height = meters_to_cells(msg.info.length_y, grid_resolution)
        grid_width = meters_to_cells(msg.info.length_x, grid_resolution)

        smoothing_sigma_m = self.get_parameter("preprocessing.smoothing_sigma_m").value
        step_window_radius_m = self.get_parameter("neighborhood.step_window_radius_m").value
        roughness_window_radius_m = self.get_parameter(
            "neighborhood.roughness_window_radius_m"
        ).value

        max_slope_rad = math.radians(self.get_parameter("normalization.max_slope_deg").value)
        max_step_height_m = self.get_parameter("normalization.max_step_height_m").value
        max_roughness_m = self.get_parameter("normalization.max_roughness_m").value

        slope_cost_weight = self.get_parameter("weights.slope").value
        step_height_cost_weight = self.get_parameter("weights.step_height").value
        surf_roughness_cost_weight = self.get_parameter("weights.surface_roughness").value

        self.analyzer = GeometricTraversabilityAnalyzer(
            device=wp.get_device(),
            verbose=verbose,
            grid_resolution=grid_resolution,
            grid_height=grid_height,
            grid_width=grid_width,
            smoothing_sigma_m=smoothing_sigma_m,
            step_window_radius_m=step_window_radius_m,
            roughness_window_radius_m=roughness_window_radius_m,
            max_slope_rad=max_slope_rad,
            max_step_height_m=max_step_height_m,
            max_roughness_m=max_roughness_m,
            slope_cost_weight=slope_cost_weight,
            step_height_cost_weight=step_height_cost_weight,
            surf_roughness_cost_weight=surf_roughness_cost_weight,
        )

        self.get_logger().info(
            f"Analyzer initialized on device '{wp.get_device()}' for a {grid_height}x{grid_width} grid."
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
