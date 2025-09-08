from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    """
    Launch file for the Geometric Traversability Node.

    This launch file starts the geometric_traversability_node and allows for the
    configuration of all its parameters. These parameters control everything from
    input/output topics to the weights and thresholds used in the traversability
    calculation.
    """

    geometric_traversability_node = Node(
        package="costmap_utils",
        executable="geometric_traversability_node",
        name="geometric_traversability_node",
        output="screen",
        parameters=[
            {
                # --- Main Configuration ---
                "input_topic": "/elevation_mapping_node/elevation_map_filter",
                "output_topic": "/geometric_traversability_cloud",
                "traversability_input_layer": "inpaint",  # Layer from GridMap to use for analysis
                "use_cpu": False,  # Set to True to force CPU execution, otherwise uses GPU if available
                "verbose": False,  # Set to True for extra debug prints from the analyzer
                # --- Cost Function Weights ---
                # These should sum to 1.0
                "weights.slope": 0.4,
                "weights.step_height": 0.4,
                "weights.surface_roughness": 0.2,
                # --- Pre-processing Parameters ---
                # Gaussian smoothing applied to the input elevation map before analysis
                "preprocessing.smoothing_sigma_m": 0.08,
                # --- Normalization Thresholds ---
                # Values above these thresholds will receive the maximum cost for that metric.
                "normalization.max_slope_deg": 70.0,
                "normalization.max_step_height_m": 0.4,
                "normalization.max_roughness_m": 0.2,
                # --- Neighborhood Parameters ---
                # Window size for calculating surface roughness
                "neighborhood.roughness_window_radius_m": 0.1,
                # --- Reliability Filter ---
                # This filter can invalidate costs in areas with sparse raw elevation data.
                "filter.enabled": True,
                "filter.raw_elevation_layer": "elevation",  # The unfiltered elevation layer
                "filter.support_radius_m": 0.1,  # Radius to check for supporting points
                "filter.support_ratio": 0.75,  # Required ratio of valid points in the radius
            }
        ],
    )

    return LaunchDescription([geometric_traversability_node])
