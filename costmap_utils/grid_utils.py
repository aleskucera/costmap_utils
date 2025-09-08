import numpy as np
from grid_map_msgs.msg import GridMap


def meters_to_cells(distance_m: float, grid_resolution: float) -> int:
    """
    Converts a distance from meters to the nearest integer number of grid cells.

    Args:
        distance_m: The distance in meters.
        grid_resolution: The size of one grid cell in meters.

    Returns:
        The corresponding number of grid cells, rounded to the nearest integer.
    """
    if grid_resolution <= 0:
        return 0
    return int(round(distance_m / grid_resolution))


def cells_to_meters(distance_cells: int, grid_resolution: float) -> float:
    """
    Converts a distance from grid cells to meters.

    Args:
        distance_cells: The number of grid cells.
        grid_resolution: The size of one grid cell in meters.

    Returns:
        The corresponding distance in meters.
    """
    return float(distance_cells * grid_resolution)


def extract_layer(msg: GridMap, layer_name: str):
    resolution = msg.info.resolution
    rows = int(round(msg.info.length_y / resolution))
    cols = int(round(msg.info.length_x / resolution))

    layer_idx = msg.layers.index(layer_name)
    layer_data = msg.data[layer_idx].data
    heightmap_flat = np.array(layer_data, dtype=np.float32)
    heightmap = heightmap_flat.reshape((rows, cols), order="C")
    return heightmap
