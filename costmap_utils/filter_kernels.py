import warp as wp


@wp.kernel
def filter_grid(
    elevation_map: wp.array(dtype=wp.float32, ndim=2),
    cost_map: wp.array(dtype=wp.float32, ndim=2),
    grid_height: wp.int32,
    grid_width: wp.int32,
    support_radius: wp.int32,  # in cells
    support_ratio: wp.float32,  # threshold ratio
    # --- Output ---
    filtered_cost: wp.array(dtype=wp.float32, ndim=2),
):
    r, c = wp.tid()
    measured_count = int(0)
    total_count = int(0)

    for dr in range(-support_radius, support_radius + 1):
        for dc in range(-support_radius, support_radius + 1):
            nr = r + dr
            nc = c + dc
            if nr >= 0 and nr < grid_height and nc >= 0 and nc < grid_width:
                val = elevation_map[nr, nc]
                total_count += 1
                if not wp.isnan(val):
                    measured_count += 1

    ratio = 0.0
    if total_count > 0:
        ratio = float(measured_count) / float(total_count)

    if ratio >= support_ratio:
        filtered_cost[r, c] = cost_map[r, c]
    else:
        filtered_cost[r, c] = 0.0 / 0.0  # NaN


@wp.kernel
def filter_box_kernel(
    cost_map: wp.array(dtype=wp.float32, ndim=2),
    grid_height: wp.int32,
    grid_width: wp.int32,
    center_r: wp.int32,  # row coordinate of box center
    center_c: wp.int32,  # column coordinate of box center
    box_half_width: wp.int32,  # half width of box in cells
    box_half_height: wp.int32,  # half height of box in cells
    # --- Output ---
    filtered_cost: wp.array(dtype=wp.float32, ndim=2),
):
    r, c = wp.tid()

    # Check if the current cell is inside the box
    inside_box = (abs(r - center_r) <= box_half_height) and (abs(c - center_c) <= box_half_width)

    if inside_box:
        # If inside the box, set to NaN
        filtered_cost[r, c] = 0.0 / 0.0  # NaN
    else:
        # If outside the box, keep the original value
        filtered_cost[r, c] = cost_map[r, c]


@wp.kernel
def filter_rotated_box_kernel(
    cost_map: wp.array(dtype=wp.float32, ndim=2),
    grid_height: wp.int32,
    grid_width: wp.int32,
    center_r: wp.int32,  # row coordinate of box center
    center_c: wp.int32,  # column coordinate of box center
    box_half_width: wp.int32,  # half width of box in cells
    box_half_height: wp.int32,  # half height of box in cells
    cos_theta: wp.float32,  # cosine of rotation angle
    sin_theta: wp.float32,  # sine of rotation angle
    # --- Output ---
    filtered_cost: wp.array(dtype=wp.float32, ndim=2),
):
    r, c = wp.tid()

    # Translate point relative to box center
    dr = r - center_r
    dc = c - center_c

    # Rotate point to align with box coordinates
    rotated_r = dr * cos_theta + dc * sin_theta
    rotated_c = -dr * sin_theta + dc * cos_theta

    # Check if the rotated point is inside the box
    inside_box = (abs(rotated_r) <= box_half_height) and (abs(rotated_c) <= box_half_width)

    if inside_box:
        # If inside the box, set to NaN
        filtered_cost[r, c] = 0.0 / 0.0  # NaN
    else:
        # If outside the box, keep the original value
        filtered_cost[r, c] = cost_map[r, c]
