import warp as wp


@wp.kernel
def filter_grid(
    elevation_map: wp.array(dtype=wp.float32, ndim=2),
    cost_map: wp.array(dtype=wp.float32, ndim=2),
    map_height: wp.int32,
    map_width: wp.int32,
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
            if nr >= 0 and nr < map_height and nc >= 0 and nc < map_width:
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
def inflate_obstacles_kernel(
    cost_map: wp.array(dtype=wp.float32, ndim=2),
    map_height: wp.int32,
    map_width: wp.int32,
    inflation_radius: wp.int32,
    obstacle_threshold: wp.float32,
    # -- Output ---
    inflated_cost_map: wp.array(dtype=wp.float32, ndim=2),
):
    r, c = wp.tid()
    inflated_cost = cost_map[r, c]

    for dr in range(-inflation_radius, inflation_radius + 1):
        for dc in range(-inflation_radius, inflation_radius + 1):
            nr = r + dr
            nc = c + dc
            if nr >= 0 and nr < map_height and nc >= 0 and nc < map_width:
                cost_val = cost_map[nr, nc]
                if not wp.isnan(cost_val) and cost_val > obstacle_threshold:
                    inflated_cost = 1000.0

    inflated_cost_map[r, c] = inflated_cost
