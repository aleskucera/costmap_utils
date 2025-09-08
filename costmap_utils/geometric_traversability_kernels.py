import warp as wp

# Initialize Warp. This is safe to do at the module level.
wp.init()


@wp.kernel
def apply_gaussian_blur_kernel(
    src_heightmap: wp.array(dtype=wp.float32, ndim=2),
    gaussian_kernel: wp.array(dtype=wp.float32, ndim=2),
    kernel_radius: wp.int32,
    grid_height: wp.int32,
    grid_width: wp.int32,
    # --- Output ---
    dst_heightmap: wp.array(dtype=wp.float32, ndim=2),
):
    """Applies a Gaussian blur to a heightmap."""
    r, c = wp.tid()
    weighted_sum = float(0.0)
    weight_total = float(0.0)
    for dr in range(-kernel_radius, kernel_radius + 1):
        for dc in range(-kernel_radius, kernel_radius + 1):
            nr, nc = r + dr, c + dc
            if nr >= 0 and nr < grid_height and nc >= 0 and nc < grid_width:
                height_val = src_heightmap[nr, nc]
                weight = gaussian_kernel[dr + kernel_radius, dc + kernel_radius]
                weighted_sum += height_val * weight
                weight_total += weight
    if weight_total > 0.0:
        dst_heightmap[r, c] = weighted_sum / weight_total
    else:
        dst_heightmap[r, c] = src_heightmap[r, c]


@wp.kernel
def compute_slope_sobel_kernel(
    heightmap: wp.array(dtype=wp.float32, ndim=2),
    resolution: wp.float32,
    grid_height: wp.int32,
    grid_width: wp.int32,
    slope_norm_factor: wp.float32,
    # --- Outputs ---
    normals: wp.array(dtype=wp.vec3, ndim=2),
    slope_cost: wp.array(dtype=wp.float32, ndim=2),
):
    """Computes surface normals and slope cost using a Sobel operator."""
    r, c = wp.tid()
    # Clamp coordinates to handle borders gracefully
    h_tl = heightmap[wp.clamp(r - 1, 0, grid_height - 1), wp.clamp(c - 1, 0, grid_width - 1)]
    h_tm = heightmap[wp.clamp(r - 1, 0, grid_height - 1), c]
    h_tr = heightmap[wp.clamp(r - 1, 0, grid_height - 1), wp.clamp(c + 1, 0, grid_width - 1)]
    h_ml = heightmap[r, wp.clamp(c - 1, 0, grid_width - 1)]
    h_mr = heightmap[r, wp.clamp(c + 1, 0, grid_width - 1)]
    h_bl = heightmap[wp.clamp(r + 1, 0, grid_height - 1), wp.clamp(c - 1, 0, grid_width - 1)]
    h_bm = heightmap[wp.clamp(r + 1, 0, grid_height - 1), c]
    h_br = heightmap[wp.clamp(r + 1, 0, grid_height - 1), wp.clamp(c + 1, 0, grid_width - 1)]

    # Sobel operator for gradients
    dzdx = (h_tr + 2.0 * h_mr + h_br - (h_tl + 2.0 * h_ml + h_bl)) / (8.0 * resolution)
    dzdy = (h_tl + 2.0 * h_tm + h_tr - (h_bl + 2.0 * h_bm + h_br)) / (8.0 * resolution)

    n = wp.normalize(wp.vec3(-dzdx, -dzdy, 1.0))
    normals[r, c] = n

    slope_angle = wp.acos(n[2])
    cost = wp.min(slope_angle / slope_norm_factor, 1.0)
    slope_cost[r, c] = cost


@wp.kernel
def morph_op_kernel(
    src: wp.array(dtype=wp.float32, ndim=2),
    grid_height: wp.int32,
    grid_width: wp.int32,
    radius: wp.int32,
    op: wp.int32,  # 0 for erode, 1 for dilate
    # --- Output ---
    dst: wp.array(dtype=wp.float32, ndim=2),
):
    """Performs morphological erosion (min) or dilation (max) with an arbitrary radius."""
    r, c = wp.tid()
    val = src[r, c]
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nr = wp.clamp(r + dr, 0, grid_height - 1)
            nc = wp.clamp(c + dc, 0, grid_width - 1)
            if op == 0:
                val = wp.min(val, src[nr, nc])  # Erode
            else:
                val = wp.max(val, src[nr, nc])  # Dilate
    dst[r, c] = val


@wp.kernel
def compute_step_height_cost_kernel(
    dilated_map: wp.array(dtype=wp.float32, ndim=2),
    eroded_map: wp.array(dtype=wp.float32, ndim=2),
    step_norm_factor: wp.float32,
    # --- Output ---
    step_height_cost: wp.array(dtype=wp.float32, ndim=2),
):
    """Calculates step height cost based on the difference between dilated and eroded maps."""
    r, c = wp.tid()
    height_diff = dilated_map[r, c] - eroded_map[r, c]
    cost = wp.min(height_diff / step_norm_factor, 1.0)
    step_height_cost[r, c] = cost


@wp.kernel
def compute_roughness_kernel(
    heightmap: wp.array(dtype=wp.float32, ndim=2),
    grid_height: wp.int32,
    grid_width: wp.int32,
    window_radius: wp.int32,
    roughness_norm_factor: wp.float32,
    # --- Output ---
    roughness_cost: wp.array(dtype=wp.float32, ndim=2),
):
    """Calculates surface roughness as the standard deviation of height in a local window."""
    r, c = wp.tid()
    sum_h, sum_sq_h, count = float(0.0), float(0.0), float(0.0)
    for dr in range(-window_radius, window_radius + 1):
        for dc in range(-window_radius, window_radius + 1):
            nr, nc = r + dr, c + dc
            if nr >= 0 and nr < grid_height and nc >= 0 and nc < grid_width:
                h = heightmap[nr, nc]
                sum_h += h
                sum_sq_h += h * h
                count += 1.0

    if count > 1.0:
        mean_h = sum_h / count
        variance = (sum_sq_h / count) - (mean_h * mean_h)
        # Clamp variance to avoid sqrt of negative due to floating point inaccuracies
        if variance < 0.0:
            variance = 0.0
        std_dev = wp.sqrt(variance)
        cost = wp.min(std_dev / roughness_norm_factor, 1.0)
        roughness_cost[r, c] = cost
    else:
        roughness_cost[r, c] = 0.0


@wp.kernel
def combine_costs_kernel(
    slope_cost: wp.array(dtype=wp.float32, ndim=2),
    step_height_cost: wp.array(dtype=wp.float32, ndim=2),
    surf_roughness_cost: wp.array(dtype=wp.float32, ndim=2),
    w_slope: wp.float32,
    w_step: wp.float32,
    w_roughness: wp.float32,
    # --- Output ---
    total_cost: wp.array(dtype=wp.float32, ndim=2),
):
    """Combines multiple cost layers into a single traversability cost map."""
    r, c = wp.tid()
    s = slope_cost[r, c]
    h = step_height_cost[r, c]
    rf = surf_roughness_cost[r, c]

    combined = w_slope * s + w_step * h + w_roughness * rf
    total_weight = w_slope + w_step + w_roughness

    if total_weight > 0.0:
        total_cost[r, c] = combined / total_weight
    else:
        total_cost[r, c] = 0.0
