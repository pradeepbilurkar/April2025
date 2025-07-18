def compute_drive_lane(polygon, trucks, offset=0.0, fallback_width=1.5):
    from shapely.geometry import box

    if not polygon or not trucks:
        return None

    try:
        lane_width = max(truck["width"] for truck in trucks)
    except Exception:
        lane_width = fallback_width

    entry_x = polygon.centroid.x + offset
    bottom_y = polygon.bounds[1]
    top_y = polygon.bounds[3]

    return box(entry_x - lane_width / 2, bottom_y,
               entry_x + lane_width / 2, top_y)