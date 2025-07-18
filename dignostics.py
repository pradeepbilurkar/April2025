

from shapely.geometry import box

def generate_layout_feedback(polygon, placed_trucks, reserved_lane, clearance=0.5):
    feedback = []

    for idx, truck in enumerate(placed_trucks):
        tid, ttype, width, length, x, y = truck  # 👈 unpack the tuple

        full_box = box(x - clearance / 2, y - clearance / 2,
                       x + width + clearance / 2, y + length + clearance / 2)

        # Drive lane violation
        if reserved_lane and reserved_lane.intersects(full_box):
            feedback.append(f"⚠️ Truck {tid} intrudes into the reserved flow lane.")

        # Overlap with other trucks
        for other in placed_trucks:
            if other == truck:
                continue
            otid, _, owidth, olength, ox, oy = other
            other_box = box(ox - clearance / 2, oy - clearance / 2,
                            ox + owidth + clearance / 2, oy + olength + clearance / 2)
            if full_box.intersects(other_box):
                feedback.append(f"❌ Truck {tid} overlaps with Truck {otid}.")
                break

    if not feedback:
        feedback.append("✅ No violations detected. Layout looks clean!")

    return feedback