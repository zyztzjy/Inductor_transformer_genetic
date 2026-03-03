import math
import os
import random

os.environ["HPEESOF_DIR"] = r"D:\Program Files\Keysight\ADS2025_Update1"

import keysight.ads.de as de
from keysight.ads.de import db_uu
from keysight.ads.de.db import LayerId


def create_multiple_spiral_inductors(library: de.Library, num_pairs: int = 10) -> db_uu.Design:
    layout = db_uu.create_layout(f"{library.name}:eee10:layout")

    cond = LayerId.create_layer_id_from_library(library, "cond", "drawing")
    cond2 = LayerId.create_layer_id_from_library(library, "cond2", "drawing")
    via = LayerId.create_layer_id_from_library(library, "resi", "drawing")
    metal1 = LayerId.create_layer_id_from_library(library, "Si", "drawing")
    n_implant = LayerId.create_layer_id_from_library(library, "Al", "drawing")

    pair_spacing_x = 800
    pair_spacing_y = 700
    pairs_per_row = 3
    net = layout.find_or_add_net("signal")
    port_index = 1
    via_size = 4
    fixed_spacing = 2
    spiral_gap = 10

    def create_rectangular_spiral(x0, y0, turns, linewidth, direction='cw',
                                  vertical_direction=1, center_gap=50, inner_cut_ratio=0.5):
        """
        Generate rectangular spiral and return path and bounding box.
        Returns: points, spiral_endpoint, (min_x, max_x, min_y, max_y)
        """
        points = []
        step = linewidth + fixed_spacing
        center_offset = center_gap / 2
        x, y = x0, y0

        for t in range(turns):
            is_inner = (t == turns - 1)
            base_length = (2 * (turns - t) - 1) * step + center_offset
            if is_inner:
                current_length = base_length * inner_cut_ratio
                inner_offset = (base_length - current_length) / 2
            else:
                current_length = base_length
                inner_offset = 0

            half_right = current_length * 0.75
            start_x = x + inner_offset
            start_y = y

            if vertical_direction == 1:
                points.extend([
                    (start_x, start_y),
                    (start_x - current_length, start_y),
                    (start_x - current_length, start_y - current_length),
                    (start_x, start_y - current_length),
                    (start_x, start_y - step)
                ])
                x -= step
                y -= step

            elif vertical_direction == -1:
                points.extend([
                    (start_x, start_y),
                    (start_x + current_length, start_y),
                    (start_x + current_length, start_y - current_length),
                    (start_x, start_y - current_length),
                    (start_x, start_y - step),
                    (start_x + half_right, start_y - step)
                ])
                x += step
                y -= step

        spiral_endpoint = points[-1]

        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        return points, spiral_endpoint, (min_x, max_x, min_y, max_y)

    def create_pgs_region(center_x, center_y, size=150, stripe_width=6, stripe_spacing=10):
        half_size = size / 2
        x0, y0 = center_x - half_size, center_y - half_size
        x1, y1 = center_x + half_size, center_y + half_size
        total_gap = stripe_width + stripe_spacing
        stripe_count = int((size / 2) / total_gap) + 1
        gap = 6

        for i in range(stripe_count-1):
            offset = i * total_gap

            h_start_x = center_x + offset + stripe_width + gap
            v_start_y = center_y + offset + stripe_width + gap
            if h_start_x <= x1 and (center_y + offset + stripe_width <= y1):
                layout.add_rectangle(metal1,
                                     (h_start_x, center_y + offset),
                                     (x1, center_y + offset + stripe_width))
            if (center_x + offset + stripe_width <= x1) and v_start_y <= y1:
                layout.add_rectangle(metal1,
                                     (center_x + offset, v_start_y),
                                     (center_x + offset + stripe_width, y1))

            h_end_x = center_x - offset - stripe_width - gap
            v_start_y = center_y + offset + stripe_width + gap
            if h_end_x >= x0 and (center_y + offset + stripe_width <= y1):
                layout.add_rectangle(metal1,
                                     (x0, center_y + offset),
                                     (h_end_x, center_y + offset + stripe_width))
            if (center_x - offset - stripe_width >= x0) and v_start_y <= y1:
                layout.add_rectangle(metal1,
                                     (center_x - offset - stripe_width, v_start_y),
                                     (center_x - offset, y1))

            h_end_x = center_x - offset - stripe_width - gap
            v_end_y = center_y - offset - stripe_width - gap
            if h_end_x >= x0 and (center_y - offset - stripe_width >= y0):
                layout.add_rectangle(metal1,
                                     (x0, center_y - offset - stripe_width),
                                     (h_end_x, center_y - offset))
            if (center_x - offset - stripe_width >= x0) and v_end_y >= y0:
                layout.add_rectangle(metal1,
                                     (center_x - offset - stripe_width, y0),
                                     (center_x - offset, v_end_y))

            h_start_x = center_x + offset + stripe_width + gap
            v_end_y = center_y - offset - stripe_width - gap
            if h_start_x <= x1 and (center_y - offset - stripe_width >= y0):
                layout.add_rectangle(metal1,
                                     (h_start_x, center_y - offset - stripe_width),
                                     (x1, center_y - offset))
            if (center_x + offset + stripe_width <= x1) and v_end_y >= y0:
                layout.add_rectangle(metal1,
                                     (center_x + offset, y0),
                                     (center_x + offset + stripe_width, v_end_y))

        layout.add_path(n_implant, [(x0, y0), (x1, y1)], width=stripe_width * 2)
        layout.add_path(n_implant, [(x0, y1), (x1, y0)], width=stripe_width * 2)

    for i in range(num_pairs):
        row = i // pairs_per_row
        col = i % pairs_per_row
        x_offset = col * pair_spacing_x
        y_offset = row * pair_spacing_y

        turns_top = random.randint(2, 2)
        turns_bot = 1
        linewidth_top = 5
        linewidth_bot = 5
        center_gap = random.randint(50, 70)
        inner_cut_ratio = 1

        cx = 200 + x_offset
        cy = 300 + y_offset

        p1_start = (cx + 120, cy + 30)

        spiral1_path, spiral1_end, spiral1_bounds = create_rectangular_spiral(
            x0=p1_start[0], y0=p1_start[1],
            turns=turns_top, linewidth=linewidth_top,
            vertical_direction=1,
            center_gap=center_gap, inner_cut_ratio=inner_cut_ratio
        )
        spiral1_min_y = spiral1_bounds[2]

        temp_p2_start = (cx+60, cy - 30)
        _, _, temp_spiral2_bounds = create_rectangular_spiral(
            x0=temp_p2_start[0], y0=temp_p2_start[1],
            turns=turns_bot, linewidth=linewidth_bot,
            vertical_direction=-1,
            center_gap=center_gap, inner_cut_ratio=inner_cut_ratio
        )

        required_offset = (spiral1_min_y - spiral_gap) - temp_spiral2_bounds[3]
        adjusted_p2_start_y = temp_p2_start[1] + required_offset

        p2_start = (temp_p2_start[0]+10, adjusted_p2_start_y)

        spiral2_path, spiral2_end, spiral2_bounds = create_rectangular_spiral(
            x0=p2_start[0], y0=p2_start[1],
            turns=turns_bot, linewidth=linewidth_bot,
            vertical_direction=-1,
            center_gap=center_gap, inner_cut_ratio=inner_cut_ratio
        )

        layout.add_path(cond, spiral1_path, width=linewidth_top)
        layout.add_path(cond, spiral2_path, width=linewidth_bot)

        layout.add_rectangle(via, (spiral1_end[0] - via_size / 2, spiral1_end[1] - via_size / 2),
                             (spiral1_end[0] + via_size / 2, spiral1_end[1] + via_size / 2))
        layout.add_rectangle(via, (spiral2_end[0] - via_size / 2, spiral2_end[1] - via_size / 2),
                             (spiral2_end[0] + via_size / 2, spiral2_end[1] + via_size / 2))

        offset_x = 40
        offset_y = 20
        mid_x = max(spiral1_end[0], spiral2_end[0]) + offset_x
        mid_y = (spiral1_end[1] + spiral2_end[1]) // 2 + offset_y
        cond2_path = [spiral1_end, (mid_x, spiral1_end[1]), (mid_x, mid_y), (mid_x, spiral2_end[1]), spiral2_end]
        layout.add_path(cond2, cond2_path, width=6)

        pgs_center_x = (spiral1_end[0] + spiral2_end[0]) / 2
        pgs_center_y = (spiral1_end[1] + spiral2_end[1]) / 2

        layout.add_pin(layout.add_term(net, f"P{port_index}"), layout.add_dot(cond, p1_start), angle=0)
        port_index += 1
        layout.add_pin(layout.add_term(net, f"P{port_index}"), layout.add_dot(cond, p2_start), angle=180)
        port_index += 1

        total_length = 0
        segments = []
        for j in range(len(cond2_path) - 1):
            x1, y1 = cond2_path[j]
            x2, y2 = cond2_path[j + 1]
            seg_len = math.hypot(x2 - x1, y2 - y1)
            segments.append((x1, y1, x2, y2, seg_len))
            total_length += seg_len

        current_len = 0
        midpoint = None
        for x1, y1, x2, y2, seg_len in segments:
            if current_len + seg_len >= total_length / 2:
                ratio = (total_length / 2 - current_len) / seg_len
                midpoint = (x1 + (x2 - x1) * ratio, y1 + (y2 - y1) * ratio)
                break
            current_len += seg_len
        if midpoint:
            layout.add_pin(layout.add_term(net, f"P{port_index}"), layout.add_dot(cond2, midpoint), angle=180)
        port_index += 1

    layout.save_design()
    return layout


if __name__ == "__main__":
    workspace_path = r"C:\Users\HP\Desktop\ADS\MyWorkspace_wrk"
    library_path = r"C:\Users\HP\Desktop\ADS\MyWorkspace_wrk\MyLibrary_lib"
    wrk = de.open_workspace(workspace_path)
    lib = wrk.open_library("MyLibrary_lib", library_path, de.LibraryMode.SHARED)
    created_layout = create_multiple_spiral_inductors(lib, num_pairs=9)
