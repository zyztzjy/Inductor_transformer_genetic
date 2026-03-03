import math
import os
import random

os.environ["HPEESOF_DIR"] = r"D:\Program Files\Keysight\ADS2025_Update1"

import keysight.ads.de as de
from keysight.ads.de import db_uu
from keysight.ads.de.db import LayerId


def create_multiple_spiral_inductors(library: de.Library, num_pairs: int = 10) -> db_uu.Design:
    layout = db_uu.create_layout(f"{library.name}:eee10:layout")

    # 定义图层
    cond = LayerId.create_layer_id_from_library(library, "cond", "drawing")
    cond2 = LayerId.create_layer_id_from_library(library, "cond2", "drawing")
    via = LayerId.create_layer_id_from_library(library, "resi", "drawing")
    metal1 = LayerId.create_layer_id_from_library(library, "Si", "drawing")
    n_implant = LayerId.create_layer_id_from_library(library, "Al", "drawing")

    # 布局基础参数
    pair_spacing_x = 800
    pair_spacing_y = 700
    pairs_per_row = 3
    net = layout.find_or_add_net("signal")
    port_index = 1
    via_size = 4
    fixed_spacing = 2
    # 新增：上层螺旋与下层螺旋之间的最小间隔
    spiral_gap = 10  # 可根据需要调整这个值

    def create_rectangular_spiral(x0, y0, turns, linewidth, direction='cw',
                                  vertical_direction=1, center_gap=50, inner_cut_ratio=0.5):
        """
        生成矩形螺旋并返回路径和边界信息
        新增：返回螺旋的边界框（min_x, max_x, min_y, max_y）
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
                # 上层螺旋从右上角开始（逆时针）
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
                # 下层螺旋从左上角开始（顺时针）
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

        # 计算螺旋的边界框
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        return points, spiral_endpoint, (min_x, max_x, min_y, max_y)

    def create_pgs_region(center_x, center_y, size=150, stripe_width=6, stripe_spacing=10):
        half_size = size / 2
        x0, y0 = center_x - half_size, center_y - half_size  # 区域左上边界
        x1, y1 = center_x + half_size, center_y + half_size  # 区域右下边界
        total_gap = stripe_width + stripe_spacing  # 条带周期（宽度+间距）
        stripe_count = int((size / 2) / total_gap) + 1  # 条带数量
        gap = 6  # 水平条与垂直条之间的间隙（统一控制）

        for i in range(stripe_count-1):
            offset = i * total_gap  # 第i个条带距离中心的偏移量


            # -------------------------- 右上象限 --------------------------
            # 水平条：起点右移（垂直条右侧 + gap），缩短左侧
            h_start_x = center_x + offset + stripe_width + gap
            # 垂直条：起点上移（水平条上侧 + gap），缩短下侧（与水平条对称）
            v_start_y = center_y + offset + stripe_width + gap
            # 水平条有效条件：起点不超过右边界
            if h_start_x <= x1 and (center_y + offset + stripe_width <= y1):
                layout.add_rectangle(metal1,
                                     (h_start_x, center_y + offset),
                                     (x1, center_y + offset + stripe_width))
            # 垂直条有效条件：起点不超过上边界
            if (center_x + offset + stripe_width <= x1) and v_start_y <= y1:
                layout.add_rectangle(metal1,
                                     (center_x + offset, v_start_y),
                                     (center_x + offset + stripe_width, y1))

            # -------------------------- 左上象限（与右上对称）--------------------------
            # 水平条：终点左移（垂直条左侧 - gap），缩短右侧
            h_end_x = center_x - offset - stripe_width - gap
            # 垂直条：起点上移（水平条上侧 + gap），缩短下侧（与水平条对称）
            v_start_y = center_y + offset + stripe_width + gap
            # 水平条有效条件：终点不小于左边界
            if h_end_x >= x0 and (center_y + offset + stripe_width <= y1):
                layout.add_rectangle(metal1,
                                     (x0, center_y + offset),
                                     (h_end_x, center_y + offset + stripe_width))
            # 垂直条有效条件：起点不超过上边界
            if (center_x - offset - stripe_width >= x0) and v_start_y <= y1:
                layout.add_rectangle(metal1,
                                     (center_x - offset - stripe_width, v_start_y),
                                     (center_x - offset, y1))

            # -------------------------- 左下象限（与右上上下对称）--------------------------
            # 水平条：终点左移（垂直条左侧 - gap），缩短右侧
            h_end_x = center_x - offset - stripe_width - gap
            # 垂直条：终点下移（水平条下侧 - gap），缩短上侧（与水平条对称）
            v_end_y = center_y - offset - stripe_width - gap
            # 水平条有效条件：终点不小于左边界
            if h_end_x >= x0 and (center_y - offset - stripe_width >= y0):
                layout.add_rectangle(metal1,
                                     (x0, center_y - offset - stripe_width),
                                     (h_end_x, center_y - offset))
            # 垂直条有效条件：终点不小于下边界
            if (center_x - offset - stripe_width >= x0) and v_end_y >= y0:
                layout.add_rectangle(metal1,
                                     (center_x - offset - stripe_width, y0),
                                     (center_x - offset, v_end_y))

            # -------------------------- 右下象限（与右上左右对称）--------------------------
            # 水平条：起点右移（垂直条右侧 + gap），缩短左侧
            h_start_x = center_x + offset + stripe_width + gap
            # 垂直条：终点下移（水平条下侧 - gap），缩短上侧（与水平条对称）
            v_end_y = center_y - offset - stripe_width - gap
            # 水平条有效条件：起点不超过右边界
            if h_start_x <= x1 and (center_y - offset - stripe_width >= y0):
                layout.add_rectangle(metal1,
                                     (h_start_x, center_y - offset - stripe_width),
                                     (x1, center_y - offset))
            # 垂直条有效条件：终点不小于下边界
            if (center_x + offset + stripe_width <= x1) and v_end_y >= y0:
                layout.add_rectangle(metal1,
                                     (center_x + offset, y0),
                                     (center_x + offset + stripe_width, v_end_y))

        # X形N+掺杂层
        layout.add_path(n_implant, [(x0, y0), (x1, y1)], width=stripe_width * 2)
        layout.add_path(n_implant, [(x0, y1), (x1, y0)], width=stripe_width * 2)

    # 批量生成电感对
    for i in range(num_pairs):
        row = i // pairs_per_row
        col = i % pairs_per_row
        x_offset = col * pair_spacing_x
        y_offset = row * pair_spacing_y

        # 电感参数
        turns_top = random.randint(2, 2)
        turns_bot = 1
        linewidth_top = 5
        linewidth_bot = 5
        center_gap = random.randint(50, 70)
        inner_cut_ratio = 1

        # 电感对中心
        cx = 200 + x_offset
        cy = 300 + y_offset

        # 上层螺旋右上角起点
        p1_start = (cx + 120, cy + 30)

        # 先创建上层螺旋获取其边界
        spiral1_path, spiral1_end, spiral1_bounds = create_rectangular_spiral(
            x0=p1_start[0], y0=p1_start[1],
            turns=turns_top, linewidth=linewidth_top,
            vertical_direction=1,
            center_gap=center_gap, inner_cut_ratio=inner_cut_ratio
        )
        # 上层螺旋的最下边缘y坐标
        spiral1_min_y = spiral1_bounds[2]

        # 计算下层螺旋起点，确保与上层螺旋保持指定间隔
        # 先临时创建下层螺旋获取其边界
        temp_p2_start = (cx+60, cy - 30)
        _, _, temp_spiral2_bounds = create_rectangular_spiral(
            x0=temp_p2_start[0], y0=temp_p2_start[1],
            turns=turns_bot, linewidth=linewidth_bot,
            vertical_direction=-1,
            center_gap=center_gap, inner_cut_ratio=inner_cut_ratio
        )

        # 计算需要调整的y偏移量
        required_offset = (spiral1_min_y - spiral_gap) - temp_spiral2_bounds[3]
        adjusted_p2_start_y = temp_p2_start[1] + required_offset

        # 下层螺旋调整后的起点
        p2_start = (temp_p2_start[0]+10, adjusted_p2_start_y)

        # 创建调整后的下层螺旋
        spiral2_path, spiral2_end, spiral2_bounds = create_rectangular_spiral(
            x0=p2_start[0], y0=p2_start[1],
            turns=turns_bot, linewidth=linewidth_bot,
            vertical_direction=-1,
            center_gap=center_gap, inner_cut_ratio=inner_cut_ratio
        )

        # 绘制螺旋
        layout.add_path(cond, spiral1_path, width=linewidth_top)
        layout.add_path(cond, spiral2_path, width=linewidth_bot)

        # 过孔
        layout.add_rectangle(via, (spiral1_end[0] - via_size / 2, spiral1_end[1] - via_size / 2),
                             (spiral1_end[0] + via_size / 2, spiral1_end[1] + via_size / 2))
        layout.add_rectangle(via, (spiral2_end[0] - via_size / 2, spiral2_end[1] - via_size / 2),
                             (spiral2_end[0] + via_size / 2, spiral2_end[1] + via_size / 2))

        # cond2连接导线
        offset_x = 40
        offset_y = 20
        mid_x = max(spiral1_end[0], spiral2_end[0]) + offset_x
        mid_y = (spiral1_end[1] + spiral2_end[1]) // 2 + offset_y
        cond2_path = [spiral1_end, (mid_x, spiral1_end[1]), (mid_x, mid_y), (mid_x, spiral2_end[1]), spiral2_end]
        layout.add_path(cond2, cond2_path, width=6)

        # PGS屏蔽
        pgs_center_x = (spiral1_end[0] + spiral2_end[0]) / 2
        pgs_center_y = (spiral1_end[1] + spiral2_end[1]) / 2
        # create_pgs_region(pgs_center_x, pgs_center_y)

        # 添加引脚
        layout.add_pin(layout.add_term(net, f"P{port_index}"), layout.add_dot(cond, p1_start), angle=0)
        port_index += 1
        layout.add_pin(layout.add_term(net, f"P{port_index}"), layout.add_dot(cond, p2_start), angle=180)
        port_index += 1

        # cond2中点引脚
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


# 使用示例
if __name__ == "__main__":
    workspace_path = r"C:\Users\HP\Desktop\ADS\MyWorkspace_wrk"
    library_path = r"C:\Users\HP\Desktop\ADS\MyWorkspace_wrk\MyLibrary_lib"
    wrk = de.open_workspace(workspace_path)
    lib = wrk.open_library("MyLibrary_lib", library_path, de.LibraryMode.SHARED)
    created_layout = create_multiple_spiral_inductors(lib, num_pairs=9)