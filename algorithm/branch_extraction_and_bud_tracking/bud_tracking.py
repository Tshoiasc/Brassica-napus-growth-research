import heapq
import logging

from numba import jit
import numpy as np
from queue import PriorityQueue
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean


@jit(nopython=True)
def calculate_spatial_cost(current_relation, previous_relation, horizontal_weight, vertical_weight):
    return np.sqrt(
        (horizontal_weight * (current_relation[0] - previous_relation[0])) ** 2 +
        (vertical_weight * (current_relation[1] - previous_relation[1])) ** 2
    )


@jit(nopython=True)
def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


@jit(nopython=True)
def get_neighbors(point, skeleton):
    x, y = point
    neighbors = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < skeleton.shape[0] and 0 <= ny < skeleton.shape[1]:
            if np.any(skeleton[max(0, nx - 1):min(skeleton.shape[0], nx + 2),
                      max(0, ny - 1):min(skeleton.shape[1], ny + 2)] > 0):
                neighbors.append((nx, ny))
    return neighbors


@jit(nopython=True)
def a_star_search(start, goal, skeleton):
    def heuristic(a, b):
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))

    open_heap = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    closed_set = set()

    while open_heap:
        current = heapq.heappop(open_heap)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        closed_set.add(current)

        for neighbor in get_neighbors(current, skeleton):
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f = g_score[neighbor] + heuristic(neighbor, goal)
                if neighbor not in f_score:
                    heapq.heappush(open_heap, (int(f), neighbor))  # Convert f to int
                f_score[neighbor] = f

    # If no path is found, return a straight line
    return line(start[0], start[1], goal[0], goal[1])

@jit(nopython=True)
def line(x1, y1, x2, y2):
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return points


class BudTracker:
    def __init__(self, total_frames):
        self.previous_buds = None
        self.previous_paths = None
        self.previous_top_index = None
        self.all_used_ids = set([0])  # 用于跟踪所有历史帧中使用过的ID
        self.id_history = {0: []}  # 记录每个ID的历史位置
        self.current_frame = 0
        self.max_id = 0
        self.total_frames = total_frames  # 添加总帧数属性
        self.all_historical_ids = set()  # 添加所有历史ID属性

    def post_process_trajectories(self, max_bud_count):
        trajectories = self.get_bud_trajectories()
        logging.debug(f"Original trajectories: {trajectories.keys()}")

        merged_trajectories = {}
        id_mapping = {id: id for id in self.all_historical_ids}

        # 合并轨迹
        for id1, traj1 in sorted(trajectories.items(), key=lambda x: len(x[1]), reverse=True):
            if id1 in merged_trajectories:
                continue
            merged_traj = traj1
            merged_ids = [id1]
            for id2, traj2 in trajectories.items():
                if id2 != id1 and id2 not in merged_trajectories:
                    if self.can_merge_trajectories(merged_traj, traj2):
                        merged_traj = self.merge_trajectories(merged_traj, traj2)
                        merged_ids.append(id2)
            new_id = min(merged_ids)
            merged_trajectories[new_id] = merged_traj
            for old_id in merged_ids:
                id_mapping[old_id] = new_id

        logging.debug(f"Merged trajectories: {merged_trajectories.keys()}")
        logging.debug(f"ID mapping after merging: {id_mapping}")

        # 保留短轨迹，但标记它们
        short_trajectories = {id: traj for id, traj in merged_trajectories.items() if len(traj) < 3}
        long_trajectories = {id: traj for id, traj in merged_trajectories.items() if len(traj) >= 3}

        # 限制轨迹数量，但保留所有短轨迹
        final_trajectories = dict(
            sorted(long_trajectories.items(), key=lambda x: len(x[1]), reverse=True)[:max_bud_count])
        final_trajectories.update(short_trajectories)

        # 重新编号轨迹
        renumbered_trajectories = {0: final_trajectories.get(0, [])}
        final_id_mapping = {0: 0}
        new_id = 1
        for old_id, traj in sorted(final_trajectories.items(), key=lambda x: len(x[1]), reverse=True):
            if old_id != 0:
                renumbered_trajectories[new_id] = traj
                final_id_mapping[old_id] = new_id
                new_id += 1

        # 更新最终的ID映射
        for old_id in self.all_historical_ids:
            if old_id in id_mapping:
                id_mapping[old_id] = final_id_mapping.get(id_mapping[old_id], id_mapping[old_id])
            else:
                id_mapping[old_id] = old_id  # 保留未合并的ID

        logging.debug(f"Final trajectories: {renumbered_trajectories.keys()}")
        logging.debug(f"Final ID mapping: {id_mapping}")

        self.id_history = renumbered_trajectories
        self.all_used_ids = set(renumbered_trajectories.keys())

        return renumbered_trajectories, id_mapping

    def renumber_trajectories(self, trajectories):
        renumbered = {0: trajectories.get(0, [])}
        id_mapping = {0: 0}  # 新增：用于存储旧ID到新ID的映射

        new_id = 1
        for old_id, traj in sorted(trajectories.items(), key=lambda x: len(x[1]), reverse=True):
            if old_id != 0:
                renumbered[new_id] = traj
                id_mapping[old_id] = new_id
                new_id += 1

        logging.debug("轨迹重新编号完成。新的ID映射：")
        for new_id, traj in renumbered.items():
            logging.debug(f"新ID {new_id}: 轨迹长度 {len(traj)}")

        return renumbered, id_mapping

    def can_merge_trajectories(self, traj1, traj2):
        if not traj1 or not traj2:
            return False

        last_point1 = traj1[-1]
        first_point2 = traj2[0]

        # 检查时间上是否连续
        time_gap = first_point2[2] - last_point1[2]
        if time_gap != 1:  # 只允许连续的帧
            return False

        # 检查空间上是否接近
        distance = euclidean(last_point1[:2], first_point2[:2])
        max_distance = 30  # 减小最大距离阈值

        return distance < max_distance

    def merge_trajectories(self, traj1, traj2):
        # 合并两个轨迹
        return traj1 + traj2

    def get_weight_factor(self):
        # 使用sigmoid函数来实现平滑过渡
        x = self.current_frame / self.total_frames
        return 1 / (1 + np.exp(-10 * (x - 0.5)))

    def get_spatial_relationship(self, bud, top_flower):
        diff = np.array(bud) - np.array(top_flower)
        return np.array([diff[0], diff[1]])  # 返回 [水平差, 垂直差]

    def a_star_search(self, start, goal, skeleton):
        def heuristic(a, b):
            return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

        def get_neighbors(point):
            x, y = point
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < skeleton.shape[0] and 0 <= ny < skeleton.shape[1]:
                    if np.any(skeleton[max(0, nx - 1):min(skeleton.shape[0], nx + 2),
                              max(0, ny - 1):min(skeleton.shape[1], ny + 2)] > 0):
                        neighbors.append((nx, ny))
            return neighbors

        start = tuple(map(int, start))
        goal = tuple(map(int, goal))

        open_heap = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        closed_set = set()

        while open_heap:
            current = heapq.heappop(open_heap)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            closed_set.add(current)

            for neighbor in get_neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f = g_score[neighbor] + heuristic(neighbor, goal)
                    if neighbor not in f_score:
                        heapq.heappush(open_heap, (f, neighbor))
                    f_score[neighbor] = f

        # 如果没有找到路径，返回直线路径
        return np.array(list(zip(*self.line(start[0], start[1], goal[0], goal[1]))))

    @staticmethod
    def line(x1, y1, x2, y2):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            yield (x1, y1)
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

    def calculate_cost(self, current_bud, current_path, previous_bud, previous_path, current_top_flower,
                       previous_top_flower, skeleton):
        # 1. 空间关系代价
        current_relation = self.get_spatial_relationship(current_bud, current_top_flower)
        previous_relation = self.get_spatial_relationship(previous_bud, previous_top_flower)

        # 为水平和垂直方向设置不同的权重
        horizontal_weight = 1.0
        vertical_weight = 2.0  # 假设垂直关系更重要

        spatial_cost = calculate_spatial_cost(current_relation, previous_relation, horizontal_weight, vertical_weight)

        # 3. A*寻路代价
        a_star_path = a_star_search(
            (int(previous_bud[1]), int(previous_bud[0])),
            (int(current_bud[1]), int(current_bud[0])),
            skeleton
        )
        a_star_cost = len(a_star_path)

        # 4. 垂直移动惩罚
        vertical_penalty = max(0, current_bud[1] - previous_bud[1]) * 20  # 增加垂直惩罚

        # 5. 跨界移动惩罚
        cross_boundary_penalty = 0
        if (current_bud[0] - current_top_flower[0]) * (previous_bud[0] - previous_top_flower[0]) < 0:
            logging.debug("Cross boundary!", current_bud, previous_bud)
            cross_boundary_penalty = 100  # 大幅增加跨界惩罚

        # 获取动态权重因子
        weight_factor = self.get_weight_factor()

        # 动态调整空间代价和A*代价的权重
        spatial_weight = (1 - weight_factor)
        a_star_weight = weight_factor
        logging.debug(
            f"Weight factor: {weight_factor:.2f}, spatial weight: {spatial_weight:.2f}, A* weight: {a_star_weight:.2f}")
        total_cost = (
                spatial_cost * spatial_weight * 2 +  # 增加空间关系的权重
                a_star_cost * a_star_weight * 3 +
                vertical_penalty * 3 +
                cross_boundary_penalty * 5
        )

        logging.debug(f"Cost details: spatial={spatial_cost:.2f}, A*={a_star_cost}, "
              f"vertical={vertical_penalty}, cross={cross_boundary_penalty},total={total_cost:.2f}")

        return total_cost

    def match_buds(self, current_buds, current_paths, top_flower_index, skeleton):
        self.current_frame += 1
        current_bud_count = len(current_buds)

        logging.debug(f"\n{'=' * 50}")
        logging.debug(f"处理第 {self.current_frame} 帧")
        logging.debug(f"{'=' * 50}")
        logging.debug(f"当前帧花蕾数量: {current_bud_count}")

        # 确保顶花始终为0号
        if top_flower_index != 0:
            current_buds[0], current_buds[top_flower_index] = current_buds[top_flower_index], current_buds[0]
            current_paths[0], current_paths[top_flower_index] = current_paths[top_flower_index], current_paths[0]
            top_flower_index = 0

        # 获取权重因子并打印
        weight_factor = self.get_weight_factor()
        spatial_weight = 2 * (1 - weight_factor)
        a_star_weight = 3 * weight_factor
        logging.debug(
            f"Weight factor: {weight_factor:.2f}, spatial weight: {spatial_weight:.2f}, A* weight: {a_star_weight:.2f}")

        if self.previous_buds is None or len(self.previous_buds) == 0:
            matched_ids = list(range(current_bud_count))
            self.all_used_ids.update(matched_ids)
            for id, bud in zip(matched_ids, current_buds):
                self.id_history[id] = [bud]
            self.max_id = max(matched_ids)
            self.previous_buds = current_buds
            self.previous_paths = current_paths
            self.previous_top_index = top_flower_index
            self.bud_ids = matched_ids
            logging.debug("初始化帧，分配ID:", matched_ids)
            return matched_ids

        logging.debug(f"上一帧花蕾数量: {len(self.previous_buds)}")
        logging.debug(f"上一帧顶花索引: {self.previous_top_index}")
        logging.debug(f"当前帧顶花索引: {top_flower_index}")

        current_buds_without_top = [bud for i, bud in enumerate(current_buds) if i != top_flower_index]
        current_paths_without_top = [path for i, path in enumerate(current_paths) if i != top_flower_index]

        matched_ids = [0] * current_bud_count
        # 确保顶花ID为0
        matched_ids[0] = 0
        self.id_history[0].append(current_buds[0])

        assigned_ids = set([0])  # 跟踪此帧中已分配的ID

        # 创建上一帧ID到花蕾的映射，排除顶花
        previous_id_to_bud = {self.bud_ids[i]: bud for i, bud in enumerate(self.previous_buds) if
                              i != self.previous_top_index}

        logging.debug("\n计算代价矩阵...")
        if len(current_buds_without_top) == 0 or len(previous_id_to_bud) == 0:
            logging.debug("当前帧或上一帧没有非顶花花蕾，跳过匹配过程")
            for i in range(current_bud_count):
                if i != top_flower_index:
                    new_id = self.get_new_id()
                    matched_ids[i] = new_id
                    self.all_used_ids.add(new_id)
                    self.id_history[new_id] = [current_buds[i]]
                    logging.debug(f"  花蕾 {i} 分配新ID: {new_id}")
        else:
            cost_matrix = np.zeros((len(current_buds_without_top), len(previous_id_to_bud)))
            for i, (curr_bud, curr_path) in enumerate(zip(current_buds_without_top, current_paths_without_top)):
                curr_original_index = [k for k, bud in enumerate(current_buds) if bud == curr_bud][0]
                for j, (prev_id, prev_bud) in enumerate(previous_id_to_bud.items()):
                    prev_path = self.previous_paths[self.bud_ids.index(prev_id)]
                    cost_matrix[i, j] = self.calculate_cost(curr_bud, curr_path, prev_bud, prev_path,
                                                            current_buds[top_flower_index],
                                                            self.previous_buds[self.previous_top_index],
                                                            skeleton)
                    logging.debug(f"  当前花蕾 {curr_original_index} 与上一帧花蕾ID {prev_id} 的代价: {cost_matrix[i, j]:.2f}")

            logging.debug("\n使用匈牙利算法进行匹配...")
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            if cost_matrix.size > 0:
                min_cost = np.min(cost_matrix)
                cost_threshold = min(1000, max(200, min_cost * 10))
            else:
                cost_threshold = 1000
            logging.debug(f"代价阈值: {cost_threshold}")

            logging.debug("\n匹配结果:")
            matched_ids = [-1] * current_bud_count  # 初始化为-1，而不是0
            matched_ids[top_flower_index] = 0  # 确保顶花ID为0
            assigned_ids = set([0])

            # 初始匹配
            for i, j in zip(row_ind, col_ind):
                curr_original_index = [k for k, bud in enumerate(current_buds) if bud == current_buds_without_top[i]][0]
                prev_id = list(previous_id_to_bud.keys())[j]
                cost = cost_matrix[i, j]
                if cost < cost_threshold:
                    matched_ids[curr_original_index] = prev_id
                    assigned_ids.add(prev_id)
                    self.all_used_ids.add(prev_id)  # 确保更新 all_used_ids
                    if prev_id not in self.id_history:
                        self.id_history[prev_id] = []
                    self.id_history[prev_id].append(current_buds[curr_original_index])  # 更新 id_history
                    logging.debug(f"  当前花蕾 {curr_original_index} 初始匹配上一帧花蕾ID {prev_id}, 代价 {cost:.2f}")
            # 后处理
            for i, curr_bud in enumerate(current_buds):
                if i != top_flower_index:
                    if matched_ids[i] == -1:  # 未匹配的花蕾
                        best_cost = float('inf')
                        best_id = None
                        for j, prev_id in enumerate(previous_id_to_bud.keys()):
                            if prev_id not in assigned_ids:
                                cost = cost_matrix[current_buds_without_top.index(curr_bud), j]
                                if cost < best_cost and cost < cost_threshold:
                                    best_cost = cost
                                    best_id = prev_id

                        if best_id is not None:
                            logging.debug(f"  后处理匹配: 当前花蕾 {i} 匹配到上一帧花蕾ID {best_id}, 代价 {best_cost:.2f}")
                            matched_ids[i] = best_id
                            assigned_ids.add(best_id)
                        else:
                            new_id = self.get_new_id()
                            logging.debug(f"  分配新ID: 当前花蕾 {i} 分配新ID {new_id}")
                            matched_ids[i] = new_id
                            assigned_ids.add(new_id)

                    else:
                        if matched_ids[i] not in self.id_history:
                            self.id_history[matched_ids[i]] = []
                        self.id_history[matched_ids[i]].append(curr_bud)

                    logging.debug(f"当前花蕾 {i} 的最终ID为: {matched_ids[i]}")
                    logging.debug(f"当前花蕾 {i} 的坐标为: {curr_bud}")

            logging.debug("\n处理未匹配的花蕾:")
            for i, bud_id in enumerate(matched_ids):
                if bud_id == 0 and i != top_flower_index:
                    new_id = self.get_new_id()
                    matched_ids[i] = new_id
                    assigned_ids.add(new_id)
                    self.all_used_ids.add(new_id)
                    self.id_history[new_id] = [current_buds[i]]
                    logging.debug(f"  花蕾 {i} 未匹配，分配新ID: {new_id}")

        self.previous_buds = current_buds
        self.previous_paths = current_paths
        self.previous_top_index = top_flower_index
        self.bud_ids = matched_ids

        logging.debug("\n最终匹配结果:")
        # 更新 all_historical_ids
        self.all_historical_ids.update(matched_ids)
        logging.debug(f"All historical IDs: {self.all_historical_ids}")
        logging.debug(f"Matched IDs: {matched_ids}")
        logging.debug(f"All used IDs: {self.all_used_ids}")
        return matched_ids

    def get_new_id(self):
        new_id = self.max_id + 1
        while new_id in self.all_used_ids:
            new_id += 1
        self.max_id = new_id
        return new_id

    def get_bud_trajectories(self):
        trajectories = {}
        for id in self.all_used_ids:
            if id in self.id_history:
                trajectories[id] = [(pos[0], pos[1], frame) for frame, pos in enumerate(self.id_history[id])]
            logging.debug(f"ID {id} 的历史轨迹: {trajectories.get(id, [])}")
        return trajectories
