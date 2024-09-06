from queue import PriorityQueue
import numpy as np

def is_connected(p1, p2, skeleton):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return True
    x_step = dx / steps
    y_step = dy / steps
    for i in range(1, steps):
        x = int(x1 + i * x_step)
        y = int(y1 + i * y_step)
        if skeleton[y, x] == 0:
            return False
    return True

def find_paths(skeleton, start, end, is_top_flower=False, force_path=False):
    def heuristic(a, b):
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def get_overall_direction(start, end):
        return np.array(end) - np.array(start)

    overall_direction = get_overall_direction(start, end)

    def get_neighbors(current, current_direction, steps_taken):
        x, y = current
        neighbors = []
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        if is_top_flower:
            directions = sorted(directions, key=lambda d: d[1])  # 优先向上
        else:
            directions = sorted(directions, key=lambda d: -d[1])  # 优先向下

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < skeleton.shape[1] and 0 <= ny < skeleton.shape[0]:
                new_direction = np.array([dx, dy])
                if skeleton[ny, nx] > 0:
                    cost = 1
                else:
                    if force_path:
                        cost = 10 + steps_taken * 0.1
                    else:
                        continue

                # 考虑方向惯性
                if np.dot(new_direction, current_direction) < 0:
                    cost += 5  # 惩罚方向反转

                # 考虑整体方向
                if np.dot(new_direction, overall_direction) < 0:
                    cost += 3  # 惩罚与整体方向相反的移动

                neighbors.append((cost, (nx, ny), tuple(new_direction)))

        return neighbors

    frontier = PriorityQueue()
    frontier.put((0, start, (0, 0), 0))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        priority, current, direction, steps = frontier.get()

        if current == end:
            break

        for cost, next, new_direction in get_neighbors(current, direction, steps):
            new_cost = cost_so_far[current] + cost
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(end, next)
                frontier.put((float(priority), next, new_direction, steps + 1))
                came_from[next] = current

    if end not in came_from:
        if force_path:
            return None
        else:
            return find_paths(skeleton, start, end, is_top_flower, force_path=True)

    path = []
    current = end
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return [path]