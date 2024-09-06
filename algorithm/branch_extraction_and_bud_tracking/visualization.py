import matplotlib.pyplot as plt
from path_finding import is_connected
from skeleton.utils import set_axis_to_image


def draw_paths(ax, paths, skeleton, colors):
    for path, color in zip(paths, colors):
        xs, ys = zip(*path)
        ax.plot(xs, ys, color=color, linewidth=2, alpha=0.7)

        # 标记非骨架点
        for x, y in path:
            if skeleton[y, x] == 0:
                ax.plot(x, y, 'o', color=color, markersize=3, alpha=0.5)

    # 可选：如果您想保留跳跃点的虚线连接，可以添加以下代码
    for path, color in zip(paths, colors):
        for i in range(1, len(path)):
            if not is_connected(path[i-1], path[i], skeleton):
                x1, y1 = path[i-1]
                x2, y2 = path[i]
                ax.plot([x1, x2], [y1, y2], color=color, linestyle=':', linewidth=1, alpha=0.5)

def draw_summary_plot(ax, all_paths, colors, image_shape):
    for path, color in zip(all_paths, colors):
        xs, ys = zip(*path)
        ax.plot(xs, ys, color=color, linewidth=2, alpha=0.7)
    ax.set_title("Summary of All Paths")
    set_axis_to_image(ax, image_shape)

def draw_individual_plots(fig, all_paths, colors, image_shape):
    n_paths = len(all_paths)
    for i, (path, color) in enumerate(zip(all_paths, colors)):
        ax = fig.add_subplot(1, n_paths, i+1)
        xs, ys = zip(*path)
        ax.plot(xs, ys, color=color, linewidth=2)
        ax.set_title(f"Path {i+1}")
        set_axis_to_image(ax, image_shape)