import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

def generate_distinct_colors(n):
    base_colors = list(mcolors.TABLEAU_COLORS.values())
    if n <= len(base_colors):
        return base_colors[:n]
    return plt.cm.hsv(np.linspace(0, 1, n))

max_possible_buds = 20
color_map = {i: color for i, color in enumerate(generate_distinct_colors(max_possible_buds))}

def set_axis_to_image(ax, image_shape):
    ax.set_xlim(0, image_shape[1])
    ax.set_ylim(image_shape[0], 0)  # Reverse y-axis
    ax.set_aspect('equal', adjustable='box')