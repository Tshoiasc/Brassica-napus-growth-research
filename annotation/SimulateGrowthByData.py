import json
import turtle
from scipy.interpolate import CubicSpline
import numpy as np


def parse_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['annotations'][0]['result']


def setup_turtle():
    screen = turtle.Screen()
    screen.setup(800, 800)
    screen.title("Oilseed Rape Plant Structure")
    screen.setworldcoordinates(0, 0, 100, 100)

    t = turtle.Turtle()
    t.speed(0)  # Fastest speed
    t.hideturtle()
    return t, screen


def draw_spline(t, points, color, width):
    t.color(color)
    t.width(width)
    t.penup()
    t.goto(points[0])
    t.pendown()

    if len(points) > 2:
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        # Create a parameterization of the curve
        t_param = np.linspace(0, 1, len(points))

        # Fit cubic splines
        cs_x = CubicSpline(t_param, x)
        cs_y = CubicSpline(t_param, y)

        # Draw the spline
        for t_val in np.linspace(0, 1, 100):
            t.goto(cs_x(t_val), cs_y(t_val))
    else:
        # If there are only two points, draw a straight line
        t.goto(points[-1])


def draw_node(t, pos, size, color):
    t.penup()
    t.goto(pos)
    t.dot(size, color)


def create_turtle_drawing(data):
    t, screen = setup_turtle()

    stem_points = []
    branches = {}
    nodes = set()

    # First pass: Identify all points and their relationships
    for item in data:
        x = item['value']['x']
        y = 100 - item['value']['y']  # Invert y-coordinate
        pos = (x, y)
        labels = item['value']['keypointlabels']
        item_id = item['id']

        if 'stem' in labels:
            stem_points.append(pos)
            nodes.add(pos)
            if 'branch_1' in labels:
                branches[item_id] = {'type': 'branch_1', 'points': [pos], 'children': []}
        elif any(label.startswith('branch_') for label in labels):
            branch_type = next(label for label in labels if label.startswith('branch_'))
            parent_id = item.get('parentID')
            if parent_id not in branches:
                branches[parent_id] = {'type': branch_type, 'points': [], 'children': []}
            branches[parent_id]['points'].append(pos)
            if len(labels) > 1:  # It's a node connecting different branch levels
                nodes.add(pos)
                branches[item_id] = {'type': branch_type, 'points': [pos], 'children': []}
                branches[parent_id]['children'].append(item_id)

    # Draw stem
    draw_spline(t, stem_points, "dark green", 2)

    # Function to recursively draw branches
    def draw_branches(branch_id):
        branch = branches[branch_id]
        branch_level = int(branch['type'].split('_')[1])
        color = ["forest green", "light green", "olive", "yellow green"][min(branch_level - 1, 3)]
        width = max(0.5, 1.5 - 0.3 * (branch_level - 1))

        draw_spline(t, branch['points'], color, width)

        for child_id in branch['children']:
            draw_branches(child_id)

    # Draw all branches
    for branch_id, branch in branches.items():
        if branch['type'] == 'branch_1':
            draw_branches(branch_id)

    # Draw nodes
    for node in nodes:
        draw_node(t, node, 5, "dark green")

    screen.update()
    return screen


if __name__ == "__main__":
    input_file = "input.json"  # Replace with your JSON file path
    data = parse_json(input_file)
    screen = create_turtle_drawing(data)
    screen.exitonclick()