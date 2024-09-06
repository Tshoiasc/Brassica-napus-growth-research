import os
import sys
import csv
import logging

from tqdm import tqdm

from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from branch_analysis import draw_branch_level_plots, calculate_branch_metrics
from bud_tracking import BudTracker
from image_processing import process_green_with_detection
from path_finding import find_paths
from skeleton.utils import set_axis_to_image
from visualization import draw_paths, draw_summary_plot, draw_individual_plots


def save_and_close_figure(fig, filename):
    fig.savefig(filename)
    plt.close(fig)


def get_top_flower_index(green_centers):
    return min(range(len(green_centers)), key=lambda i: green_centers[i][1])

def calculate_angle(branch, top_flower_path):
    branch_vector = np.array(branch[-1]) - np.array(branch[0])
    top_flower_vector = np.array(top_flower_path[-1]) - np.array(top_flower_path[0])
    angle = np.arctan2(np.cross(branch_vector, top_flower_vector), np.dot(branch_vector, top_flower_vector))
    return np.degrees(angle)


def process_folder(folder_path, output_dir, log_file):
    logging.info(f"Processing folder: {folder_path}")

    sys.path.append('/Users/tshoiasc/Documents/文稿 - 陈跃千纪的MacBook Pro (325)/durham/毕业论文/代码/ultralytics-main')
    custom_model = YOLO("../bets2.pt")

    image_list = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])
    logging.info(f"Found {len(image_list)} images")

    folder_output_dir = os.path.join(output_dir, os.path.basename(folder_path))
    os.makedirs(folder_output_dir, exist_ok=True)

    lowest_point = None
    lowest_box = None
    bud_tracker = None
    start_tracking = False
    consecutive_bud_frames = 0
    required_consecutive_frames = 3
    first_valid_frame = None
    detection_buffer = []

    csv_data = []
    total_images = len(image_list)
    for frame, image_path in tqdm(enumerate(image_list), total=total_images, desc="处理图片"):
        result = custom_model.predict(image_path)

        # 检查是否检测到目标
        if len(result[0].boxes.xyxy.cpu().numpy()) > 0:
            detection_buffer.append((frame, result[0].boxes.xyxy.cpu().numpy()))
        else:
            detection_buffer = []

        # 如果连续三帧都检测到目标
        if len(detection_buffer) == required_consecutive_frames and not start_tracking:
            start_tracking = True
            first_valid_frame, first_frame_boxes = detection_buffer[0]
            lowest_box = max(first_frame_boxes, key=lambda box: box[3])
            lowest_point = int(lowest_box[3])
            bud_tracker = BudTracker(total_frames=len(image_list) - first_valid_frame)
            logging.info(f"Started tracking from frame {first_valid_frame}")

        if not start_tracking:
            continue

        adjusted_frame = frame - first_valid_frame

        color_mask, red_centers, green_centers, skeleton = process_green_with_detection(image_path, result, lowest_box)

        if len(green_centers) == 0:
            logging.info(f"No green centers detected in frame {frame}")
            continue

        image_shape = skeleton.shape

        all_paths = []
        for i, green_center in enumerate(green_centers):
            start = (green_center[0], green_center[1])
            end = (red_centers[0][0], red_centers[0][1])
            path = find_paths(skeleton, start, end, is_top_flower=(i == 0))
            if path:
                all_paths.extend(path)

        top_flower_index = get_top_flower_index(green_centers)

        if top_flower_index != 0:
            green_centers[0], green_centers[top_flower_index] = green_centers[top_flower_index], green_centers[0]
            all_paths[0], all_paths[top_flower_index] = all_paths[top_flower_index], all_paths[0]
            top_flower_index = 0

        bud_ids = bud_tracker.match_buds(green_centers, all_paths, top_flower_index, skeleton)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(all_paths)))

        # 创建并保存图像
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(color_mask)
        draw_paths(ax1, all_paths, skeleton, colors)
        for i, (center, bud_id) in enumerate(zip(green_centers, bud_ids)):
            ax1.plot(center[0], center[1], 'go', markersize=10)
            ax1.text(center[0], center[1], str(bud_id), color='white', fontweight='bold', ha='center', va='center')
        for center in red_centers:
            ax1.plot(center[0], center[1], 'ro', markersize=10)
        ax1.axhline(y=lowest_point, color='r', linestyle='--')
        ax1.set_title(f"Tracked Buds: {os.path.basename(image_path)}")
        set_axis_to_image(ax1, image_shape)
        draw_summary_plot(ax2, all_paths, colors, image_shape)
        save_and_close_figure(fig1,
                              os.path.join(folder_output_dir, f"frame_{adjusted_frame:03d}_skeleton_and_summary.png"))

        fig2, axes = plt.subplots(1, len(all_paths), figsize=(5 * len(all_paths), 5), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        draw_individual_plots(fig2, all_paths, colors, image_shape)
        save_and_close_figure(fig2, os.path.join(folder_output_dir, f"frame_{adjusted_frame:03d}_individual_paths.png"))

        fig3, axes = plt.subplots(1, len(all_paths), figsize=(5 * len(all_paths), 5), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        draw_branch_level_plots(fig3, all_paths, colors, image_shape, green_centers, bud_ids)
        save_and_close_figure(fig3, os.path.join(folder_output_dir, f"frame_{adjusted_frame:03d}_branch_levels.png"))

        branch_metrics = calculate_branch_metrics(all_paths, green_centers)
        top_flower_path = branch_metrics[0]['path']

        for i, metrics in enumerate(branch_metrics):
            bud_id = bud_ids[i] if i < len(bud_ids) else 'N/A'
            angle = calculate_angle(metrics['path'], top_flower_path) if metrics['level'] == 2 else ''

            path_str = ';'.join([f'{x},{y}' for x, y in metrics['path']])

            bud_position = green_centers[i] if i < len(green_centers) else ''

            csv_data.append([
                os.path.basename(image_path),
                adjusted_frame,
                bud_id,
                metrics['level'],
                bud_position,
                path_str,
                angle,
                metrics['length'],
                metrics['vertical_length'],
                metrics['horizontal_length']
            ])

    if bud_tracker:
        max_bud_count = max(len(frame_buds) for frame_buds in bud_tracker.id_history.values())

        trajectories, id_mapping = bud_tracker.post_process_trajectories(max_bud_count)
        print(f"ID Mapping: {id_mapping}")

        fig4 = plt.figure(figsize=(12, 8))
        for bud_id, trajectory in trajectories.items():
            x = [point[0] for point in trajectory]
            y = [point[1] for point in trajectory]
            plt.plot(x, y, '-o', label=f'Bud {bud_id}')

        plt.title('Bud Trajectories (Post-processed)')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.gca().invert_yaxis()
        save_and_close_figure(fig4, os.path.join(folder_output_dir, "all_bud_trajectories_post_processed.png"))

        logging.info(f"Generated post-processed trajectory plot for all buds")

        updated_csv_data = []
        for row in csv_data:
            image_name, frame, old_bud_id, level, bud_position, *rest = row
            if old_bud_id != 'N/A':
                old_bud_id = int(old_bud_id)
                if old_bud_id in bud_tracker.all_used_ids and old_bud_id in id_mapping:
                    new_bud_id = str(id_mapping[old_bud_id])
                else:
                    new_bud_id = ''
                updated_csv_data.append([image_name, frame, str(old_bud_id), new_bud_id, level, bud_position] + rest)
            else:
                updated_csv_data.append([image_name, frame, old_bud_id, '', level, bud_position] + rest)

        csv_data = updated_csv_data
    else:
        logging.warning("No stable bud detection achieved. Unable to generate trajectories.")

    csv_file = os.path.join(folder_output_dir, "branch_and_trajectory_data.csv")
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Image Name', 'Frame', 'Old Bud ID', 'New Bud ID', 'Branch Level', 'Bud Position', 'Branch Path',
            'Angle', 'Length', 'Vertical Length', 'Horizontal Length'
        ])
        writer.writerows(csv_data)


def main():
    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "processing.log")
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    with open("image_paths.txt", "r") as f:
        folders = f.read().splitlines()

    for folder in tqdm(folders, desc="处理文件夹", unit="folder"):
        process_folder(folder, output_dir, log_file)

        # 从image_paths.txt中删除已处理的文件夹
        with open("image_paths.txt", "r") as f:
            lines = f.readlines()
        with open("image_paths.txt", "w") as f:
            for line in lines:
                if line.strip() != folder:
                    f.write(line)


if __name__ == "__main__":
    main()