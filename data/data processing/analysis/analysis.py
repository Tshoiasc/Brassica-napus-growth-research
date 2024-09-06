import concurrent
import concurrent.futures
import concurrent.futures
import json
import logging
import multiprocessing
import os
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from scipy.stats import stats, lognorm, weibull_min
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm
# 设置全局样式
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 8,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})

# 新的颜色方案
color_palette = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974']
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def monotonic_spline(x, y):
    """Create a monotonic spline interpolation"""
    spl = make_interp_spline(x, y, k=3)
    x_new = np.linspace(x.min(), x.max(), 1000)
    y_new = spl(x_new)
    # Ensure monotonicity
    y_new = np.maximum.accumulate(y_new)
    return x_new, y_new


def logistic_function(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))


def fit_logistic(x, y):
    # Initial parameter guess
    p0 = [max(y), 1, np.median(x)]

    # Fit the data
    popt, _ = curve_fit(logistic_function, x, y, p0, maxfev=10000)

    return popt


def plot_regression(ax, x, y, color, label):
    # Convert dates to numbers
    x_num = mdates.date2num(x)

    # Ensure x and y are numpy arrays
    x_num = np.array(x_num)
    y = np.array(y)

    # Sort data
    sort_idx = np.argsort(x_num)
    x_num = x_num[sort_idx]
    y = y[sort_idx]

    # Fit logistic function
    popt = fit_logistic(x_num, y)

    # Generate points for the fitted line
    x_fit = np.linspace(x_num.min(), x_num.max(), 1000)
    y_fit = logistic_function(x_fit, *popt)

    # Plot the regression line
    ax.plot(mdates.num2date(x_fit), y_fit, color=color, label=label, linestyle='-', linewidth=2)

    return x_fit, y_fit



def load_genotype_info(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return {item['plant_id']: {'geno_type': item['geno_type'], 'crop_type': item['crop_type']} for item in data}


genotype_info = load_genotype_info('data_available.json')


def extract_info_from_filename(filename):
    parts = filename.split('-')
    plant_id = f"{parts[0]}-{parts[1][:6]}"
    position = parts[1][3]
    temperature = '5C' if parts[1][4] == '1' else '10C'
    repetition = parts[1][5]
    info = genotype_info.get(plant_id, {'geno_type': 'Unknown', 'crop_type': 'Unknown'})
    return plant_id, info['geno_type'], info['crop_type'], position, temperature, repetition


def process_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    plant_id, geno_type, crop_type, position, temperature, repetition = extract_info_from_filename(
        os.path.basename(file_path))

    results = []
    for angle in ['sv-000', 'sv-045', 'sv-090']:
        if angle not in data:
            continue
        for image_data in data[angle]:
            try:
                date_str = image_data['image_name'].split('-')
                date_str = date_str[2] + "_" + date_str[3] + "_" + date_str[4].split("_")[0]
                date = datetime.strptime(date_str, '%Y_%m_%d').date()

                predictions = image_data.get('predictions', [])

                for prediction in predictions:
                    box = prediction['boxes'][0]
                    x_center = (box[0] + box[2]) / 2
                    y_center = (box[1] + box[3]) / 2

                    results.append({
                        'date': date,
                        'plant_id': plant_id,
                        'geno_type': geno_type,
                        'crop_type': crop_type,
                        'position': position,
                        'temperature': temperature,
                        'repetition': repetition,
                        'angle': angle,
                        'x_position': x_center,
                        'y_position': y_center,
                        'confidence': prediction['conf'][0]
                    })
            except (IndexError, ValueError) as e:
                logging.error(f"Error processing image data in file {file_path}: {str(e)}")

    return results


def process_all_files(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]

    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_json_file, files), total=len(files), desc="Processing files"))

    all_results = [item for sublist in results for item in sublist]

    logging.info("Creating DataFrame from processed data")
    df = pd.DataFrame(all_results)

    # 统计信息
    num_files = len(files)
    num_genotypes = df['geno_type'].nunique()
    num_crop_types = df['crop_type'].nunique()

    crop_type_counts = df.groupby('crop_type')['plant_id'].nunique().to_dict()
    genotype_counts = df.groupby('geno_type')['plant_id'].nunique().to_dict()

    statistics = {
        'num_files': num_files,
        'num_genotypes': num_genotypes,
        'num_crop_types': num_crop_types,
        'crop_type_counts': crop_type_counts,
        'genotype_counts': genotype_counts
    }

    return df, statistics


def create_output_directory():
    output_dir = 'output_charts2'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def create_genotype_temperature_heatmap(df, output_dir):
    # 确保每个基因型只有一行数据
    pivot_df = df.groupby(['geno_type', 'temperature']).size().unstack(fill_value=0)

    plt.figure(figsize=(15, 20))  # 增加图表高度以适应所有基因型
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='d', cbar_kws={'label': 'Bud Count'})
    plt.title('Total Bud Count by Genotype and Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Genotype')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'genotype_temperature_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    logging.info("Genotype-Temperature heatmap created")


def create_bud_count_distribution(df, output_dir):
    logging.info("Creating bud count distribution chart")

    # 首先，对每个日期、植株、基因型、温度和视角组合计算花蕾数
    daily_bud_count = df.groupby(['date', 'plant_id', 'geno_type', 'temperature', 'angle']).size().reset_index(
        name='bud_count')

    # 然后，计算每个日期、植株、基因型和温度组合的平均花蕾数（跨三个视角）
    daily_avg_count = daily_bud_count.groupby(['date', 'plant_id', 'geno_type', 'temperature'])[
        'bud_count'].mean().reset_index()

    # 找出每个植株的最大平均花蕾数
    max_bud_count = daily_avg_count.groupby(['plant_id', 'geno_type', 'temperature'])['bud_count'].max().reset_index(
        name='max_bud_count')

    # 计算一些统计信息
    stats = max_bud_count.groupby(['geno_type', 'temperature'])['max_bud_count'].agg(
        ['mean', 'median', 'min', 'max']).reset_index()
    print(stats)

    plt.figure(figsize=(15, 20))
    sns.boxplot(y='geno_type', x='max_bud_count', hue='temperature', data=max_bud_count, orient='h')
    plt.title('Maximum Average Bud Count Distribution by Genotype and Temperature')
    plt.ylabel('Genotype')
    plt.xlabel('Maximum Average Bud Count (Average of 3 Views)')
    plt.legend(title='Temperature', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bud_count_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    logging.info("Bud count distribution chart created")


def create_angle_comparison(df, output_dir):
    logging.info("Creating angle comparison chart")
    plt.figure(figsize=(15, 20))
    angle_counts = df.groupby(['geno_type', 'angle']).size().reset_index(name='bud_count')
    sns.barplot(y='geno_type', x='bud_count', hue='angle', data=angle_counts)
    plt.title('Bud Count Comparison by Angle')
    plt.ylabel('Genotype')
    plt.xlabel('Bud Count')
    plt.legend(title='Angle', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'angle_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def calculate_max_bud_count(df):
    # 计算每个植株在每个日期和每个视角的花蕾数
    daily_bud_count = df.groupby(
        ['date', 'plant_id', 'crop_type', 'geno_type', 'temperature', 'angle']).size().reset_index(name='bud_count')

    # 计算每个日期每个植株across三个视角的平均花蕾数
    daily_avg_count = daily_bud_count.groupby(['date', 'plant_id', 'crop_type', 'geno_type', 'temperature'])[
        'bud_count'].mean().reset_index()

    # 找出每个植株的最大平均花蕾数
    max_bud_count = daily_avg_count.groupby(['plant_id', 'crop_type', 'geno_type', 'temperature'])[
        'bud_count'].max().reset_index()

    return max_bud_count


def perform_clustering_analysis(df, output_dir):
    logging.info("Performing clustering analysis")

    # 选择用于聚类的特征
    features = ['x_position', 'y_position']
    X = df[features]

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 获取唯一的 crop_type 数量
    n_clusters = df['crop_type'].nunique()

    # 执行 K-means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # 可视化聚类结果
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['x_position'], df['y_position'], c=df['cluster'], cmap='viridis')
    plt.colorbar(scatter)
    plt.title('Clustering Results')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.savefig(os.path.join(output_dir, 'clustering_results.png'))
    plt.close()

    # 比较聚类结果与实际的 crop_type
    crop_type_mapping = {crop: i for i, crop in enumerate(df['crop_type'].unique())}
    df['crop_type_numeric'] = df['crop_type'].map(crop_type_mapping)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(df['crop_type_numeric'], df['cluster'])

    # 绘制混淆矩阵热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix: Clusters vs Crop Types')
    plt.xlabel('Predicted Cluster')
    plt.ylabel('Actual Crop Type')
    plt.savefig(os.path.join(output_dir, 'cluster_crop_type_confusion_matrix.png'))
    plt.close()

    # 计算调整兰德指数
    ari = adjusted_rand_score(df['crop_type_numeric'], df['cluster'])
    logging.info(f"Adjusted Rand Index: {ari}")

    # 为每个 crop_type 绘制花蕾分布图
    crop_types = df['crop_type'].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(crop_types)))

    plt.figure(figsize=(12, 8))
    for crop_type, color in zip(crop_types, colors):
        crop_data = df[df['crop_type'] == crop_type]
        plt.scatter(crop_data['x_position'], crop_data['y_position'], c=[color], label=crop_type, alpha=0.6)

    plt.title('Bud Distribution by Crop Type')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'bud_distribution_by_crop_type.png'))
    plt.close()

    logging.info("Clustering analysis completed")

def create_improved_crop_type_comparison(df, output_dir):
    max_bud_count = calculate_max_bud_count(df)

    # 1. 箱线图
    plt.figure(figsize=(15, 10))
    sns.boxplot(x='crop_type', y='bud_count', hue='temperature', data=max_bud_count)
    plt.title('Distribution of Maximum Bud Count by Crop Type and Temperature')
    plt.xlabel('Crop Type')
    plt.ylabel('Maximum Bud Count')
    plt.xticks()
    plt.legend(title='Temperature', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'crop_type_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 统计显著性检验
    crop_types = max_bud_count['crop_type'].unique()
    for crop in crop_types:
        group_5c = max_bud_count[(max_bud_count['crop_type'] == crop) & (max_bud_count['temperature'] == '5C')][
            'bud_count']
        group_10c = max_bud_count[(max_bud_count['crop_type'] == crop) & (max_bud_count['temperature'] == '10C')][
            'bud_count']
        t_stat, p_value = stats.ttest_ind(group_5c, group_10c)
        print(f"{crop}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

    # 3. 相对变化
    avg_by_temp = max_bud_count.groupby(['crop_type', 'temperature'])['bud_count'].mean().unstack()
    relative_change = (avg_by_temp['10C'] - avg_by_temp['5C']) / avg_by_temp['5C'] * 100

    plt.figure(figsize=(12, 6))
    relative_change.plot(kind='bar')
    plt.title('Relative Change in Maximum Bud Count (5C to 10C)')
    plt.xlabel('Crop Type')
    plt.ylabel('Relative Change (%)')
    plt.xticks()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'crop_type_relative_change.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 热图
    plt.figure(figsize=(12, 8))
    sns.heatmap(avg_by_temp, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Heatmap of Average Maximum Bud Count by Crop Type and Temperature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'crop_type_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. 带误差棒的柱状图
    plt.figure(figsize=(15, 10))
    sns.barplot(x='crop_type', y='bud_count', hue='temperature', data=max_bud_count, ci='sd', capsize=0.1)
    plt.title('Average Maximum Bud Count by Crop Type and Temperature (with Error Bars)')
    plt.xlabel('Crop Type')
    plt.ylabel('Average Maximum Bud Count')
    plt.xticks()
    plt.legend(title='Temperature', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'crop_type_barplot_with_error.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_improved_genotype_comparison(df, output_dir):
    max_bud_count = calculate_max_bud_count(df)

    # 计算每种基因型的平均最大花蕾数
    genotype_avg = max_bud_count.groupby(['geno_type', 'temperature'])['bud_count'].mean().reset_index()

    # 计算每个基因型的总平均花蕾数（跨温度），用于排序
    genotype_total_avg = genotype_avg.groupby('geno_type')['bud_count'].mean().sort_values(ascending=False)

    # 根据总平均花蕾数对基因型进行排序
    genotype_order = genotype_total_avg.index.tolist()

    # 设置基因型的顺序
    genotype_avg['geno_type'] = pd.Categorical(genotype_avg['geno_type'], categories=genotype_order, ordered=True)

    plt.figure(figsize=(12, 20))  # 调整图表尺寸以适应垂直方向和标签
    sns.barplot(y='geno_type', x='bud_count', hue='temperature', data=genotype_avg, orient='h', order=genotype_order,palette=color_palette[:2])
    plt.title('Average Maximum Bud Count by Genotype and Temperature')
    plt.ylabel('Genotype')
    plt.xlabel('Average Maximum Bud Count')
    plt.legend(title='Temperature', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 调整布局以确保所有标签都可见
    plt.tight_layout()

    # 保存图片
    plt.savefig(os.path.join(output_dir, 'improved_genotype_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    logging.info("Improved genotype comparison chart created")


def create_bud_position_analysis(df, output_dir):
    logging.info("Creating bud position analysis")
    logging.info(f"Available columns: {df.columns.tolist()}")

    # Filter for sv-000 view only
    df_sv000 = df[df['angle'] == 'sv-000']

    temperatures = df_sv000['temperature'].unique()
    genotypes = df_sv000['geno_type'].unique()
    crop_types = df_sv000['crop_type'].unique()
    max_subplots_per_image = 4

    for start_idx in range(0, len(temperatures), max_subplots_per_image):
        end_idx = min(start_idx + max_subplots_per_image, len(temperatures))
        current_temps = temperatures[start_idx:end_idx]

        fig, axes = plt.subplots(len(current_temps), 2, figsize=(20, 10 * len(current_temps)))
        if len(current_temps) == 1:
            axes = np.array([axes])

        for i, temp in enumerate(current_temps):
            temp_data = df_sv000[df_sv000['temperature'] == temp]

            # Scatter plot
            scatter = sns.scatterplot(data=temp_data, x='x_position', y='y_position', hue='geno_type',
                                      style='geno_type', ax=axes[i, 0], s=10, alpha=0.6)
            axes[i, 0].set_title(f'Bud Distribution at {temp} (sv-000 view)')
            axes[i, 0].set_xlabel('X Position')
            axes[i, 0].set_ylabel('Y Position')

            # Set axis limits based on data
            x_min, x_max = temp_data['x_position'].min(), temp_data['x_position'].max()
            y_min, y_max = temp_data['y_position'].min(), temp_data['y_position'].max()
            axes[i, 0].set_xlim(x_min, x_max)
            axes[i, 0].set_ylim(y_min, y_max)

            axes[i, 0].get_legend().remove()

            # Heatmap
            heatmap_data, xedges, yedges = np.histogram2d(temp_data['x_position'], temp_data['y_position'], bins=50)
            axes[i, 1].imshow(heatmap_data.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                              origin='lower', cmap='YlOrRd', aspect='auto')
            axes[i, 1].set_title(f'Bud Density Heatmap at {temp} (sv-000 view)')
            axes[i, 1].set_xlabel('X Position')
            axes[i, 1].set_ylabel('Y Position')

        # Create a unified legend
        handles, labels = scatter.get_legend_handles_labels()
        fig.legend(handles, labels, title='Genotype', loc='lower center', ncol=min(len(genotypes), 7),
                   bbox_to_anchor=(0.5, -0.06), bbox_transform=fig.transFigure)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)  # Adjust this value to move the legend further down
        plt.savefig(os.path.join(output_dir, f'bud_position_genotype_comparison_{start_idx + 1}-{end_idx}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    logging.info("Bud position analysis completed")

def create_bud_height_distribution(df, output_dir):
    plt.figure(figsize=(15, 10))
    sns.boxplot(x='crop_type', y='y_position', hue='temperature', data=df)
    plt.title('Bud Height Distribution by Crop Type and Temperature')
    plt.xlabel('Crop Type')
    plt.ylabel('Bud Height (Y Position)')
    plt.xticks()
    plt.legend(title='Temperature', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bud_height_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def analyze_bud_positions(df, output_dir):
    logging.info("Analyzing bud positions")
    # Calculate average bud height and count for each plant and date
    plant_data = df.groupby(['plant_id', 'date']).agg({
        'y_position': 'mean',
        'plant_id': 'count'  # This gives us the bud count
    }).rename(columns={'plant_id': 'bud_count'}).reset_index()

    # Add other information
    plant_data = pd.merge(plant_data, df[['plant_id', 'geno_type', 'crop_type', 'temperature']].drop_duplicates(),
                          on='plant_id')

    for crop in plant_data['crop_type'].unique():
        plt.figure(figsize=(15, 10))
        for temp in plant_data['temperature'].unique():
            data = plant_data[(plant_data['crop_type'] == crop) & (plant_data['temperature'] == temp)]

            # Scatter plot
            plt.scatter(data['bud_count'], data['y_position'], label=f'{temp}', alpha=0.5)

            # Fit a polynomial curve
            z = np.polyfit(data['bud_count'], data['y_position'], 2)
            p = np.poly1d(z)
            x_range = np.linspace(data['bud_count'].min(), data['bud_count'].max(), 100)
            plt.plot(x_range, p(x_range), '--', label=f'{temp} fit')

        plt.title(f'Average Bud Height vs Bud Count for {crop}')
        plt.xlabel('Number of Buds')
        plt.ylabel('Average Bud Height')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'bud_height_vs_count_{crop}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return plant_data

from sklearn.metrics import r2_score, adjusted_rand_score, confusion_matrix


def create_bud_count_over_time(df, output_dir):
    logging.info("Creating bud count over time chart")
    df['date'] = pd.to_datetime(df['date'])

    # Count buds for each plant, date, crop_type, temperature, and angle
    daily_bud_count = df.groupby(['date', 'plant_id', 'crop_type', 'temperature', 'angle']).size().reset_index(name='daily_count')

    # Calculate average bud count across angles
    daily_avg_count = daily_bud_count.groupby(['date', 'plant_id', 'crop_type', 'temperature'])['daily_count'].mean().reset_index(name='avg_count')

    # Calculate cumulative growth of average bud count
    daily_avg_count['cumulative_growth'] = daily_avg_count.groupby(['plant_id', 'temperature'])['avg_count'].cumsum()

    # Average across plants
    df_grouped = daily_avg_count.groupby(['date', 'crop_type', 'temperature'])['cumulative_growth'].mean().reset_index()

    # Find the overall date range and maximum cumulative growth
    min_date = df_grouped['date'].min()
    max_date = df_grouped['date'].max()
    max_cumulative_growth = df_grouped['cumulative_growth'].max()

    for temperature in df_grouped['temperature'].unique():
        plt.figure(figsize=(15, 10))

        legend_elements = []

        for crop in df_grouped['crop_type'].unique():
            data = df_grouped[(df_grouped['crop_type'] == crop) & (df_grouped['temperature'] == temperature)]
            data = data.sort_values('date')

            x = mdates.date2num(data['date'])
            y = data['cumulative_growth'].values

            # Improved initial parameter estimation
            L_init = np.max(y) * 1.2
            k_init = 1 / (np.max(x) - np.min(x))
            x0_init = np.median(x)

            try:
                popt, _ = curve_fit(logistic_function, x, y, p0=[L_init, k_init, x0_init], maxfev=10000)
            except RuntimeError:
                logging.warning(f"Curve fitting failed for {crop} at {temperature}. Using initial parameters.")
                popt = [L_init, k_init, x0_init]

            x_fit = np.linspace(x.min(), x.max(), 1000)
            y_fit = logistic_function(x_fit, *popt)

            r2 = r2_score(y, logistic_function(x, *popt))

            color = next(plt.gca()._get_lines.prop_cycler)['color']
            line, = plt.plot(mdates.num2date(x_fit), y_fit, label=f'{crop} (R² = {r2:.2f})', color=color)
            plt.scatter(data['date'], y, alpha=0.5, color=color, s=30)

            legend_elements.append(line)

            # Find and annotate the inflection point
            inflection_index = np.argmax(np.gradient(y_fit))
            inflection_date = mdates.num2date(x_fit[inflection_index])
            inflection_value = y_fit[inflection_index]

            # Plot inflection point
            plt.scatter(inflection_date, inflection_value, color=color, s=100, zorder=5)

            # Add inflection point info to legend
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                              label=f'{crop} Steepest growth: {inflection_date.strftime("%Y-%m-%d")} ({inflection_value:.2f})',
                                              markerfacecolor=color, markersize=10))

        plt.title(f'Cumulative Bud Growth Over Time ({temperature})')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Bud Count')
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlim(min_date, max_date)  # Set x-axis limits to be the same for both temperatures
        plt.ylim(0, max_cumulative_growth * 1.1)  # Set y-axis limit to the overall maximum
        plt.xticks()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'bud_count_over_time_{temperature}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    logging.info("Bud count over time charts created successfully")

def create_temperature_comparison(df, output_dir):
    logging.info("Creating temperature comparison chart")
    df['date'] = pd.to_datetime(df['date'])

    # 定义颜色
    colors = ['#5471AB', '#6AA66E']

    # Count buds for each plant, date, crop_type, and temperature (remove angle)
    daily_bud_count = df.groupby(['date', 'plant_id', 'crop_type', 'temperature']).size().reset_index(name='total_count')

    # Calculate average bud count across three angles (by dividing the total by 3)
    daily_bud_count['avg_count'] = daily_bud_count['total_count'] / 3

    # Average across plants
    df_grouped = daily_bud_count.groupby(['date', 'crop_type', 'temperature'])['avg_count'].mean().reset_index()

    crop_types = df_grouped['crop_type'].unique()

    # 计算全局最大纵坐标值
    y_max = df_grouped['avg_count'].max() * 1.1  # 添加10%的额外空间

    for crop in crop_types:
        plt.figure(figsize=(15, 10))

        legend_elements = []

        for i, temperature in enumerate(df_grouped['temperature'].unique()):
            data = df_grouped[(df_grouped['crop_type'] == crop) & (df_grouped['temperature'] == temperature)]
            data = data.sort_values('date')

            logging.info(f"Plotting data for {crop} at {temperature}: {len(data)} points")
            if data.empty:
                logging.warning(f"No data available for {crop} at {temperature}. Skipping.")
                continue

            x = mdates.date2num(data['date'])
            y = data['avg_count'].values  # 直接使用平均花蕾数

            color = colors[i % len(colors)]  # 从预定义的颜色列表中选择颜色

            # 使用相同的颜色绘制散点图
            plt.scatter(data['date'], y, alpha=0.5, color=color, s=30)

            try:
                # 使用fit_curve函数进行动态曲线拟合
                y_fit, method, r2 = fit_curve(x, y, crop, temperature)

                # 生成拟合数据点以便于绘制平滑曲线
                if len(y_fit) != len(x):  # 如果y_fit长度与x不匹配，需要重新插值
                    x_fit = np.linspace(x.min(), x.max(), 1000)  # 限制x_fit的范围在数据范围内
                    y_fit = np.interp(x_fit, x, y_fit)
                else:
                    x_fit = x

                line, = plt.plot(mdates.num2date(x_fit), y_fit, label=f'{temperature} ({method}, R² = {r2:.2f})', color=color)

                legend_elements.append(line)

                # 计算拐点（最陡峭增长点）
                dy = np.gradient(y_fit, x_fit)
                inflection_index = np.argmax(dy)
                inflection_date = mdates.num2date(x_fit[inflection_index])
                inflection_value = y_fit[inflection_index]

                # 绘制拐点
                plt.scatter(inflection_date, inflection_value, color=color, s=100, zorder=5)

                # 添加拐点信息到图例中
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                  label=f'{temperature} Steepest growth: {inflection_date.strftime("%Y-%m-%d")} ({inflection_value:.2f})',
                                                  markerfacecolor=color, markersize=10))

            except RuntimeError:
                logging.warning(f"Curve fitting failed for {crop} at {temperature}. Skipping this dataset.")

        plt.title(f'Temperature Comparison of Average Bud Growth Over Time ({crop})')
        plt.xlabel('Date')
        plt.ylabel('Average Bud Count')
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98), prop={'size': 14})

        # 统一设置y轴的最大值
        plt.ylim(0, y_max)  # 设置y轴的最大值一致

        plt.xticks()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'temperature_comparison_{crop}.png'), dpi=300)
        plt.close()

    logging.info("Temperature comparison charts created successfully")








def create_bud_growth_rate(df, output_dir):
    logging.info("Creating bud growth rate chart")
    df['date'] = pd.to_datetime(df['date'])

    # Calculate the cumulative growth
    df_sorted = df.sort_values(['plant_id', 'date'])
    df_sorted['cumulative_growth'] = df_sorted.groupby('plant_id')['bud_count'].cumsum()

    # Group by date, crop_type, and temperature
    df_grouped = df_sorted.groupby(['date', 'crop_type', 'temperature'])['cumulative_growth'].mean().reset_index()

    crop_types = df_grouped['crop_type'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(crop_types)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20), sharex=True)

    for ax, temperature in zip([ax1, ax2], ['5C', '10C']):
        for crop, color in zip(crop_types, colors):
            data = df_grouped[(df_grouped['crop_type'] == crop) & (df_grouped['temperature'] == temperature)]

            # Sort data by date
            data = data.sort_values('date')

            x_fit, y_fit = plot_regression(ax, data['date'].values, data['cumulative_growth'].values, color, crop)

            ax.scatter(data['date'], data['cumulative_growth'], color=color, alpha=0.5)

            # Find and annotate the inflection point (steepest growth)
            inflection_index = np.argmax(np.gradient(y_fit))
            inflection_date = mdates.num2date(x_fit[inflection_index])
            inflection_value = y_fit[inflection_index]
            ax.annotate(f'Steepest growth: {inflection_date.strftime("%Y-%m-%d")}',
                        (inflection_date, inflection_value),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        color=color)
            ax.axvline(x=inflection_date, color=color, linestyle=':', alpha=0.5)

        ax.set_title(f'Cumulative Bud Growth Over Time ({temperature})')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Weighted Bud Count')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xticks()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bud_growth_rate_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


# def perform_clustering_analysis(df, output_dir):
#     logging.info("Performing clustering analysis")
#
#     # 选择用于聚类的特征
#     features = ['x_position', 'y_position', 'confidence']
#     X = df[features]
#
#     # 标准化数据
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # 使用肘部法则确定最佳聚类数
#     inertias = []
#     K = range(1, 11)
#     for k in K:
#         kmeans = KMeans(n_clusters=k, random_state=42)
#         kmeans.fit(X_scaled)
#         inertias.append(kmeans.inertia_)
#
#     # 绘制肘部曲线
#     plt.figure(figsize=(10, 6))
#     plt.plot(K, inertias, 'bx-')
#     plt.xlabel('k')
#     plt.ylabel('Inertia')
#     plt.title('Elbow Method For Optimal k')
#     plt.savefig(os.path.join(output_dir, 'elbow_curve.png'))
#     plt.close()
#
#     # 选择最佳聚类数（这里假设为3，您可以根据肘部曲线调整）
#     best_k = 3
#     kmeans = KMeans(n_clusters=best_k, random_state=42)
#     df['cluster'] = kmeans.fit_predict(X_scaled)
#
#     # 可视化聚类结果
#     plt.figure(figsize=(12, 8))
#     scatter = plt.scatter(df['x_position'], df['y_position'], c=df['cluster'], cmap='viridis')
#     plt.colorbar(scatter)
#     plt.title('Clustering Results')
#     plt.xlabel('X Position')
#     plt.ylabel('Y Position')
#     plt.savefig(os.path.join(output_dir, 'clustering_results.png'))
#     plt.close()
#
#     logging.info(f"Clustering analysis completed with {best_k} clusters")


# def perform_pca_analysis(df, output_dir):
#     logging.info("Performing PCA analysis")
#
#     # 选择用于PCA的特征
#     features = ['x_position', 'y_position', 'confidence']
#     X = df[features]
#
#     # 标准化数据
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # 执行PCA
#     pca = PCA()
#     pca_result = pca.fit_transform(X_scaled)
#
#     # 计算解释方差比和累积解释方差比
#     explained_variance_ratio = pca.explained_variance_ratio_
#     cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
#
#     # 绘制改进的碎石图
#     plt.figure(figsize=(12, 6))
#
#     # 柱状图
#     bars = plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7)
#
#     # 折线图
#     plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'ro-')
#
#     # 在柱形图上方显示数值
#     for i, bar in enumerate(bars):
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width() / 2., height,
#                  f'{height:.2f}',
#                  ha='center', va='bottom')
#
#     plt.xlabel('Principal Component')
#     plt.ylabel('Explained Variance Ratio')
#     plt.title('Scree Plot: Explained Variance Ratio by Principal Components')
#     plt.legend(['Cumulative Explained Variance Ratio', 'Individual Explained Variance Ratio'])
#     plt.xticks(range(1, len(explained_variance_ratio) + 1))
#     plt.ylim(0, 1.1)  # 设置y轴范围，留出一些空间显示数值
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'pca_scree_plot.png'), dpi=300, bbox_inches='tight')
#     plt.close()
#
#     # 将温度转换为数值
#     temp_map = {'5C': 0, '10C': 1}
#     temp_numeric = df['temperature'].map(temp_map)
#
#     # 绘制前两个主成分的散点图
#     plt.figure(figsize=(12, 8))
#     scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=temp_numeric, cmap='coolwarm')
#     plt.colorbar(scatter, label='Temperature', ticks=[0, 1],
#                  format=plt.FuncFormatter(lambda x, p: '5C' if x == 0 else '10C'))
#     plt.title('PCA: First Two Principal Components')
#     plt.xlabel('First Principal Component')
#     plt.ylabel('Second Principal Component')
#     plt.savefig(os.path.join(output_dir, 'pca_scatter.png'))
#     plt.close()
#
#     # 3D散点图展示前三个主成分
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=temp_numeric, cmap='coolwarm')
#     cbar = fig.colorbar(scatter, label='Temperature', ticks=[0, 1],
#                         format=plt.FuncFormatter(lambda x, p: '5C' if x == 0 else '10C'))
#     ax.set_xlabel('First Principal Component')
#     ax.set_ylabel('Second Principal Component')
#     ax.set_zlabel('Third Principal Component')
#     ax.set_title('PCA: First Three Principal Components')
#     plt.savefig(os.path.join(output_dir, 'pca_3d_scatter.png'))
#     plt.close()
#
#     logging.info("PCA analysis completed")
def create_distribution_plots(df, statistics, output_dir):
    # 基因型分布
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(statistics['genotype_counts'].keys()), y=list(statistics['genotype_counts'].values()))
    plt.title('Number of Plants per Genotype')
    plt.xlabel('Genotype')
    plt.ylabel('Number of Plants')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'genotype_distribution.png'))
    plt.close()

    # 作物类型分布
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(statistics['crop_type_counts'].keys()), y=list(statistics['crop_type_counts'].values()))
    plt.title('Number of Plants per Crop Type')
    plt.xlabel('Crop Type')
    plt.ylabel('Number of Plants')
    plt.xticks()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'crop_type_distribution.png'))
    plt.close()

    # 温度分布
    plt.figure(figsize=(8, 6))
    df['temperature'].value_counts().plot(kind='bar')
    plt.title('Distribution of Temperature Conditions')
    plt.xlabel('Temperature')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_distribution.png'))
    plt.close()


from scipy.interpolate import UnivariateSpline


def gaussian(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))


def modified_gompertz(x, a, b, c, d, e):
    return a * np.exp(-b * np.exp(-c * x)) - d * x + e


def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c

def sigmoid(x, L, x0, k, b):
    """
    L: the curve's maximum value
    x0: the x-value of the sigmoid's midpoint
    k: the steepness of the curve
    b: the y-offset of the curve
    """
    return L / (1 + np.exp(-k * (x - x0))) + b

def lognormal(x, s, loc, scale):
    return lognorm.pdf(x, s, loc, scale)


def weibull(x, c, loc, scale):
    return weibull_min.pdf(x, c, loc, scale)

def linear(x, a, b):
    return a * x + b
def fit_curve(x, y, crop_type, temperature):
    if temperature == '5C' and crop_type in ['Winter OSR', 'Leafy vegetable']:
        popt, _ = curve_fit(gaussian, x, y, p0=[max(y), np.mean(x), np.std(x)])
        y_fit = gaussian(x, *popt)
        method = 'Gaussian'

    elif temperature == '10C':
        if crop_type in ['Spring OSR', 'Semiwinter OSR', 'Spring fodder']:
            popt, _ = curve_fit(gaussian, x, y, p0=[max(y), np.mean(x), np.std(x)])
            y_fit = gaussian(x, *popt)
            method = 'Gaussian'
        elif crop_type == 'Leafy vegetable':
            lowess_fit = lowess(y, x, frac=0.2,it=10)  # frac参数控制平滑程度，值越大越平滑
            y_fit = lowess_fit[:, 1]  # 获取拟合后的y值
            method = 'LOWESS'
        elif crop_type == 'Swede':
            lowess_fit = lowess(y, x, frac=0.13,it=10)  # frac参数控制平滑程度，值越大越平滑
            y_fit = lowess_fit[:, 1]  # 获取拟合后的y值
            method = 'LOWESS'
            # popt, _ = curve_fit(gaussian, x, y, p0=[max(y), np.mean(x), np.std(x)])
            # y_fit = gaussian(x, *popt)
            # method = 'Gaussian'
        elif crop_type == 'Winter fodder':
            lowess_fit = lowess(y, x, frac=0.3,it=10)  # frac参数控制平滑程度，值越大越平滑
            y_fit = lowess_fit[:, 1]  # 获取拟合后的y值
            method = 'LOWESS'
        else:
            # 使用 Gompertz 拟合作为10C温度下的默认拟合
            p0 = [max(y), 1, 0.1, 0.01, np.mean(y)]
            popt, _ = curve_fit(modified_gompertz, x - x.min(), y, p0=p0, maxfev=10000)
            y_fit = modified_gompertz(x - x.min(), *popt)
            method = 'Gompertz'
    else:
        # 处理其他温度或默认情况
        p0 = [max(y), 1, 0.1, 0.01, np.mean(y)]
        popt, _ = curve_fit(modified_gompertz, x - x.min(), y, p0=p0, maxfev=10000)
        y_fit = modified_gompertz(x - x.min(), *popt)
        method = 'Gompertz'

    # 计算R²值
    r2 = r2_score(y, y_fit)
    return y_fit, method, r2



def create_daily_bud_count_chart(df, output_dir):
    logging.info("Creating daily bud count charts")
    df['date'] = pd.to_datetime(df['date'])

    daily_bud_count = df.groupby(['date', 'plant_id', 'crop_type', 'temperature']).size().reset_index(
        name='total_count')
    daily_bud_count['daily_count'] = daily_bud_count['total_count'] / 3  # Assuming 3 angles

    df_grouped = daily_bud_count.groupby(['date', 'crop_type', 'temperature'])['daily_count'].mean().reset_index()

    temperatures = df_grouped['temperature'].unique()
    crop_types = df_grouped['crop_type'].unique()

    # Determine global y-axis max and x-axis range
    y_max = df_grouped['daily_count'].max() * 1.1  # Add 10% margin
    x_min = df_grouped['date'].min()
    x_max = df_grouped['date'].max()

    # Define the color palette
    color_palette = plt.cm.get_cmap('tab10')
    crop_color_map = {crop: color_palette(i) for i, crop in enumerate(crop_types)}

    for temperature in temperatures:
        fig, ax = plt.subplots(figsize=(15, 10))

        for crop in crop_types:
            data = df_grouped[(df_grouped['crop_type'] == crop) & (df_grouped['temperature'] == temperature)]
            data = data.sort_values('date')

            if len(data) > 5:
                x = mdates.date2num(data['date'])
                y = data['daily_count'].values

                color = crop_color_map[crop]
                ax.scatter(data['date'], y, alpha=0.5, color=color, s=30)

                try:
                    # 调用修改后的 fit_curve 函数
                    y_fit, method, r2 = fit_curve(x, y, crop, temperature)

                    label = f'{crop} ({method}, R² = {r2:.2f})'
                    ax.plot(mdates.num2date(x), y_fit, '-', color=color)

                    # Find max point
                    max_index = np.argmax(y_fit)
                    max_date = mdates.num2date(x[max_index])
                    max_value = y_fit[max_index]

                    ax.scatter(max_date, max_value, color=color, s=100, zorder=5)
                    label += f', Max: {max_date.strftime("%Y-%m-%d")}'

                    ax.plot([], [], color=color, label=label)

                except RuntimeError:
                    logging.warning(f"Curve fitting failed for {crop} at {temperature}")
                    continue

        ax.set_title(f'Average Daily Bud Count Over Time ({temperature})', fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Average Bud Count', fontsize=14)
        ax.legend(loc='upper right', fontsize=12)  # 图例移到右上角
        ax.set_ylim(0, y_max)  # Set uniform y-axis limit
        ax.set_xlim(x_min, x_max)  # Set uniform x-axis limit

        # Format x-axis
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=12)  # 横坐标水平，字体增大

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'daily_bud_count_chart_{temperature}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    logging.info("Daily bud count charts created successfully")

def main():
    results_dir = 'results2'
    output_dir = create_output_directory()
    logging.info(f"Output directory created: {output_dir}")

    try:
        logging.info("Starting data processing")
        df, statistics = process_all_files(results_dir)
        if df.empty:
            logging.warning("No data was processed. Please check your input files and paths.")
            return

        logging.info(f"Processed data for {df['plant_id'].nunique()} unique plants.")

        # 打印统计信息
        logging.info(f"Number of files processed: {statistics['num_files']}")
        logging.info(f"Number of unique genotypes: {statistics['num_genotypes']}")
        logging.info(f"Number of unique crop types: {statistics['num_crop_types']}")
        logging.info("Number of plants per crop type:")
        for crop_type, count in statistics['crop_type_counts'].items():
            logging.info(f"  {crop_type}: {count}")
        logging.info("Number of plants per genotype:")
        for genotype, count in statistics['genotype_counts'].items():
            logging.info(f"  {genotype}: {count}")

        # 创建分布图表
        # create_distribution_plots(df, statistics, output_dir)

        # 原有的图表创建函数
        chart_functions = [
            (create_daily_bud_count_chart, df),
            (create_temperature_comparison, df),
            (create_genotype_temperature_heatmap, df),
            (create_bud_count_distribution, df),
            (create_improved_crop_type_comparison, df),  # 替换原来的函数
            (create_improved_genotype_comparison, df),
            (create_bud_position_analysis, df),
            (create_bud_height_distribution, df),
            (create_bud_count_over_time, df)
        ]

        for func, data in tqdm(chart_functions, desc="Creating charts"):
            logging.info(f"Creating chart: {func.__name__}")
            func(data, output_dir)

        perform_clustering_analysis(df, output_dir)

        # # 执行聚类分析和PCA
        # perform_clustering_analysis(df, output_dir)
        # perform_pca_analysis(df, output_dir)

        logging.info(f"All analyses completed. Results have been saved in the '{output_dir}' directory.")
    except Exception as e:
        logging.error(f"An error occurred during processing: {str(e)}")
        logging.exception("Exception details:")


if __name__ == "__main__":
    main()


