import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df = pd.read_csv('output_results.csv')

# 查看每个基因型和作物类型的样本数
print(df.groupby('geno_type').size())
print(df.groupby('crop_type').size())

# 查看缺失值
print(df.isnull().sum())

# 去除缺失值
df_clean = df.dropna()


df_clean['error1'] = abs(df_clean['max_buds'] - df_clean['combo1'])
df_clean['error2'] = abs(df_clean['max_buds'] - df_clean['combo2'])
df_clean['relative_error1'] = df_clean['error1'] / df_clean['max_buds']
df_clean['relative_error2'] = df_clean['error2'] / df_clean['max_buds']
df_clean['best_relative_error'] = df_clean[['relative_error1', 'relative_error2']].min(axis=1)



# 再次查看样本数
print(df_clean.groupby('geno_type').size())
print(df_clean.groupby('crop_type').size())



# 选择样本数大于等于5的基因型
geno_types_to_analyze = df_clean['geno_type'].value_counts()[df_clean['geno_type'].value_counts() >= 5].index

df_analysis = df_clean[df_clean['geno_type'].isin(geno_types_to_analyze)]

# 对这些组进行Kruskal-Wallis检验
from scipy import stats

# 对 best_relative_error 进行正态性检验
stat, p = stats.shapiro(df_clean['best_relative_error'])

print("Shapiro-Wilk test results:")
print(f"Statistic: {stat}, p-value: {p}")

# 如果p值小于0.05，我们就拒绝数据符合正态分布的假设
if p > 0.05:
    print("Data is likely normally distributed (fail to reject H0)")
else:
    print("Data is not normally distributed (reject H0)")

# 我们也可以对每个作物类型进行单独的正态性检验
for crop in df_clean['crop_type'].unique():
    data = df_clean[df_clean['crop_type'] == crop]['best_relative_error']
    stat, p = stats.shapiro(data)
    print(f"\nShapiro-Wilk test for {crop}:")
    print(f"Statistic: {stat}, p-value: {p}")


# 1. 原始 Kruskal-Wallis 检验
kruskal_result = stats.kruskal(*[group['best_relative_error'].values for name, group in df_clean.groupby('crop_type')])
print("Original Kruskal-Wallis result:", kruskal_result)

# 2. 随机抽样方法
min_samples = df_clean['crop_type'].value_counts().min()
balanced_df = df_clean.groupby('crop_type').apply(lambda x: x.sample(min_samples) if len(x) > min_samples else x).reset_index(drop=True)
kruskal_balanced = stats.kruskal(*[group['best_relative_error'].values for name, group in balanced_df.groupby('crop_type')])
print("Balanced Kruskal-Wallis result:", kruskal_balanced)

# 3. 分组分析
winter_osr = df_clean[df_clean['crop_type'] == 'Winter OSR']
other_crops = df_clean[df_clean['crop_type'] != 'Winter OSR']
t_test_result = stats.ttest_ind(winter_osr['best_relative_error'], other_crops['best_relative_error'])
print("T-test Winter OSR vs Others:", t_test_result)

# 4. 可视化
plt.figure(figsize=(12, 6))
sns.boxplot(x='crop_type', y='best_relative_error', data=df_clean)
plt.title('Relative Error by Crop Type (Original Data)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('crop_type_relative_error_original.png')
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(x='crop_type', y='best_relative_error', data=balanced_df)
plt.title('Relative Error by Crop Type (Balanced Data)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('crop_type_relative_error_balanced.png')
plt.close()

# 5. 效应大小计算 (Epsilon-squared for Kruskal-Wallis)
def epsilon_squared(h_statistic, n):
    return h_statistic / (n - 1)

n_total = len(df_clean)
effect_size = epsilon_squared(kruskal_result.statistic, n_total)
print(f"Effect size (Epsilon-squared): {effect_size}")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# 假设 df_clean 是我们的清洗后的数据框

# 1. 详细的统计分析
def detailed_stats(group):
    return pd.Series({
        'mean': group.mean(),
        'median': group.median(),
        'std': group.std(),
        'min': group.min(),
        'max': group.max()
    })

crop_type_stats = df_clean.groupby('crop_type')['best_relative_error'].apply(detailed_stats)
geno_type_stats = df_clean.groupby('geno_type')['best_relative_error'].apply(detailed_stats)

# 2. 相关性分析
correlation_matrix = df_clean[['max_buds', 'combo1', 'combo2', 'best_relative_error']].corr()
import statsmodels.api as sm
# 3. 多变量分析 (简化版)
from statsmodels.formula.api import ols
model = ols('best_relative_error ~ C(crop_type) + C(geno_type)', data=df_clean).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# 4. 聚类分析
kmeans = KMeans(n_clusters=3)
df_clean['cluster'] = kmeans.fit_predict(df_clean[['best_relative_error']])

# 5. 异常值分析
Q1 = df_clean['best_relative_error'].quantile(0.25)
Q3 = df_clean['best_relative_error'].quantile(0.75)
IQR = Q3 - Q1
outliers = df_clean[(df_clean['best_relative_error'] < (Q1 - 1.5 * IQR)) | (df_clean['best_relative_error'] > (Q3 + 1.5 * IQR))]

# 6. 机器学习方法
rf = RandomForestRegressor(n_estimators=100, random_state=42)
features = pd.get_dummies(df_clean[['crop_type', 'geno_type']])
rf.fit(features, df_clean['best_relative_error'])
feature_importance = pd.Series(rf.feature_importances_, index=features.columns).sort_values(ascending=False)

# 7. 误差类型分析
df_clean['error_type'] = np.where(df_clean['max_buds'] > df_clean['combo1'], 'underestimate', 'overestimate')

# 可视化
plt.figure(figsize=(12, 6))
sns.boxplot(x='crop_type', y='best_relative_error', hue='error_type', data=df_clean)
plt.title('Relative Error by Crop Type and Error Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('error_type_analysis.png')
plt.close()

# 输出结果
print(crop_type_stats)
print(geno_type_stats)
print(correlation_matrix)
print(anova_table)
print(feature_importance.head(10))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 假设 df_clean 是我们的清洗后的数据框

# 1. 基因型相对误差箱线图
plt.figure(figsize=(20, 10))
sns.boxplot(x='geno_type', y='best_relative_error', data=df_clean)
plt.title('Relative Error by Genotype')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('genotype_relative_error.png')
plt.close()

# 2. 基因型和作物类型的热图
pivot = df_clean.pivot_table(values='best_relative_error', index='geno_type', columns='crop_type', aggfunc='mean')
plt.figure(figsize=(15, 15))
sns.heatmap(pivot, annot=True, cmap='YlOrRd')
plt.title('Mean Relative Error by Genotype and Crop Type')
plt.tight_layout()
plt.savefig('genotype_croptype_heatmap.png')
plt.close()

# 3. 基因型的相对误差分布
plt.figure(figsize=(20, 10))
for geno in df_clean['geno_type'].unique():
    sns.kdeplot(df_clean[df_clean['geno_type'] == geno]['best_relative_error'], label=geno)
plt.title('Distribution of Relative Error by Genotype')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('genotype_error_distribution.png')
plt.close()

# 4. 基因型的样本量条形图
plt.figure(figsize=(20, 10))
df_clean['geno_type'].value_counts().plot(kind='bar')
plt.title('Sample Size by Genotype')
plt.ylabel('Sample Count')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('genotype_sample_size.png')
plt.close()

# 5. 基因型与max_buds的散点图
plt.figure(figsize=(20, 12))  # 增加图表高度以适应底部的图例
sns.scatterplot(x='max_buds', y='best_relative_error', hue='geno_type', data=df_clean)
plt.title('Relative Error vs Max Buds by Genotype')

# 移动图例到底部
plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=5, borderaxespad=0.)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # 调整底部边距以容纳图例
plt.savefig('genotype_maxbuds_scatter.png', dpi=300, bbox_inches='tight')
plt.close()



# 1. Kruskal-Wallis 检验（适用于非正态分布数据）
genotype_kruskal = stats.kruskal(*[group['best_relative_error'].values for name, group in df_clean.groupby('geno_type')])
print("Genotype Kruskal-Wallis result:", genotype_kruskal)

# 2. 单因素方差分析（ANOVA）
genotype_f_statistic, genotype_p_value = stats.f_oneway(*[group['best_relative_error'].values for name, group in df_clean.groupby('geno_type')])
print(f"Genotype ANOVA result: F-statistic = {genotype_f_statistic}, p-value = {genotype_p_value}")

# 3. 事后检验：Tukey HSD
# genotype_tukey = pairwise_tukeyhsd(df_clean['best_relative_error'], df_clean['geno_type'])
# print("Genotype Tukey HSD result:")
# print(genotype_tukey)

# 4. 效应大小计算 (Epsilon-squared for Kruskal-Wallis)
def epsilon_squared(h_statistic, n):
    return h_statistic / (n - 1)

n_total_genotype = len(df_clean)
genotype_effect_size = epsilon_squared(genotype_kruskal.statistic, n_total_genotype)
print(f"Genotype effect size (Epsilon-squared): {genotype_effect_size}")

# # 5. 可视化：基因型间的多重比较
# plt.figure(figsize=(12, 8))
# genotype_tukey.plot_simultaneous()
# plt.title("Tukey HSD: Multiple Comparison of Genotypes")
# # 调节纵坐标标签的字体大小
# plt.yticks(fontsize=4)
# # 调节纵坐标间隔
# plt.subplots_adjust(bottom=0.2)
# plt.tight_layout()
# plt.savefig('genotype_tukey_comparison.png')
# plt.close()

import numpy as np


def calculate_accuracy(actual, predicted, tolerance=1):
    """
    计算准确率

    参数:
    actual: 实际值（max_buds）
    predicted: 预测值（combo1 或 combo2）
    tolerance: 容忍的绝对误差，默认为1

    返回:
    准确率百分比
    """
    absolute_diff = np.abs(actual - predicted)

    # 对于小数值（比如 max_buds <= 5），我们使用绝对差异
    small_values_mask = actual <= 5
    small_values_accuracy = np.mean(absolute_diff[small_values_mask] <= tolerance) * 100

    # 对于大数值，我们使用相对差异
    large_values_mask = actual > 5
    large_values_accuracy = np.mean(absolute_diff[large_values_mask] / actual[large_values_mask] <= tolerance / 5) * 100

    # 计算总体准确率
    total_accuracy = (np.sum(small_values_mask) * small_values_accuracy +
                      np.sum(large_values_mask) * large_values_accuracy) / len(actual)

    return total_accuracy


# 计算 combo1 和 combo2 的准确率
accuracy_combo1 = calculate_accuracy(df_clean['max_buds'], df_clean['combo1'])
accuracy_combo2 = calculate_accuracy(df_clean['max_buds'], df_clean['combo2'])

print(f"Combo1 accuracy: {accuracy_combo1:.2f}%")
print(f"Combo2 accuracy: {accuracy_combo2:.2f}%")

# 计算不同容忍度下的准确率
for tolerance in [0,0.5, 1, 1.5, 2,3 ,4, 5]:
    acc1 = calculate_accuracy(df_clean['max_buds'], df_clean['combo1'], tolerance)
    acc2 = calculate_accuracy(df_clean['max_buds'], df_clean['combo2'], tolerance)
    print(f"tolerance {tolerance}:")
    print(f"  Combo1 accuracy: {acc1:.2f}%")
    print(f"  Combo2 accuracy: {acc2:.2f}%")


df_clean['error_diff'] = df_clean['relative_error1'] - df_clean['relative_error2']
genotype_error_diff = df_clean.groupby('geno_type')['error_diff'].mean().sort_values()

plt.figure(figsize=(15, 10))
genotype_error_diff.plot(kind='bar')
plt.title('Difference in Relative Error (Combo1 - Combo2) by Genotype')
plt.xlabel('Genotype')
plt.ylabel('Error Difference')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('genotype_error_difference.png')
plt.close()

from sklearn.cluster import KMeans

X = df_clean.groupby('geno_type')[['relative_error1', 'relative_error2']].mean()
crop_types = df_clean.groupby('geno_type')['crop_type'].first()  # 获取每个基因型对应的作物类型

# 执行K-means聚类
kmeans = KMeans(n_clusters=3, random_state=42)
X['cluster'] = kmeans.fit_predict(X)

# 创建颜色映射
unique_crop_types = crop_types.unique()
color_map = plt.cm.get_cmap('tab10')  # 使用 tab10 颜色图，最多支持 10 种不同的颜色
color_dict = {crop: color_map(i/len(unique_crop_types)) for i, crop in enumerate(unique_crop_types)}



# 绘制散点图
for crop in unique_crop_types:
    mask = crop_types == crop
    plt.scatter(X.loc[mask, 'relative_error1'], X.loc[mask, 'relative_error2'],
                c=[color_dict[crop]], label=crop, alpha=0.7)

# 添加 y=x 线
max_error = max(X['relative_error1'].max(), X['relative_error2'].max())
plt.plot([0, max_error], [0, max_error], 'k--', label='y=x')

plt.xlabel('Relative Error (Combo1)')
plt.ylabel('Relative Error (Combo2)')
plt.title('Genotype Clusters based on Relative Errors')

# # 添加基因型标签
# for i, txt in enumerate(X.index):
#     plt.annotate(txt, (X['relative_error1'][i], X['relative_error2'][i]), fontsize=8)

plt.legend()
plt.tight_layout()
plt.savefig('genotype_clusters_with_crop_types.png', dpi=300)
plt.close()




# 计算每个基因型的平均 Combo1 和 Combo2 相对误差
genotype_errors = df_clean.groupby('geno_type')[['relative_error1', 'relative_error2']].mean()

# 找出 Combo2 误差比 Combo1 误差小的基因型
better_combo2_genotypes = genotype_errors[genotype_errors['relative_error2'] < genotype_errors['relative_error1']]

# 排序并输出结果
better_combo2_genotypes['error_diff'] = better_combo2_genotypes['relative_error1'] - better_combo2_genotypes['relative_error2']
better_combo2_genotypes = better_combo2_genotypes.sort_values('error_diff', ascending=False)

print("Genotypes where Combo2 performs better than Combo1:")
print(better_combo2_genotypes)

# 获取这些基因型的原始 Combo1 和 Combo2 值
genotypes_to_analyze = better_combo2_genotypes.index
combo_values = df_clean[df_clean['geno_type'].isin(genotypes_to_analyze)].groupby('geno_type')[['combo1', 'combo2', 'max_buds']].mean()

# 合并误差和实际值
result = pd.concat([better_combo2_genotypes, combo_values], axis=1)

print("\nDetailed comparison for genotypes where Combo2 is better:")
print(result)

# 可视化比较
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = range(len(result))

plt.bar(index, result['combo1'], bar_width, label='Combo1', alpha=0.8)
plt.bar([i + bar_width for i in index], result['combo2'], bar_width, label='Combo2', alpha=0.8)
plt.bar([i + 2*bar_width for i in index], result['max_buds'], bar_width, label='Max Buds', alpha=0.8)

plt.xlabel('Genotype')
plt.ylabel('Value')
plt.title('Comparison of Combo1, Combo2, and Max Buds for Genotypes where Combo2 is Better')
plt.xticks([i + bar_width for i in index], result.index, rotation=90)
plt.legend()

plt.tight_layout()
plt.savefig('combo_comparison_for_better_combo2.png')
plt.close()

# 计算和输出总体统计
total_genotypes = len(genotype_errors)
better_combo2_count = len(better_combo2_genotypes)
percentage = (better_combo2_count / total_genotypes) * 100

print(f"\nOut of {total_genotypes} total genotypes, {better_combo2_count} ({percentage:.2f}%) have smaller errors with Combo2.")