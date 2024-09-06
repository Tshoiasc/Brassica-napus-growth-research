import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv("bud_tracking_and_branch_data.csv")

def process_plant(plant_data):
    # 将Date列转换为datetime类型
    plant_data['Date'] = pd.to_datetime(plant_data['Date'])

    # 找出最后一天的日期
    last_day = plant_data['Date'].max()

    # 去除最后一天的数据
    plant_data = plant_data[plant_data['Date'] < last_day]

    # 只考虑 'Branch Level' 为 2 的数据
    level_2_data = plant_data[plant_data['Branch Level'] == 2]

    if level_2_data.empty:
        return np.nan

    # 找出最大的花蕾数并除以3
    max_buds = level_2_data.groupby('Date').size().max() / 3

    return max_buds

# 对每个植株应用处理函数
results = df.groupby('plant_id').apply(process_plant)

# 创建结果DataFrame
results_df = pd.DataFrame({'plant_id': results.index, 'max_buds': results.values})

# 读取额外的Excel文件
df1 = pd.read_excel("BR017 Branching data for Chen Aug 2024.xlsx")

# 计算combo1和combo2
df1['combo1'] = df1['Branches with pods (2)'] + df1['Branches with flowers/buds (3)'] + df1['Other Nodes (4)']
df1['combo2'] = df1['Branches with pods (2)'] + df1['Branches with flowers/buds (3)']

# 获取crop_type和geno_type
crop_type_dict = df.groupby('plant_id')['crop_type'].first().to_dict()
geno_type_dict = df.groupby('plant_id')['geno_type'].first().to_dict()

# 创建最终输出数据框
output_df = pd.DataFrame({
    'plant_id': df1['identifier'],
    'crop_type': df1['identifier'].map(crop_type_dict),
    'geno_type': df1['identifier'].map(geno_type_dict),
    'max_buds': df1['identifier'].map(results.to_dict()),
    'combo1': df1['combo1'],
    'combo2': df1['combo2']
})

# 删除包含NaN值的行
output_df = output_df.dropna()

# 输出到CSV文件
output_df.to_csv('output_results.csv', index=False)

print("处理完成，结果已保存到 output_results.csv")

# 打印前几行结果用于验证
print(output_df.head())