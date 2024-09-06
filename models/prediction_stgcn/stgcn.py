import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import networkx as nx
from torch.optim.lr_scheduler import ReduceLROnPlateau

class SpatioTemporalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpatioTemporalGNN, self).__init__()
        self.spatial_conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.spatial_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.temporal_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.temporal_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, spatial_edge_index, temporal_edge_index):
        x = F.relu(self.bn1(self.spatial_conv1(x, spatial_edge_index)))
        x = F.relu(self.bn2(self.spatial_conv2(x, spatial_edge_index)))
        x = F.relu(self.bn3(self.temporal_conv1(x, temporal_edge_index)))
        x = F.relu(self.bn4(self.temporal_conv2(x, temporal_edge_index)))
        x = self.fc(x)
        return x

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    
    # Rename columns to match our code
    df = df.rename(columns={
        'Branch Level': 'Branch_Level',
        'New Bud ID': 'Branch_ID',
        'Days_Since_Start': 'time'
    })
    
    # Select only the columns we need
    columns_to_keep = ['Date', 'plant_id', 'Branch_Level', 'Length', 'Angle', 
                       'vernalization_temp', 'Branch_ID', 'x_coord', 'y_coord', 'time']
    df = df[columns_to_keep]
    df['Angle'] = df['Angle'].abs()
    # Ensure correct data types
    df['Branch_Level'] = df['Branch_Level'].astype(int)
    df['Branch_ID'] = df['Branch_ID'].astype(int)
    df['time'] = df['time'].astype(int)
    
    print("Columns in the dataframe:", df.columns.tolist())
    print("First few rows of the dataframe:")
    print(df.head())
    
    return df

def visualize_predictions_vs_true(predictions, targets, features=['Length', 'x_coord', 'y_coord', 'Angle']):
    num_features = predictions.shape[1]  # 预测值的维度
    plt.figure(figsize=(20, 5*num_features))
    
    for i in range(num_features):
        plt.subplot(num_features, 1, i + 1)  # 根据特征数量创建子图
        plt.scatter(targets[:, i].cpu().numpy(), predictions[:, i].cpu().numpy(), label=f'Predicted vs True {features[i]}')
        plt.plot([targets[:, i].min().cpu().numpy(), targets[:, i].max().cpu().numpy()],
                 [targets[:, i].min().cpu().numpy(), targets[:, i].max().cpu().numpy()], 
                 color='red', linestyle='--', label='y=x')  # 绘制y=x的斜线
        
        plt.xlabel(f'True {features[i]}')
        plt.ylabel(f'Predicted {features[i]}')
        plt.title(f'Predicted vs True {features[i]}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('output/predictions_vs_true.png')
    plt.close()

    print("Prediction vs True scatter plots saved as 'output/predictions_vs_true.png'.")


def create_graph_data(df):
    graphs = []
    for plant_id in df['plant_id'].unique():
        plant_data = df[df['plant_id'] == plant_id]
        max_time = plant_data['time'].max()
        for time in range(int(max_time)):
            current_data = plant_data[plant_data['time'] == time]
            next_data = plant_data[plant_data['time'] == time + 1]
            if len(current_data) > 0 and len(next_data) > 0:
                min_rows = min(len(current_data), len(next_data))
                current_data = current_data.iloc[:min_rows]
                next_data = next_data.iloc[:min_rows]

                features = ['Length', 'x_coord', 'y_coord', 'vernalization_temp']
                level_2_mask = current_data['Branch_Level'] == 2
                x = current_data[features].values
                x = np.column_stack([x, np.where(level_2_mask, current_data['Angle'], 0)])
                x = torch.tensor(x, dtype=torch.float)
                
                y_length = torch.tensor(next_data['Length'].values, dtype=torch.float)
                y_x_coord = torch.tensor(next_data['x_coord'].values, dtype=torch.float)
                y_y_coord = torch.tensor(next_data['y_coord'].values, dtype=torch.float)
                y_angle = torch.tensor(np.where(next_data['Branch_Level'] == 2, next_data['Angle'], 0), dtype=torch.float)
                y = torch.stack([y_length, y_x_coord, y_y_coord, y_angle], dim=1)
                
                num_branches = len(current_data)
                spatial_edge_index = torch.combinations(torch.arange(num_branches), r=2).t().contiguous()
                
                if time > 0:
                    prev_data = plant_data[plant_data['time'] == time - 1]
                    temporal_edge_index = torch.tensor([[i, i] for i in range(num_branches) if i < len(prev_data)]).t().contiguous()
                else:
                    temporal_edge_index = torch.empty((2, 0), dtype=torch.long)
                
                branch_levels = torch.tensor(current_data['Branch_Level'].values, dtype=torch.long)
                
                graphs.append(Data(x=x, y=y, spatial_edge_index=spatial_edge_index, temporal_edge_index=temporal_edge_index, 
                                   plant_id=plant_id, time=time, branch_levels=branch_levels))
    
    return graphs

# Add this function for debugging
def print_data_info(df):
    print("\nDataset Information:")
    print(f"Total number of rows: {len(df)}")
    print(f"Number of unique plants: {df['plant_id'].nunique()}")
    print(f"Number of unique days: {df['time'].nunique()}")
    print("\nSample of data for the first plant:")
    first_plant = df['plant_id'].iloc[0]
    print(df[df['plant_id'] == first_plant].head(10))

def custom_loss(pred, target, branch_levels):
    mse_loss = nn.MSELoss(reduction='none')
    
    # Loss for branch length, x_coord, and y_coord
    loss_length_pos = mse_loss(pred[:, :3], target[:, :3]).mean()
    
    # Loss for branch angle (only for level 2 branches)
    mask_level_2 = (branch_levels == 2)
    if mask_level_2.sum() > 0:
        loss_angle = mse_loss(pred[mask_level_2, 3], target[mask_level_2, 3]).mean()
    else:
        loss_angle = torch.tensor(0.0, device=pred.device)
    
    return loss_length_pos + loss_angle

def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.spatial_edge_index, batch.temporal_edge_index)
        loss = custom_loss(out, batch.y, batch.branch_levels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(train_loader.dataset)

def validate(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch.x, batch.spatial_edge_index, batch.temporal_edge_index)
            loss = custom_loss(out, batch.y, batch.branch_levels)
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(val_loader.dataset)

# Load and preprocess data
file_path = 'daily_interpolated_bud_tracking_filled.csv'  # Replace with your actual file path
df = load_and_preprocess_data(file_path)
print_data_info(df)  # Add this line to print debugging information
graph_data = create_graph_data(df)


# Split data into train and validation sets
train_data = graph_data[:int(0.8 * len(graph_data))]
val_data = graph_data[int(0.8 * len(graph_data)):]

# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# Initialize model
input_dim = 5  # Length, x_coord, y_coord, vernalization_temp, Angle (0 for non-level 2)
hidden_dim = 64
output_dim = 4  # Predicting Length, x_coord, y_coord, and Angle
model = SpatioTemporalGNN(input_dim, hidden_dim, output_dim)

# Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        
        
# Training loop
num_epochs = 200
train_losses = []
val_losses = []



def train_with_early_stopping(model, train_loader, val_loader, optimizer, scheduler, num_epochs, patience=20):
    best_val_loss = float('inf')
    no_improve_epochs = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve_epochs += 1
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return train_losses, val_losses

def plot_training_curve(train_losses, val_losses):
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, [loss/10000 for loss in train_losses], label='Train Loss')
    plt.plot(epochs, [loss/10000 for loss in val_losses], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (1e-4)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/training_loss.png')
    plt.close()

# 在主程序中，修改训练部分：
train_losses, val_losses = train_with_early_stopping(model, train_loader, val_loader, optimizer, scheduler, num_epochs, patience=20)

# 绘制损失曲线
plot_training_curve(train_losses, val_losses)






# Visualize the effect of vernalization temperature on growth for different branch levels
plt.figure(figsize=(12, 6))
for level in [1, 2, 3]:
    level_data = df[df['Branch_Level'] == level]
    sns.scatterplot(data=level_data, x='vernalization_temp', y='Length', label=f'Level {level}')
plt.title('Effect of Vernalization Temperature on Branch Length by Branch Level')
plt.savefig('output/vernalization_effect_by_level.png')
plt.close()

# Visualize branch angle change over time for level 2 branches
plt.figure(figsize=(12, 6))
level_2_data = df[df['Branch_Level'] == 2]
for plant_id in level_2_data['plant_id'].unique()[:5]:  # Plot for the first 5 plants
    plant_data = level_2_data[level_2_data['plant_id'] == plant_id]
    for branch in plant_data['Branch_ID'].unique():
        branch_data = plant_data[plant_data['Branch_ID'] == branch]
        plt.plot(branch_data['time'], branch_data['Angle'], label=f'Plant {plant_id}, Branch {branch}')
plt.xlabel('Time')
plt.ylabel('Branch Angle')
plt.title('Branch Angle Change Over Time (Level 2 Branches)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('output/branch_angle_change_level_2.png')
plt.close()

print("Training complete. Loss plot saved as 'output/training_loss.png'.")
print("Vernalization effect by branch level plot saved as 'output/vernalization_effect_by_level.png'.")
print("Branch angle change plot for level 2 branches saved as 'output/branch_angle_change_level_2.png'.")

def visualize_graph_structure(graph_data, df, num_graphs=5):
    plt.figure(figsize=(20, 6*num_graphs))
    for i, data in enumerate(graph_data[:num_graphs]):
        G = nx.Graph()
        
        # 使用 spatial_edge_index 而不是 edge_index
        if data.spatial_edge_index is not None:
            G.add_edges_from(data.spatial_edge_index.t().tolist())
        
        plt.subplot(num_graphs, 1, i+1)
        pos = nx.spring_layout(G, k=0.5, iterations=50)  # 增加 k 值和迭代次数以分散节点
        
        # 获取当前图的 plant_id 和 time
        plant_id = data.plant_id
        time = data.time
        
        # 从原始数据中获取对应的 Branch_ID
        current_data = df[(df['plant_id'] == plant_id) & (df['time'] == time)]
        branch_ids = current_data['Branch_ID'].values
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color=data.branch_levels, cmap=plt.cm.viridis, node_size=300)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos)
        
        # 添加节点标签
        labels = {node: f"B{branch_ids[node]}" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title(f"Graph Structure for Plant {data.plant_id} at Time {data.time}")
        plt.axis('off')  # 关闭坐标轴
    
    plt.tight_layout()
    plt.savefig('output/graph_structures.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_branch_growth(df):
    plt.figure(figsize=(15, 10))
    for plant_id in df['plant_id'].unique()[:5]:  # Plot for the first 5 plants
        plant_data = df[df['plant_id'] == plant_id]
        for branch in plant_data['Branch_ID'].unique():
            branch_data = plant_data[plant_data['Branch_ID'] == branch]
            plt.plot(branch_data['time'], branch_data['Length'], 
                     label=f'Plant {plant_id}, Branch {branch}')
    
    plt.xlabel('Time')
    plt.ylabel('Branch Length')
    plt.title('Branch Growth Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('output/branch_growth.png')
    plt.close()

def visualize_branch_angles(df):
    plt.figure(figsize=(15, 10))
    level_2_data = df[df['Branch_Level'] == 2]
    for plant_id in level_2_data['plant_id'].unique()[:5]:  # Plot for the first 5 plants
        plant_data = level_2_data[level_2_data['plant_id'] == plant_id]
        for branch in plant_data['Branch_ID'].unique():
            branch_data = plant_data[plant_data['Branch_ID'] == branch]
            plt.plot(branch_data['time'], branch_data['Angle'], 
                     label=f'Plant {plant_id}, Branch {branch}')
    
    plt.xlabel('Time')
    plt.ylabel('Branch Angle')
    plt.title('Branch Angle Change Over Time (Level 2 Branches)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('output/branch_angle_change.png')
    plt.close()

def visualize_vernalization_effect(df):
    plt.figure(figsize=(15, 10))
    sns.boxplot(x='vernalization_temp', y='Length', hue='Branch_Level', data=df)
    plt.title('Effect of Vernalization Temperature on Branch Length by Branch Level')
    plt.savefig('output/vernalization_effect.png')
    plt.close()

# After your training loop, add these visualization calls:

visualize_graph_structure(graph_data, df)
visualize_branch_growth(df)
visualize_branch_angles(df)
visualize_vernalization_effect(df)

print("Graph structures visualization saved as 'output/graph_structures.png'.")
print("Branch growth visualization saved as 'output/branch_growth.png'.")
print("Branch angle change visualization saved as 'output/branch_angle_change.png'.")
print("Vernalization effect visualization saved as 'output/vernalization_effect.png'.")

# 获取预测值和真实值
model.eval()
predictions_list = []
targets_list = []
with torch.no_grad():
    for batch in val_loader:  # 使用验证集进行预测
        out = model(batch.x, batch.spatial_edge_index, batch.temporal_edge_index)
        predictions_list.append(out)
        targets_list.append(batch.y)

# 将预测值和真实值拼接在一起
predictions = torch.cat(predictions_list, dim=0)
targets = torch.cat(targets_list, dim=0)

# 调用可视化函数
visualize_predictions_vs_true(predictions, targets)

print("ok")