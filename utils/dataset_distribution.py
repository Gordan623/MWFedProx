import matplotlib.pyplot as plt
import numpy as np
import json

# 读取 JSON 文件
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 绘制比例图
def plot_class_proportions(json_data):
    plt.figure(figsize=(18, 10))
    client_ids = []
    client_proportions = []
    
    for client_id in json_data['client_stats'].keys():
        client = json_data['client_stats'][client_id]
        train_distribution = client['train_class_distribution']
        client_ids.append(f"Client {client_id} / {client['train_size']}")
        
        # 计算比例
        total_train = client['train_size']
        proportions = [train_distribution.get(str(i), 0) / total_train for i in range(10)]
        client_proportions.append(proportions)
    
    # 转置数据以按类别分组
    transposed_data = np.array(client_proportions).T
    
    # 绘制堆叠柱状图
    colors = ['steelblue', 'skyblue', 'darkorange', 'sandybrown', 'green',
              'lightgreen', 'red', 'salmon', 'mediumpurple', 'blueviolet']
    bottom = np.zeros(len(client_ids))
    for i in range(10):
        plt.bar(client_ids, transposed_data[i], color=colors[i], label=f'Class {i}', bottom=bottom)
        bottom += transposed_data[i]
    
    # 设置自定义字体
    plt.xlabel('Client ID', fontsize=14, fontfamily='Arial', fontweight='bold')
    plt.ylabel('Proportion of Data', fontsize=14, fontfamily='Arial', fontweight='bold')
    plt.title('Proportion of Each Class per Client', fontsize=16, fontfamily='Arial', fontweight='bold', pad=20)
    
    # 设置图例
    plt.legend(title='Class ID', loc='upper right', fontsize=10, frameon=True, facecolor='white', shadow=True)
    
    # 设置坐标轴字体
    plt.xticks(rotation=45, ha='right', fontsize=10, fontfamily='Arial')
    plt.yticks(fontsize=12, fontfamily='Arial')
    
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.show()

# 示例用法
if __name__ == "__main__":
    json_data = load_json('/data/yxli/datasets/cifar10_nc30_distdir0.1_blc1/stats.json')
    plot_class_proportions(json_data)