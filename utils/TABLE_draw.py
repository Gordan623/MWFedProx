import os
import re
import pandas as pd

def parse_log(log_path):
    """
    解析日志文件，提取每个宽度的测试准确率、损失值以及模拟轮次时间成本。
    """
    data = {
        "width": [],
        "test_acc": [],
        "test_loss": [],
        "time": []
    }
    total_time = 0.0  # 用于累加时间

    with open(log_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            # 匹配模拟轮次时间成本的行
            time_match = re.search(r'Simulated round time cost: (\d+\.\d+)', line)
            if time_match:
                round_time = float(time_match.group(1)) / 1000  # 转换为10^3秒
                total_time += round_time  # 累加时间

            # 匹配宽度、准确率和损失值的行
            width_match = re.search(r'width:(\d+\.\d+)', line)
            if width_match and total_time is not None:
                width = width_match.group(1)
                test_acc = re.search(r'test acc: (\d+\.\d+)', line).group(1)
                test_loss = re.search(r'test_loss: (\d+\.\d+)', line).group(1)
                data["width"].append(width)
                data["test_acc"].append(float(test_acc))
                data["test_loss"].append(float(test_loss))
                data["time"].append(float(total_time))

    # 转换为 DataFrame
    df = pd.DataFrame(data)
    return df

def parse_log_single_width(log_path, method):
    """
    解析其他方法的日志文件（每个宽度单独存储在一个文件中），提取测试准确率和训练时间。
    """
    test_acc = None
    train_time = 0.0
    with open(log_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            # 提取测试准确率
            if method == "FedProx":
                # 修正正则表达式以匹配整数或浮点数百分比
                acc_match = re.search(r'Accuracy: \d+/\d+ \((\d+\.?\d*)%\)', line)
                if acc_match:
                    test_acc = float(acc_match.group(1))
            elif method == "FedFOMO":
                acc_match = re.search(r'acc_after_train: (\d+\.\d+)', line)
                if acc_match:
                    test_acc = float(acc_match.group(1))
            
            # 提取训练时间（确保时间行匹配实际日志格式）
            time_match = re.search(r'Simulated round time cost: (\d+\.\d+)', line)
            if time_match:
                current_time = float(time_match.group(1)) / 1000  # 转换为10^3秒
                train_time += current_time  # 累加时间
    return test_acc, train_time

def generate_table(df_FLSNN, df_RandomSNN, FedProx_logs_dir, FedFOMO_logs_dir):
    """
    生成表格，展示不同框架在不同宽度下的准确率和加速比。
    """
    # 提取宽度列表
    widths = df_FLSNN["width"].unique()

    # 创建表格数据字典
    table_data = {
        "Applications": [],
        "Models": [],
        "FLSNN Accuracy": [],
        "FLSNN Speedup": [],
        "RandomSNN Accuracy": [],
        "RandomSNN Speedup": [],
        "FedProx Accuracy": [],
        "FedProx Speedup": [],
        "FedFOMO Accuracy": [],
        "FedFOMO Speedup": []
    }

    # 填充表格数据
    for width in widths:
        # 获取FedSNN和RandomSNN的数据
        FLSNN_data = df_FLSNN[df_FLSNN["width"] == width]
        RandomSNN_data = df_RandomSNN[df_RandomSNN["width"] == width]

        # 构造正确的日志文件路径（示例中使用下划线分隔，根据实际路径调整）
        FedProx_log_path = os.path.join(FedProx_logs_dir, f'width({width})_noniid_het/server.log')  # 假设路径中使用下划线
        FedFOMO_log_path = os.path.join(FedFOMO_logs_dir, f'width({width})_noniid/server.log')      # 调整路径格式

        FedProx_acc, FedProx_time = parse_log_single_width(FedProx_log_path, "FedProx")
        FedFOMO_acc, FedFOMO_time = parse_log_single_width(FedFOMO_log_path, "FedFOMO")

        # 计算加速比（其他方法时间 / FLSNN时间）
        flsnn_time = FLSNN_data["time"].iloc[-1] if not FLSNN_data.empty else 1  # 避免除以零
        speedup_RandomSNN = (RandomSNN_data["time"].iloc[-1] / flsnn_time) if not RandomSNN_data.empty else 0
        speedup_FedProx = (FedProx_time / flsnn_time) if FedProx_time > 0 else 0
        speedup_FedFOMO = (FedFOMO_time / flsnn_time) if FedFOMO_time > 0 else 0

        # 填充表格
        table_data["Applications"].append("noniid")  
        table_data["Models"].append(f"AlexNet {width}")    
        table_data["FLSNN Accuracy"].append(FLSNN_data["test_acc"].iloc[-1])
        table_data["FLSNN Speedup"].append(1.0)
        table_data["RandomSNN Accuracy"].append(RandomSNN_data["test_acc"].iloc[-1] if not RandomSNN_data.empty else 0)
        table_data["RandomSNN Speedup"].append(speedup_RandomSNN)
        table_data["FedProx Accuracy"].append(FedProx_acc if FedProx_acc is not None else 0)
        table_data["FedProx Speedup"].append(speedup_FedProx)
        table_data["FedFOMO Accuracy"].append(FedFOMO_acc if FedFOMO_acc is not None else 0)
        table_data["FedFOMO Speedup"].append(speedup_FedFOMO)

    # 转换为 DataFrame
    table_df = pd.DataFrame(table_data)
    return table_df

# 解析FLSNN和RandomSNN的日志文件
df_FLSNN = parse_log('../runs/cifar10/FedSNN_clustered_noniid_het/server.log')
df_RandomSNN = parse_log('../runs/cifar10/RandSNN_clustered_noniid_het/server.log')

# 指定其他方法的日志文件夹（根据实际路径调整）
FedProx_logs_dir = '/data/yxli/FLamingo_examples/FedProx/runs/cifar10'
FedFOMO_logs_dir = '/data/yxli/FLamingo_examples/FedFOMO/runs/cifar10'

# 生成表格
table_df = generate_table(df_FLSNN, df_RandomSNN, FedProx_logs_dir, FedFOMO_logs_dir)

# 打印并保存表格
print(table_df)
output_dir = "../paper_tables"
os.makedirs(output_dir, exist_ok=True)
table_df.to_csv(os.path.join(output_dir, "results_table.csv"), index=False)