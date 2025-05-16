import re
import pandas as pd
import matplotlib.pyplot as plt
import os

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
    cumulative_time = 0.0  # 用于累加时间

    with open(log_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            # 匹配模拟轮次时间成本的行
            time_match = re.search(r'Simulated round time cost: (\d+\.\d+)', line)
            if time_match:
                current_time = float(time_match.group(1))/1000  # 转换为10^3秒
                cumulative_time += current_time  # 累加时间
            # 匹配宽度、准确率和损失值的行
            width_match = re.search(r'width:(\d+\.\d+)', line)
            if width_match and cumulative_time is not None:
                width = width_match.group(1)
                test_acc = re.search(r'test acc: (\d+\.\d+)', line).group(1)
                test_loss = re.search(r'test_loss: (\d+\.\d+)', line).group(1)
                data["width"].append(width)
                data["test_acc"].append(float(test_acc))
                data["test_loss"].append(float(test_loss))
                data["time"].append(float(cumulative_time))

    # 转换为 DataFrame
    df = pd.DataFrame(data)
    return df

def compare_and_plot(df1, df2, label1, label2, output_dir="../paper_charts"):
    """
    对比两个方法的准确率、损失和加速比，并绘制相应的图表。
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 提取宽度列表
    widths = df1["width"].unique()

    # 准确率对比图
    for width in widths:
        plt.figure(figsize=(10, 6))
        # 提取当前宽度的数据
        df1_width = df1[df1["width"] == width]
        df2_width = df2[df2["width"] == width]
        plt.plot(df1_width["time"], df1_width["test_acc"], label=f'{label1} Width {width}')
        plt.plot(df2_width["time"], df2_width["test_acc"], label=f'{label2} Width {width}')

        # 获取首个收敛方法的最后一个时间点作为x轴范围, 此处为1号的FLSNN
        max_x = df1_width["time"].iloc[-1] if not df1_width.empty else 0

        plt.title(f'Accuracy for width {width}')
        plt.xlabel('Time ($10^3$s)')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, max_x)  # 设置横坐标范围为0到最大时间
        plt.ylim(0, 100)  # 设置纵坐标范围为0到100
        plt.savefig(os.path.join(output_dir, f"FL&R_acc_comparison_width({width}).png"))
        plt.show()

    # 损失对比图
    for width in widths:
        plt.figure(figsize=(10, 6))
        df1_width = df1[df1["width"] == width]
        df2_width = df2[df2["width"] == width]
        plt.plot(df1_width["time"], df1_width["test_loss"], label=f'{label1} Width {width}')
        plt.plot(df2_width["time"], df2_width["test_loss"], label=f'{label2} Width {width}')

        max_x = 5

        plt.title(f'Loss for width {width}')
        plt.xlabel('Time ($10^3$ s)')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, max_x)  # 设置横坐标范围为0到最大时间
        plt.savefig(os.path.join(output_dir, f"FL&R_loss_comparison_width({width}).png"))
        plt.show()

    # 计算加速比
    # 获取最后一次的时间作为总时间
    time1 = df1["time"].iloc[-1] if not df1.empty else 0
    time2 = df2["time"].iloc[-1] if not df2.empty else 0
    speedup = time2 / time1 if time1 != 0 else 0

    # 加速比图
    plt.figure(figsize=(10, 6))
    plt.bar(['FLSNN vs RandomSNN'], [speedup])
    plt.title('Speedup of FLSNN over RandomSNN')
    plt.ylabel('Speedup')
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_dir, "FL&R_Speedup.png"))
    plt.show()

# 解析两个日志文件
flsnn_df = parse_log('../runs/cifar10/FedSNN_clustered_noniid_het/server.log')
randomsnn_df = parse_log('../runs/cifar10/RandSNN_clustered_noniid_het/server.log')

# 绘制对比图
compare_and_plot(flsnn_df, randomsnn_df, 'FLSNN', 'RandomSNN')