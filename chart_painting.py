import re
import pandas as pd

def parse_log(log_path):
    # 初始化数据存储
    data = {
        "epoch": [],
        "test_loss": [],
        "test_acc": [],
        "avg_train_loss": [],
        "avg_test_acc": [],
        "avg_test_loss": []
    }

    with open(log_path, "r") as f:
        lines = f.readlines()
        current_epoch = 0
        for line in lines:
            # 提取轮次（如 "==========End of Round 1=========="）
            if "End of Round" in line:
                match = re.search(r"Round (\d+)", line)
                if match:
                    current_epoch = int(match.group(1))
                    data["epoch"].append(current_epoch)

            # 提取全局测试结果
            if "Test set: Average loss" in line:
                test_loss = re.search(r"Average loss: ([\d.]+)", line).group(1)
                test_acc = re.search(r"Accuracy: (\d+)/", line).group(1)
                data["test_loss"].append(float(test_loss))
                data["test_acc"].append(float(test_acc) / 100)  # 转换为百分比

            # 提取聚合后的平均信息
            if "Avg global info" in line:
                next_line1 = lines[lines.index(line) + 1]
                next_line2 = lines[lines.index(line) + 2]
                next_line3 = lines[lines.index(line) + 3]
                avg_train_loss = re.search(r"train loss ([\d.]+)", next_line1).group(1)
                avg_test_acc = re.search(r"test acc ([\d.]+)", next_line2).group(1)
                avg_test_loss = re.search(r"test loss ([\d.]+)", next_line3).group(1)
                data["avg_train_loss"].append(float(avg_train_loss))
                data["avg_test_acc"].append(float(avg_test_acc))
                data["avg_test_loss"].append(float(avg_test_loss))

    # 转换为 DataFrame
    df = pd.DataFrame(data)
    return df

# 解析日志
df = parse_log("./runs/cifar10/2025-04-01-13-56-02/server.log")


import os
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')  # 这一步没用, matplotlib的版本不符

def plot_training_curves(df):
    plt.figure(figsize=(12, 8))

    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(df["epoch"], df["test_loss"], label="Test Loss")
    plt.plot(df["epoch"], df["avg_test_loss"], label="Avg Test Loss")
    plt.plot(df["epoch"], df["avg_train_loss"], label="Avg Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(2, 1, 2)
    plt.plot(df["epoch"], df["test_acc"], label="Test Accuracy (%)")
    plt.plot(df["epoch"], df["avg_test_acc"], label="Avg Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # 保存
    output_dir = "./charts"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "chart-2025-04-01-13-56-02.png")
    plt.savefig(output_file)
    plt.show()

# 绘图
plot_training_curves(df)