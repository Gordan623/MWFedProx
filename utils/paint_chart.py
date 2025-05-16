import re
import pandas as pd
import matplotlib.pyplot as plt
import os

import sys
sys.path.append("..")

def parse_log(log_path):
    # 初始化数据存储
    data = {
        "epoch": [],
        "width_0.25_test_acc": [],
        "width_0.25_test_loss": [],
        "width_0.5_test_acc": [],
        "width_0.5_test_loss": [],
        "width_0.75_test_acc": [],
        "width_0.75_test_loss": [],
        "width_1.0_test_acc": [],
        "width_1.0_test_loss": [],
        "avg_train_loss": [],
        "avg_test_loss": [],
        "avg_test_acc": []
    }

    with open(log_path, "r") as f:
        lines = f.readlines()
        current_epoch = 0
        in_round = False

        for i, line in enumerate(lines):
            # 提取轮次（如 "==========End of Round 495==========")
            if "End of Round" in line:
                match = re.search(r"Round (\d+)", line)
                if match:
                    current_epoch = int(match.group(1))
                    data["epoch"].append(current_epoch)
                    in_round = True

            # 提取宽度 0.25 的测试结果
            if "width:0.25" in line:
                test_acc = re.search(r"test acc: (\d+\.\d+)", line).group(1)
                test_loss = re.search(r"test_loss: (\d+\.\d+)", line).group(1)
                data["width_0.25_test_acc"].append(float(test_acc))
                data["width_0.25_test_loss"].append(float(test_loss))

            # 提取宽度 0.5 的测试结果
            if "width:0.5" in line:
                test_acc = re.search(r"test acc: (\d+\.\d+)", line).group(1)
                test_loss = re.search(r"test_loss: (\d+\.\d+)", line).group(1)
                data["width_0.5_test_acc"].append(float(test_acc))
                data["width_0.5_test_loss"].append(float(test_loss))

            # 提取宽度 0.75 的测试结果
            if "width:0.75" in line:
                test_acc = re.search(r"test acc: (\d+\.\d+)", line).group(1)
                test_loss = re.search(r"test_loss: (\d+\.\d+)", line).group(1)
                data["width_0.75_test_acc"].append(float(test_acc))
                data["width_0.75_test_loss"].append(float(test_loss))

            # 提取宽度 1.0 的测试结果
            if "width:1.0" in line:
                test_acc = re.search(r"test acc: (\d+\.\d+)", line).group(1)
                test_loss = re.search(r"test_loss: (\d+\.\d+)", line).group(1)
                data["width_1.0_test_acc"].append(float(test_acc))
                data["width_1.0_test_loss"].append(float(test_loss))

            # 提取聚合后的平均信息
            if "avg train_loss" in line:
                avg_train_loss = re.search(r"avg train_loss: (\d+\.\d+)", line).group(1)
                data["avg_train_loss"].append(float(avg_train_loss))

            if "avg test_loss" in line:
                avg_test_loss = re.search(r"avg test_loss: (\d+\.\d+)", line).group(1)
                data["avg_test_loss"].append(float(avg_test_loss))

            if "avg test_acc" in line:
                avg_test_acc = re.search(r"avg test_acc: (\d+\.\d+)", line).group(1)
                data["avg_test_acc"].append(float(avg_test_acc))

    # 转换为 DataFrame
    df = pd.DataFrame(data)
    return df

def plot_training_curves(df):
    plt.figure(figsize=(15, 10))

    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(df["epoch"], df["width_0.25_test_loss"], label="Width 0.25 Test Loss")
    plt.plot(df["epoch"], df["width_0.5_test_loss"], label="Width 0.5 Test Loss")
    plt.plot(df["epoch"], df["width_0.75_test_loss"], label="Width 0.75 Test Loss")
    plt.plot(df["epoch"], df["width_1.0_test_loss"], label="Width 1.0 Test Loss")
    plt.plot(df["epoch"], df["avg_test_loss"], label="Avg Test Loss", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Test Loss for Different Widths")
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(2, 1, 2)
    plt.plot(df["epoch"], df["width_0.25_test_acc"], label="Width 0.25 Test Accuracy")
    plt.plot(df["epoch"], df["width_0.5_test_acc"], label="Width 0.5 Test Accuracy")
    plt.plot(df["epoch"], df["width_0.75_test_acc"], label="Width 0.75 Test Accuracy")
    plt.plot(df["epoch"], df["width_1.0_test_acc"], label="Width 1.0 Test Accuracy")
    plt.plot(df["epoch"], df["avg_test_acc"], label="Avg Test Accuracy", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy for Different Widths")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # 保存图表
    output_dir = "../charts"  # 在MWFedProx/utils下运行
    # output_dir = "./charts"  # 在MWFedProx/下运行
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "RandSNN_clustered_noniid_het.png")
    plt.savefig(output_file)
    plt.show()

# 解析日志
df = parse_log("../runs/cifar10/RandSNN_clustered_noniid_het/server.log")  # 在MWFedProx/utils下运行
# df = parse_log("./runs/cifar10/2025-04-12-four-widths/server.log")  # 在MWFedProx/下运行

# 绘图
plot_training_curves(df)