import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("..")

# FLSNN/RandomSNN日志解析（多宽度单文件）
def parse_multiwidth_log(log_path):
    data = {
        "epoch": [],
        "width_0.25_test_acc": [], "width_0.25_test_loss": [],
        "width_0.5_test_acc": [], "width_0.5_test_loss": [],
        "width_0.75_test_acc": [], "width_0.75_test_loss": [],
        "width_1.0_test_acc": [], "width_1.0_test_loss": [],
        "avg_test_acc": [], "avg_test_loss": [],
        "time_cost": []  # 新增：存储每轮时间成本
    }

    with open(log_path, "r") as f:
        current_epoch = 0
        for line in f:
            if "End of Round" in line:
                current_epoch = int(re.search(r"Round (\d+)", line).group(1))
                data["epoch"].append(current_epoch)
            
            # 新增：提取时间成本
            if "Simulated round time cost" in line:
                time_cost = float(re.search(r"cost: (\d+\.\d+)", line).group(1))
                data["time_cost"].append(time_cost)

            for width in ["0.25", "0.5", "0.75", "1.0"]:
                if f"width:{width}" in line:
                    acc = float(re.search(r"test acc: (\d+\.\d+)", line).group(1))
                    loss = float(re.search(r"test_loss: (\d+\.\d+)", line).group(1))
                    data[f"width_{width}_test_acc"].append(acc)
                    data[f"width_{width}_test_loss"].append(loss)
            
            if "avg test_acc" in line:
                data["avg_test_acc"].append(float(re.search(r"avg test_acc: (\d+\.\d+)", line).group(1)))
            if "avg test_loss" in line:
                data["avg_test_loss"].append(float(re.search(r"avg test_loss: (\d+\.\d+)", line).group(1)))

    # 新增：计算累计时间
    df = pd.DataFrame(data)
    df["cumulative_time"] = df["time_cost"].cumsum()
    return df


# FedProx/FedFOMO日志解析（单宽度多文件）
def parse_singlewidth_logs(log_dir, widths, method):
    dfs = []
    for width in widths:
        if method == "FedProx":
            log_path = f"{log_dir}/noniid_iter10/server.log" 
        elif method == "FedFOMO":
            log_path = f"{log_dir}/noniid_iter10/server.log"
        else:
            continue
            
        # 初始化数据结构
        data = {
            "epoch": [],
            "avg_test_acc": [],
            "time_cost": []
        }
        
        current_epoch = 0
        current_acc = None
        
        with open(log_path, "r") as f:
            for line in f:
                # 提取轮次
                if "End of Round" in line:
                    current_epoch = int(re.search(r"Round (\d+)", line).group(1))
                
                # 提取准确率
                if method == "FedProx":
                    if "test acc" in line and "avg_bf_test_acc" not in line:
                        acc_match = re.search(r"test acc (\d+\.\d+)", line)
                        if acc_match:
                            current_acc = float(acc_match.group(1))
                
                elif method == "FedFOMO":
                    if "acc_after_train" in line:
                        current_acc = float(re.search(r"acc_after_train: (\d+\.\d+)", line).group(1))
                
                # 提取时间成本
                if "Simulated round time cost" in line:
                    time_cost = float(re.search(r"cost: (\d+\.\d+)", line).group(1))
                    if current_acc is not None:  # 确保有准确率数据
                        data["epoch"].append(current_epoch)
                        data["avg_test_acc"].append(current_acc)
                        data["time_cost"].append(time_cost)
                        current_acc = None  # 重置准确率
        
        # 创建DataFrame并计算累计时间
        df = pd.DataFrame(data)
        if not df.empty:
            df["cumulative_time"] = df["time_cost"].cumsum()
            df["width"] = width
            dfs.append(df)
    
    return pd.concat(dfs) if dfs else pd.DataFrame()

def plot_curves(flsnn_df, rand_df, fedprox_df, fedfomo_df):
    plt.rcParams.update({
        'font.size': 16,          # 基础字体大小
        'axes.titlesize': 20,     # 标题字体大小
        'axes.labelsize': 18,     # 坐标轴标签字体大小
        'xtick.labelsize': 18,    # X轴刻度字体大小
        'ytick.labelsize': 18,    # Y轴刻度字体大小
        'legend.fontsize': 16,    # 图例字体大小
        'legend.loc': 'upper center',  # 图例位置设置
        # 'legend.bbox_to_anchor': (0.5, -0.15),  # 图例在图表下方
        # 'legend.ncol': 4  # 图例分4列显示
        # 'font.weight': 'bold',    # 字体加粗
        # 'axes.labelweight': 'bold',  # 坐标轴标签加粗
        # 'axes.titleweight': 'bold'   # 标题加粗
    })
    # 创建输出目录
    output_dir = "../../paper_charts/noniid"
    os.makedirs(output_dir, exist_ok=True)
    
    # 方法样式配置（添加markevery参数）
    method_styles = {
        "FLSNN": {"color": "blue", "ls": "-", "marker": "o", "markevery": 10},
        "RandomSNN": {"color": "orange", "ls": ":", "marker": "s", "markevery": 10},
        "FedProx": {"color": "green", "ls": "--", "marker": "^", "markevery": 10},
        "FedFOMO": {"color": "red", "ls": "-.", "marker": "d", "markevery": 10}
    }
    
    # 遍历每个宽度生成独立图表
    for width in ["0.75"]:
        # 提取当前宽度的数据
        fedprox_width = fedprox_df[fedprox_df["width"] == width]
        fedfomo_width = fedfomo_df[fedfomo_df["width"] == width]
        
        # ========== 准确率图表 ==========
        plt.figure(figsize=(10, 6))
        # FLSNN（多宽度单文件）
        flsnn_sub = flsnn_df[flsnn_df["epoch"] <= 500]  # 对齐轮次
        plt.plot(flsnn_sub["cumulative_time"], flsnn_sub["avg_test_acc"], 
                label="FLSNN", **method_styles["FLSNN"])
        
        # RandomSNN（多宽度单文件）
        rand_sub = rand_df[rand_df["epoch"] <= 500]
        plt.plot(rand_sub["cumulative_time"], rand_sub["avg_test_acc"], 
                label="RandomSNN", **method_styles["RandomSNN"])
        
        # FedProx（单宽度多文件）
        fedprox_sub = fedprox_df[(fedprox_df["width"] == width) & (fedprox_df["epoch"] <= 200)]
        if not fedprox_sub.empty:
            plt.plot(fedprox_sub["cumulative_time"], fedprox_sub["avg_test_acc"], 
                    label="FedProx", **method_styles["FedProx"])
        
        # FedFOMO（单宽度多文件）
        fedfomo_sub = fedfomo_df[(fedfomo_df["width"] == width) & (fedfomo_df["epoch"] <= 200)]
        if not fedfomo_sub.empty:
            plt.plot(fedfomo_sub["cumulative_time"], fedfomo_sub["avg_test_acc"], 
                    label="FedFOMO", **method_styles["FedFOMO"])
        
        plt.title(f"Local iteration = 10")
        plt.xlabel("Time Cost (s)")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        # 新增：设置X轴范围0-10000秒
        plt.xlim(0, 6000)
        # 新增：转换X轴刻度标签为千秒单位
        plt.gca().set_xticklabels([f"{int(x/1000)}" for x in plt.gca().get_xticks()])
        plt.xlabel("Cumulative Time Cost ($10^3$ s)")  # 更新标签
        plt.legend(
            loc='upper center',          # 初始定位
            bbox_to_anchor=(0.5, -0.15), # 下移25%图表高度
            ncol=4,                      # 4列排列
            # nrow=4,                      # 4行排列
            fontsize=16,                 # 字体大小
            frameon=False                # 去掉边框
        )
        plt.savefig(f"{output_dir}/AvgTestAccIter10.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # # ========== 损失值图表 ==========
        # plt.figure(figsize=(10, 6))
        # # FLSNN
        # plt.plot(flsnn_df["epoch"], flsnn_df[f"width_{width}_test_loss"], 
        #         label="FLSNN", **method_styles["FLSNN"])
        # # RandomSNN
        # plt.plot(rand_df["epoch"], rand_df[f"width_{width}_test_loss"], 
        #         label="RandomSNN", **method_styles["RandomSNN"])
        # # FedProx
        # if not fedprox_width.empty:
        #     plt.plot(fedprox_width["epoch"], fedprox_width["test_loss"], 
        #             label="FedProx", **method_styles["FedProx"])
        # # FedFOMO
        # if not fedfomo_width.empty:
        #     plt.plot(fedfomo_width["epoch"], fedfomo_width["test_loss"], 
        #             label="FedFOMO", **method_styles["FedFOMO"])
        
        # plt.title(f"Width {width} - Test Loss")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.grid(True)
        # plt.legend(loc="upper right")
        # plt.savefig(f"{output_dir}/width_{width}_loss.png", dpi=300, bbox_inches='tight')
        # plt.close()

# 主流程
if __name__ == "__main__":
    # 解析日志
    flsnn_df = parse_multiwidth_log("../../runs/cifar10/FLSNN_noniid_iter10/server.log")
    rand_df = parse_multiwidth_log("../../runs/cifar10/RandSNN_noniid_iter10/server.log")
    fedprox_df = parse_singlewidth_logs("/data/yxli/FLamingo_examples/FedProx/runs/cifar10", ["0.25", "0.5", "0.75", "1.0"], "FedProx")
    fedfomo_df = parse_singlewidth_logs("/data/yxli/FLamingo_examples/FedFOMO/runs/cifar10", ["0.25", "0.5", "0.75", "1.0"], "FedFOMO")
    
    # 绘图
    plot_curves(flsnn_df, rand_df, fedprox_df, fedfomo_df)
    print("图表已保存到", "../../paper_charts/noniid")