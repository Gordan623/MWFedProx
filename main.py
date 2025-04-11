import sys
sys.path.append(".") # Adds higher directory to python modules path.
sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append('/data/yxli/FLamingo/')  # 注意, 这里建议修改为FLamingo的绝对路径, 不然会默认导入pytorch环境里的FLamingo
from FLamingo.core.runner import Runner
# from FLamingo.datasets import generate_cifar10


if __name__ == "__main__":
    runner = Runner(cfg_file='./config.yaml')
    runner.run()