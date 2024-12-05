#!/bin/bash
#SBATCH --job-name=test             # 为你的作业创建一个短名称
#SBATCH --nodes=1                   # 请求的节点数
#SBATCH --gpus=1                    # 每个节点的 GPU 数量
#SBATCH --time=01:00:00           # 设置为 1 小时
#SBATCH --partition=normal          # 提交的分区（normal）
#SBATCH --account=mscbdt2024        # 指定账户



module purge                        # 清除提交时继承的环境模块
module load Anaconda3    # 加载所需的模块

source /home/zshiap/.bashrc
conda activate nlp


python /home/zshiap/NLP/ind_pro/finetune/roberta/roberta_tune.py  # 运行你的 Python 脚本