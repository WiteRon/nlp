#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "how to use: ./run.sh <python_script.py>"
    exit 1
fi

PYTHON_SCRIPT=$1

# 提交作业，并将输出和错误重定向到指定文件
sbatch --account=mscbdt2024 <<EOF
#!/bin/bash
#SBATCH --job-name=T5_tune_all     # 为你的作业创建一个短名称
#SBATCH --nodes=1                   # 请求的节点数
#SBATCH --gpus=1                    # 每个节点的 GPU 数量
#SBATCH --time=01:00:00             # 设置为 1 小时
#SBATCH --partition=normal          # 提交的分区
#SBATCH --account=mscbdt2024        # 指定账户
#SBATCH --output=/home/zshiap/start/result/slurm_%j.out       # 输出文件，%j 将被替换为作业 ID
#SBATCH --error=/home/zshiap/start/result/slurm_%j.err        # 错误文件，%j 将被替换为作业 ID

module purge                     # 清除环境模块
module load Anaconda3/2023.09-0  # 加载所需模块

# 使用 conda 环境运行 Python 脚本
conda run -n nlp python $PYTHON_SCRIPT  
             
EOF

echo "Successfully submitted job. You can monitor output in slurm_<job_id>.out and errors in slurm_<job_id>.err"