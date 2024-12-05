#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "how to use: ./run.sh <python_script.py>"
    exit 1
fi

PYTHON_SCRIPT=$1

# �ύ��ҵ����������ʹ����ض���ָ���ļ�
sbatch --account=mscbdt2024 <<EOF
#!/bin/bash
#SBATCH --job-name=T5_tune_all     # Ϊ�����ҵ����һ��������
#SBATCH --nodes=1                   # ����Ľڵ���
#SBATCH --gpus=1                    # ÿ���ڵ�� GPU ����
#SBATCH --time=01:00:00             # ����Ϊ 1 Сʱ
#SBATCH --partition=normal          # �ύ�ķ���
#SBATCH --account=mscbdt2024        # ָ���˻�
#SBATCH --output=/home/zshiap/start/result/slurm_%j.out       # ����ļ���%j �����滻Ϊ��ҵ ID
#SBATCH --error=/home/zshiap/start/result/slurm_%j.err        # �����ļ���%j �����滻Ϊ��ҵ ID

module purge                     # �������ģ��
module load Anaconda3/2023.09-0  # ��������ģ��

# ʹ�� conda �������� Python �ű�
conda run -n nlp python $PYTHON_SCRIPT  
             
EOF

echo "Successfully submitted job. You can monitor output in slurm_<job_id>.out and errors in slurm_<job_id>.err"