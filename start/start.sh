salloc --account=mscbdt2024 --partition=normal --job-name=nlp --time=00:120:00 --gpus=1 --nodes=1
srun --pty /bin/bash
module load cuda12.2
module load Anaconda3

